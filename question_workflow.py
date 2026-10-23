"""LangGraph workflow: topic -> semantic search -> LLM writes new questions -> independent verification -> output.

    retrieve --> figures --+--> generate --> verify --+--> finalize
                           |        ^                   |
                           |        +-------------------+  (text questions, until enough pass)
                           +--> finalize                    (when figure questions already filled the request)

    figures: diagram questions built by code (figure_problems.py); at most 3 attempts each, then a text question instead.
                    ^                   |
                    +---- (not enough verified questions yet, retries left)

Usage:
    python question_workflow.py "Circles and tangents" -n 5
    python question_workflow.py "Heights and distances" -n 3 --type case_study --out out.json
"""
import argparse
import difflib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import diagram
import figure_problems
import mathcheck
from embed_to_vectordb import BUCKET, INDEX, clients, embed

LLM_MODEL_ID = os.getenv("BEDROCK_LLM_MODEL_ID", "apac.amazon.nova-pro-v1:0")  # writes the questions
JUDGE_MODEL_ID = os.getenv("BEDROCK_JUDGE_MODEL_ID", "apac.amazon.nova-lite-v1:0")  # checks them
DIAGRAM_MODEL_ID = os.getenv("BEDROCK_DIAGRAM_MODEL_ID", LLM_MODEL_ID)  # turns a question into a figure spec
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "5000"))  # Nova output limit is about 5K tokens per call
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
MAX_ROUNDS = int(os.getenv("MAX_GENERATION_ROUNDS", "3"))
GEN_CHUNK = 3  # questions per generation call (each is bilingual, so keep the reply short)
VERIFY_CHUNK = 4

bedrock, s3v = clients()
UNAVAILABLE: set[str] = set()  # models this account cannot call (e.g. Anthropic use-case form not submitted)


def converse(model_id: str, **kw):
    """bedrock.converse, but a checker/figure model the account cannot use falls back to the writer model."""
    from botocore.exceptions import ClientError

    if model_id in UNAVAILABLE:
        model_id = LLM_MODEL_ID
    try:
        return bedrock.converse(modelId=model_id, **kw)
    except ClientError as e:
        if model_id != LLM_MODEL_ID and e.response["Error"]["Code"] in ("ResourceNotFoundException", "AccessDeniedException"):
            UNAVAILABLE.add(model_id)
            print(f"{model_id} is not available to this account ({e.response['Error']['Message'][:80]}...); using {LLM_MODEL_ID}")
            return bedrock.converse(modelId=LLM_MODEL_ID, **kw)
        raise


class State(TypedDict, total=False):
    topic: str
    n: int
    question_type: str | None
    figures: bool  # True = every question must come with a verified figure
    examples: list[dict]  # questions found by semantic search
    candidates: list[dict]  # freshly generated, not yet verified
    accepted: list[dict]  # verified questions
    feedback: str  # why earlier candidates were rejected
    rounds: int
    fig_log: list[str]  # what happened on each diagram-question attempt (shown in the UI)


# ---------------------------------------------------------------- LLM helper
def ask_llm(model_id: str, system: str, user: str, tool_name: str, schema: dict, temperature: float) -> dict:
    """A Bedrock Converse call that returns JSON matching `schema`.

    Tries Nova's tool-use mode first; Nova often fails there ("invalid sequence as part of ToolUse"), so it then
    falls back to a plain-text reply containing the JSON, which Nova produces far more reliably."""
    from botocore.exceptions import ClientError

    try:
        resp = converse(
            model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": temperature},
            toolConfig={
                "tools": [{"toolSpec": {"name": tool_name, "description": "Return the result.", "inputSchema": {"json": schema}}}],
                "toolChoice": {"tool": {"name": tool_name}},
            },
        )
        for block in resp["output"]["message"]["content"]:
            if "toolUse" in block:
                return block["toolUse"]["input"]
    except ClientError as e:
        if e.response["Error"]["Code"] != "ModelErrorException":
            raise
        print(f"{tool_name}: tool-use error, falling back to plain JSON")
    text_user = (user + "\n\nReply with ONLY one JSON object inside a ```json code block, matching this JSON schema:\n"
                 + json.dumps(schema, ensure_ascii=False))
    last = None
    for _ in range(2):
        try:
            return extract_json(ask_text(model_id, system, text_user, temperature=0))
        except (json.JSONDecodeError, ValueError) as e:
            last = e
    raise RuntimeError(f"{tool_name}: no valid JSON from the model ({last})")


def ask_text(model_id: str, system: str, user: str, temperature: float) -> str:
    """Plain-text Converse call (no tool use). Used for figure specs, where Nova's tool-use mode often fails."""
    resp = converse(
        model_id,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": temperature},
    )
    return "".join(b.get("text", "") for b in resp["output"]["message"]["content"])


def extract_json(text: str):
    """The first JSON object in a reply (inside a ```json block or bare)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(raw)


STR = {"type": "string"}
QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_info": {**STR, "description": "Full English question text, with options (A)-(D) on new lines for MCQs."},
                    "question_info_hindi": {**STR, "description": "Faithful Hindi version of the same question and options."},
                    "topic": STR,
                    "topic_hindi": STR,
                    "subtopic": STR,
                    "subtopic_hindi": STR,
                    "question_type": STR,
                    "answer": {**STR, "description": "Option letter for MCQ / assertion-reason, otherwise the final answer."},
                    "solution": {**STR, "description": "Short worked solution."},
                    "answer_expression": {**STR, "description": "One arithmetic expression whose value is the numeric answer, e.g. 18*360/(2*pi*9) or 20*tan(60); use pi, sqrt, sin/cos/tan in degrees. Empty if the answer is not a number."},
                    "needs_figure": {"type": "boolean", "description": "true if the question uses a figure (our system draws it)."},
                    "diagram_description": {**STR, "description": "If needs_figure: every point, line, circle, angle and length the figure shows. Else empty."},
                },
                "required": ["question_info", "question_info_hindi", "topic", "topic_hindi", "subtopic", "subtopic_hindi",
                             "question_type", "answer", "solution"],
            },
        }
    },
    "required": ["questions"],
}
DIAGRAM_FORMAT = """DIAGRAM FORMAT (you never draw; our code draws exactly what you specify, then checks it):
__CONSTRUCTIONS__
Lists: "segments": [["P","A"], ...], "dashed": [...], "angle_marks": [{"at":"O","from":"P","to":"B","label":"115°"}],
"right_angles": [{"at":"A","from":"O","to":"P"}], "length_labels": [{"from":"P","to":"A","label":"10 m"}], "dots": ["O"].
Rules: every point named in the question must be defined; build points with constructions so every labelled angle is EXACT
(an angle labelled 115° must really measure 115°, or the figure is rejected); use tangent_from for tangents and mark the right angle.
Example for "PA is a tangent at A, AB is a diameter, ∠POB = 115°" (OP at 180°, OB at -65°, so OA at 115°, and OP = 3/cos 65° = 7.1):
{"circles":[{"name":"c","center":"O","radius":3}],
 "points":[{"name":"O","at":[0,0]},{"name":"B","on_circle":"c","angle_deg":-65},{"name":"A","on_circle":"c","angle_deg":115},
           {"name":"P","polar":{"from":"O","dist":7.1,"angle_deg":180}}],
 "segments":[["P","A"],["A","B"],["P","O"]],"angle_marks":[{"at":"O","from":"P","to":"B","label":"115°"}],
 "right_angles":[{"at":"A","from":"O","to":"P"}],"dots":["O"]}
""".replace("__CONSTRUCTIONS__", diagram.CONSTRUCTIONS)
BLIND_SCHEMA = {  # the checker solves WITHOUT seeing the key, so it cannot just agree with it
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "working": {**STR, "description": "Step-by-step calculation, done before choosing an answer."},
                    "my_answer": {**STR, "description": "Option letter (A-D) for multiple choice, otherwise the final answer."},
                    "expression": {**STR, "description": "One arithmetic expression whose value is the numeric answer (pi, sqrt, sin/cos/tan in degrees), or empty."},
                    "well_posed": {"type": "boolean", "description": "Exactly one option/answer is correct and nothing is missing or ambiguous."},
                    "issue": {**STR, "description": "What is wrong with the question, or empty."},
                },
                "required": ["index", "working", "my_answer", "well_posed", "issue"],
            },
        }
    },
    "required": ["results"],
}
MATCH_SCHEMA = {  # only for written answers, where equivalence needs judgement
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"index": {"type": "integer"}, "matches": {"type": "boolean"}, "reason": STR},
                "required": ["index", "matches", "reason"],
            },
        }
    },
    "required": ["results"],
}
LETTER_TYPES = {"mcq", "assertion_reason"}
FIGURE_WORDS = re.compile(r"given figure|shown in the figure|in the figure|figure below|दी गई आकृति|आकृति में", re.I)
# A solution that corrects itself means the writer was unsure; such keys were wrong in testing.
HEDGE_WORDS = re.compile(r"\bhowever\b|\bactually\b|miscalculation|\bcorrecting\b|not possible|\bbut considering\b|re-?evaluat|\bwait\b", re.I)


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


# ---------------------------------------------------------------- nodes
def retrieve(state: State) -> State:
    """Embed the topic and pull the closest stored questions."""
    kwargs = {"filter": {"question_type": state["question_type"]}} if state.get("question_type") else {}
    resp = s3v.query_vectors(
        vectorBucketName=BUCKET, indexName=INDEX,
        queryVector={"float32": embed(bedrock, state["topic"])},
        topK=TOP_K, returnMetadata=True, returnDistance=True, **kwargs,
    )
    examples = [{**v["metadata"], "distance": v["distance"]} for v in resp["vectors"]]
    print(f"retrieved {len(examples)} similar questions")
    return {"examples": examples, "accepted": [], "candidates": [], "feedback": "", "rounds": 0}


FIGURE_RULE = """
- EVERY question must use a figure: set needs_figure to true and write "In the given figure". Base each one on one of these
  standard figures, and use their point names (O = centre):
  two tangents PA, PB from external point P (angle APB or AOB given); tangent PA at A with diameter AB (angle POB given);
  chord AB with perpendicular OM from centre O (two of radius, OM, AB given); tower BC seen from point A (angle of elevation at A);
  tower BC seen from two points P (near) and Q (far); tower EC seen from A on the ground and B directly above A (AB given);
  triangle ABC with DE parallel to BC (D on AB, E on AC)."""


def split_hindi(q: dict) -> None:
    """Nova sometimes puts the Hindi translation inside the English text; move it to question_info_hindi."""
    text = q["question_info"]
    m = re.search(r"[\u0900-\u097F]", text)
    if not m:
        return
    cut = text.rfind("\n", 0, m.start()) + 1  # start of the line where Hindi begins
    english, hindi = text[:cut].strip(), text[cut:].strip()
    if english:
        q["question_info"] = english
        if not re.search(r"[\u0900-\u097F]", q.get("question_info_hindi", "")):
            q["question_info_hindi"] = hindi


def generate(state: State) -> State:
    need = min(state["n"] - len(state["accepted"]), GEN_CHUNK)
    shots = "\n\n".join(
        f"[Example {i + 1} | {e['question_type']} | {e['topic']} > {e['subtopic']}]\n{e['question_info']}"
        + (f"\n(Figure shown in the original: {e['image_description']})" if e.get("image_description") else "")
        for i, e in enumerate(state["examples"])
    )
    system = (
        "You are a senior exam paper-setter. You write new questions in the same exam, style, difficulty and format as the "
        "examples. Every question must be correct, unambiguous, self-contained and have exactly one right answer."
    )
    user = f"""Topic requested by the user: {state['topic']}

Existing questions retrieved from our question bank (for style and concept only):
{shots}

Write {need} NEW questions on the requested topic.
Rules:
- Similar to the examples in style and difficulty, but better: sharper wording, plausible distractors, a different angle or numbers. Never copy or lightly reword an example.
- Do NOT use figures here (diagram questions are made separately): set needs_figure to false, never refer to a figure, and describe every point, length and angle in words.
- question_info must be the question text itself in ENGLISH ONLY (Hindi goes only in question_info_hindi), never JSON.
- Write maths in plain Unicode (π, √, ², °, ×), never LaTeX such as \\( \\pi \\).
- If any example above had a figure, or the topic mentions a figure or diagram, at least one of your questions must set needs_figure to true.
- Calculators are not allowed in this exam: in trigonometry use only 0°, 30°, 45°, 60°, 90°, and choose numbers that give exact or simple answers.
- question_type must be one of: mcq, assertion_reason, very_short_answer, short_answer, long_answer, case_study.{(' Use only: ' + state['question_type'] + '.') if state.get('question_type') else ' Prefer the types seen in the examples.'}
- For mcq write the options on new lines as (A) ... (B) ... (C) ... (D) ...; for assertion_reason include the Assertion (A) and Reason (R) lines and the four standard options.
- Give a Hindi version of each question in the same wording style as CBSE Hindi papers; keep maths notation and option labels unchanged.
- Work out the correct answer FIRST, then build the options around it: exactly one option must equal the correct answer, and the distractors must be clearly wrong values.
- Double-check every number in the solution. The solution must be clean and correct, with no "correction" or second attempt inside it.
- answer: option letter for mcq/assertion_reason, otherwise the final answer. solution: short worked steps.
- answer_expression: if the answer is a number, one arithmetic expression that computes it; our code evaluates it and it must equal the keyed option.

{('Problems found in the previous attempt, avoid them: ' + state['feedback']) if state.get('feedback') else ''}"""
    try:
        out = ask_llm(LLM_MODEL_ID, system, user, "submit_questions", QUESTION_SCHEMA, temperature=0.7)
    except Exception as e:
        print(f"writer failed this round: {e}")
        out = {"questions": []}
    items = out.get("questions", [])
    if isinstance(items, str):  # Nova sometimes returns the array as a JSON string
        try:
            items = json.loads(items)
        except json.JSONDecodeError:
            items = []
    required = QUESTION_SCHEMA["properties"]["questions"]["items"]["required"]
    good = [q for q in items if isinstance(q, dict) and all(q.get(k) for k in required)]
    if len(good) < len(items):
        bad_keys = [sorted(q) if isinstance(q, dict) else type(q).__name__ for q in items if q not in good]
        print(f"dropped {len(items) - len(good)} malformed item(s) from the writer; keys were: {bad_keys}")
    out["questions"] = good
    out["questions"] = [q for q in out["questions"] if not q["question_info"].lstrip().startswith(("{", "["))]
    for q in out["questions"]:
        split_hindi(q)
    if state.get("question_type"):  # the user asked for one type; drop anything else
        kept = [q for q in out["questions"] if q.get("question_type") == state["question_type"]]
        if len(kept) < len(out["questions"]):
            print(f"  dropped {len(out['questions']) - len(kept)} question(s) of the wrong type")
        out["questions"] = kept
    # Same question with new numbers is a valid variation of a bank question, so only exact copies of examples are dropped;
    # among the new questions, near-identical repeats are dropped too.
    bank = {norm(e["question_info"]) for e in state["examples"]}
    seen = [norm(q["question_info"]) for q in state["accepted"]]
    fresh = []
    for q in out["questions"]:
        t = norm(q["question_info"])
        if t in bank:
            print("  dropped: exact copy of a bank question")
        elif any(difflib.SequenceMatcher(None, t, x).ratio() >= 0.92 for x in seen):
            print("  dropped: near-duplicate of another new question")
        else:
            fresh.append(q)
            seen.append(t)
    print(f"round {state['rounds'] + 1}: generated {len(out['questions'])} ({len(fresh)} kept)")
    for q in fresh:
        q["diagram"] = None
        # free-form figures from the writer are off by default; diagram questions come from figure_questions()
        if os.getenv("FREEFORM_FIGURES") and (q.get("needs_figure") or FIGURE_WORDS.search(q["question_info"])):
            q["diagram"], errs = draw(q)
            if errs:
                q["diagram_error"] = "; ".join(errs[:3])
            shown = q["diagram"] or q.pop("diagram_draft", None)
            q.pop("diagram_draft", None)
            if shown:
                try:
                    q["diagram_svg"] = diagram.render_svg(shown)
                except Exception:  # a broken draft just shows no picture
                    pass
    return {"candidates": fresh, "rounds": state["rounds"] + 1}


def draw(q: dict, tries: int = 3):
    """Ask the writer model for the figure spec of ONE question; check it in code and feed errors back."""
    system = "You turn a geometry question into a precise figure specification. You never draw; our code draws exactly what you specify."
    errors = []
    for _ in range(tries):
        user = f"""QUESTION:
{q['question_info']}

FIGURE MUST SHOW: {q.get('diagram_description', '')}

{diagram.TEMPLATE_DOC}
You MUST use a template whenever the figure is one of those (templates are drawn exactly; free-form figures usually fail).
Rename points with "names" if the question uses other letters, and put every given angle/length in "labels".

Only if truly no template fits, use the free format below. In it every radius, dist and angle_deg must be a NUMBER, never a letter:
{DIAGRAM_FORMAT}
Reply with ONLY the figure as one JSON object inside a ```json code block.{(' Your previous figure was rejected: ' + '; '.join(errors) + '. Fix exactly these problems.') if errors else ''}"""
        try:
            spec = extract_json(ask_text(DIAGRAM_MODEL_ID, system, user, temperature=0.2))
            raw_spec = json.dumps(spec, ensure_ascii=False)[:400]
        except Exception as e:
            errors = [f"the reply did not contain valid JSON ({e})"]
            continue
        try:  # any problem in a figure only rejects that figure, never the whole run
            if isinstance(spec, dict) and spec.get("template"):
                try:
                    spec = diagram.from_template(spec)
                except (ValueError, TypeError) as e:
                    errors = [f"template problem: {e}"]
                    continue
            if not isinstance(spec, dict) or not spec.get("points"):
                errors = ["diagram_json must be an object with a points list"]
                continue
            spec = diagram.complete(spec)  # draw any line a mark or label sits on
            q["diagram_draft"] = spec  # shown in the UI even if rejected
            errors = diagram.check(spec, q["question_info"])
        except Exception as e:
            errors = [f"figure could not be built ({type(e).__name__}: {e})"]
            continue
        if not errors:
            return spec, []
    print(f"figure failed after {tries} tries: {errors[:2]}")
    if os.getenv("DEBUG_FIGURES"):
        print(f"  question: {q['question_info'][:160]!r}\n  last figure spec: {locals().get('raw_spec', '')}")
    return None, errors


def figure_facts(c: dict) -> str:
    d = c.get("diagram") if isinstance(c.get("diagram"), dict) else {}
    if not d.get("points"):
        return ""
    marks = [f"angle {m['from']}{m['at']}{m['to']} = {m['label']}" for m in d.get("angle_marks", []) if m.get("label")]
    marks += [f"angle {m['from']}{m['at']}{m['to']} = 90°" for m in d.get("right_angles", [])]
    marks += [f"{m['from']}{m['to']} = {m['label']}" for m in d.get("length_labels", []) if m.get("label")]
    return f"\nFIGURE: {diagram.describe(d)}\nMarked in the figure: {'; '.join(marks) or 'nothing'}"


def verify(state: State) -> State:
    """Rule checks, then a blind solve by the checker model (no key shown), then compare its answer with the key."""
    cands = state["candidates"]
    if not cands:  # the writer produced nothing new this round; keep the state shape the same for the UI
        return {"candidates": [], "accepted": state["accepted"], "feedback": "the writer produced no new (non-duplicate) questions"}
    problems, todo = [], []
    for i, c in enumerate(cands):  # cheap rule checks before spending an LLM call
        has_fig = bool(isinstance(c.get("diagram"), dict) and c["diagram"].get("points"))
        fig_errors = [c["diagram_error"]] if c.get("diagram_error") else (
            diagram.check(c["diagram"], c["question_info"]) if has_fig else [])
        if fig_errors:
            problems.append(f"'{c['question_info'][:60]}...': diagram is wrong: {'; '.join(fig_errors[:3])}")
        elif not has_fig and (FIGURE_WORDS.search(c["question_info"]) or FIGURE_WORDS.search(c["question_info_hindi"])):
            problems.append(f"'{c['question_info'][:60]}...': refers to a figure but gives no diagram")
        elif re.search(r"[\u0900-\u097F]", c["question_info"]):
            problems.append(f"'{c['question_info'][:60]}...': the English question contains Hindi text (Hindi belongs in question_info_hindi)")
        elif re.search(r"^\s*(Points|Lines|Segments|Circles)\s*:", c["question_info"], re.M | re.I):
            problems.append(f"'{c['question_info'][:60]}...': the question text contains figure-spec notes instead of a clean question")
        elif HEDGE_WORDS.search(c["solution"]):
            problems.append(f"'{c['question_info'][:60]}...': the solution contradicts itself, so the key is unreliable")
        else:
            todo.append(i)

    # 1) Blind solve: the checker never sees the key or the solution.
    system = "You are a strict exam reviewer. Solve each question carefully from scratch, showing your working."
    blind = {}
    for start in range(0, len(todo), VERIFY_CHUNK):
        chunk = todo[start:start + VERIFY_CHUNK]
        listing = "\n\n".join(f"#{i}\n{cands[i]['question_info']}" + figure_facts(cands[i]) for i in chunk)
        user = f"""Solve each question below (keep the # number as index).
- working: your full calculation first.
- my_answer: the option letter (A-D) for multiple choice / assertion-reason questions, otherwise the final answer.
- expression: if the answer is a number, ONE arithmetic expression that computes it (e.g. 18*360/(2*pi*9)); our code evaluates it exactly.
- well_posed: false if no option is correct, more than one is correct, or information is missing or contradictory.
- issue: what is wrong with the question, or an empty string.

{listing}"""
        try:
            results = ask_llm(JUDGE_MODEL_ID, system, user, "submit_solutions", BLIND_SCHEMA, temperature=0).get("results", [])
        except Exception as e:  # these questions stay unchecked, so they are rejected below
            print(f"checker batch failed: {e}")
            results = []
        for r in results:
            if isinstance(r, dict) and isinstance(r.get("index"), int):
                blind[r["index"]] = r

    # 2) Compare with the key: letters in code, written answers with one more judge call.
    def letter(s):
        m = re.search(r"\b([A-D])\b", s.upper().replace("(", " ").replace(")", " "))
        return m.group(1) if m else None

    matches, written = {}, [i for i in todo if i in blind and cands[i]["question_type"] not in LETTER_TYPES]
    math_problem = {}
    for i in todo:
        if i in blind and cands[i]["question_type"] in LETTER_TYPES:
            c, key = cands[i], letter(cands[i]["answer"])
            mine = letter(blind[i]["my_answer"])
            # Exact arithmetic beats model arithmetic: evaluate both formulas and map them onto the numeric options.
            wv = mathcheck.safe_eval(c.get("answer_expression", ""))
            wm = mathcheck.matching_options(c["question_info"], wv) if wv is not None else None
            if wm is not None and wm != [key]:
                math_problem[i] = f"the key's own formula gives {wv:.4g}, which matches option(s) {wm or 'none'}, not {key}"
            cv = mathcheck.safe_eval(blind[i].get("expression", ""))
            cm = mathcheck.matching_options(c["question_info"], cv) if cv is not None else None
            if cm is not None:
                if len(cm) != 1:
                    math_problem[i] = f"the checker's formula gives {cv:.4g}, which matches option(s) {cm or 'none'}"
                mine = cm[0] if len(cm) == 1 else None
            matches[i] = i not in math_problem and mine is not None and mine == key
    if written:
        listing = "\n\n".join(f"#{i}\nQUESTION:\n{cands[i]['question_info']}\nANSWER 1: {cands[i]['answer']}\n"
                              f"ANSWER 2: {blind[i]['my_answer']}" for i in written)
        user = f"For each item, are ANSWER 1 and ANSWER 2 mathematically equivalent final answers? Keep the # number as index.\n\n{listing}"
        try:
            results = ask_llm(JUDGE_MODEL_ID, "You compare final answers strictly.", user, "submit_matches", MATCH_SCHEMA, temperature=0).get("results", [])
        except Exception as e:
            print(f"answer comparison failed: {e}")
            results = []
        for r in results:
            if isinstance(r, dict) and isinstance(r.get("index"), int):
                matches[r["index"]] = bool(r.get("matches"))

    accepted = list(state["accepted"])
    for i in todo:
        r = blind.get(i)
        if r and r["well_posed"] and matches.get(i):
            accepted.append(cands[i])
        elif not r:
            problems.append(f"'{cands[i]['question_info'][:60]}...': not checked")
        else:
            why = math_problem.get(i) or r["issue"] or f"checker got {r['my_answer']!r} but key says {cands[i]['answer']!r}"
            problems.append(f"'{cands[i]['question_info'][:60]}...': {why}")
    print(f"verified {len(accepted) - len(state['accepted'])}/{len(cands)}; accepted so far {len(accepted)}/{state['n']}")
    return {"accepted": accepted[: state["n"]], "feedback": "; ".join(problems), "candidates": []}


FIG_TRIES = 3  # attempts per diagram question before falling back to a text question
FIG_LOG = Path(__file__).with_name("logs") / "figure_attempts.jsonl"
FAITHFUL_SCHEMA = {"type": "object", "properties": {"faithful": {"type": "boolean"}, "issue": STR},
                   "required": ["faithful", "issue"]}
WORDING_SCHEMA = {"type": "object", "properties": {"question_info": STR, "question_info_hindi": STR},
                  "required": ["question_info", "question_info_hindi"]}


def log_attempt(record: dict) -> None:
    """One JSON line per diagram attempt, so failure reasons per figure type can be reviewed and fixed over time."""
    FIG_LOG.parent.mkdir(exist_ok=True)
    with FIG_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **record}, ensure_ascii=False) + "\n")


def make_figure_question(name: str, rng: random.Random, state: State) -> tuple[dict | None, str]:
    """Code builds the figure, numbers, answer and options; the LLM only writes the wording. Returns (question, problem)."""
    p = figure_problems.build(name, rng)
    t = p["template"]
    spec = t if diagram.is_graph(t) else diagram.complete(diagram.from_template(t))
    errs = diagram.check(spec, " ".join(p["facts"]))
    if errs:
        return None, "figure: " + "; ".join(errs[:2])
    style = "\n".join(e["question_info"].split("\n")[0] for e in state["examples"][:3])
    user = f"""Write the wording of ONE exam question for this figure. Do not solve it.

FACTS SHOWN IN THE FIGURE (use exactly these numbers and point names, add no other numbers):
- {chr(10).join('- ' + f for f in p['facts'])[2:]}
THE QUESTION MUST ASK FOR: {p['ask']}

Style of our exam (for tone only): {style}

Rules: question_info is English only and starts with "In the given figure,"; question_info_hindi is the same question in
natural CBSE-paper Hindi starting with "दी गई आकृति में,". No options, no answer, no solution, plain Unicode maths (∠, °, √)."""
    w = ask_llm(LLM_MODEL_ID, "You word school exam questions clearly and exactly.", user, "submit_wording", WORDING_SCHEMA, 0.4)
    q = {"question_info": (w.get("question_info") or "").strip(), "question_info_hindi": (w.get("question_info_hindi") or "").strip()}
    split_hindi(q)
    en, hi = q["question_info"], q["question_info_hindi"]
    given = set(re.findall(r"\d+(?:\.\d+)?", " ".join(p["facts"])))
    missing = [x for x in given if not re.search(rf"(?<![\d.]){re.escape(x)}(?![\d.])", en)]
    if not en or not hi or not re.search(r"[\u0900-\u097F]", hi):
        return None, "wording: missing English or Hindi text"
    if missing:
        return None, f"wording: the question leaves out given value(s) {', '.join(sorted(missing))}"
    tok = p["answer"].split()[0]
    if (tok.rstrip("°") not in given and re.search(rf"(?<![\d.]){re.escape(tok)}(?![\d.√])", en)) or "(A)" in en:
        return None, "wording: the question gives away the answer or adds options"

    mcq = state.get("question_type") in (None, "", "mcq")
    if mcq:
        opts = p["wrong"] + [p["answer"]]
        rng.shuffle(opts)
        if p.get("sort_options"):  # counts read best in order: 0, 1, 2, 3
            opts.sort(key=lambda o: mathcheck.option_value(o) if mathcheck.option_value(o) is not None else 0)
        key = "ABCD"[opts.index(p["answer"])]
        block = "\n".join(f"({c}) {o}" for c, o in zip("ABCD", opts))
        en, hi = f"{en}\n{block}", f"{hi}\n{block}"
    q.update(question_info=en, question_info_hindi=hi, question_type="mcq" if mcq else state["question_type"],
             answer=key if mcq else p["answer"], solution=p["solution"], topic=p["topic"][0], topic_hindi=p["topic"][1],
             subtopic=p["subtopic"][0], subtopic_hindi=p["subtopic"][1], diagram=spec, diagram_svg=diagram.render_svg(spec))

    # The answer is right by construction (code computed it), so the checker does not re-solve the maths; it only
    # confirms the wording states exactly these facts and asks for exactly this quantity.
    user = f"""Compare an exam question with the facts it must state.

FACTS: {'; '.join(p['facts'])}
IT MUST ASK FOR: {p['ask']}

QUESTION:
{en}

faithful: true only if the question states every fact (same numbers and point names), asks for exactly that quantity,
adds no other numbers or conditions, and is clear and grammatical. issue: what is wrong, or empty."""
    try:
        r = ask_llm(JUDGE_MODEL_ID, "You check exam wording strictly.", user, "submit_check", FAITHFUL_SCHEMA, 0)
    except Exception as e:
        return None, f"checker failed: {e}"
    if not r.get("faithful", False):
        return None, f"checker: wording does not match the figure ({r.get('issue') or 'no reason given'})"
    return q, ""


def figure_questions(state: State) -> State:
    """Diagram questions first: up to FIG_TRIES attempts each; whatever is not filled becomes a text question."""
    n, log, accepted = state["n"], [], list(state.get("accepted", []))
    if state.get("question_type") in ("assertion_reason", "case_study"):
        return {"accepted": accepted, "fig_log": ["diagram questions are not made for this question type; text questions only"]}
    names = figure_problems.matching(state["topic"]) or figure_problems.matching(
        " ".join(f"{e.get('topic', '')} {e.get('subtopic', '')}" for e in state["examples"]))
    want = n if state.get("figures") else (1 if any(e.get("image_description") for e in state["examples"]) else 0)
    if not names or want == 0:
        why = "no standard figure fits this topic yet" if not names else "the matching bank questions have no figures"
        return {"accepted": accepted, "fig_log": [f"{why}; making text questions"] if want or not names else []}
    rng = random.Random()
    order = rng.sample(names, len(names))  # different figure types across the questions of one request
    tries = 0
    for slot in range(1, want + 1):
        for attempt in range(1, FIG_TRIES + 1):
            name = order[tries % len(order)]
            tries += 1
            try:
                q, problem = make_figure_question(name, rng, state)
            except Exception as e:  # one failed attempt never stops the run
                q, problem = None, f"error: {type(e).__name__}: {e}"
            log_attempt({"topic": state["topic"], "generator": name, "slot": slot, "attempt": attempt,
                         "ok": q is not None, "problem": problem})
            if q:
                accepted.append(q)
                log.append(f"diagram question {slot}: made with '{name}' on attempt {attempt}")
                break
            log.append(f"diagram question {slot}, attempt {attempt} ('{name}'): {problem}")
        else:
            log.append(f"diagram question {slot}: {FIG_TRIES} attempts failed, a text question will be made instead")
    print("\n".join(log))
    return {"accepted": accepted, "fig_log": log}


def after_figures(state: State) -> str:
    return "finalize" if len(state["accepted"]) >= state["n"] else "generate"


def more_needed(state: State) -> str:
    max_rounds = MAX_ROUNDS + -(-state["n"] // GEN_CHUNK)  # a few spare rounds beyond the minimum needed
    if state.get("figures"):  # figure questions are rejected more often, so allow more attempts
        max_rounds += 3
    if len(state["accepted"]) >= state["n"] or state["rounds"] >= max_rounds:
        return "finalize"
    return "generate"


def finalize(state: State) -> State:
    out = []
    for q in state["accepted"]:
        has_fig = bool(isinstance(q.get("diagram"), dict) and q["diagram"].get("points"))
        item = {k: v for k, v in q.items() if k not in ("answer", "solution", "diagram", "diagram_description", "diagram_error",
                                                       "needs_figure", "answer_expression", "diagram_svg")}
        item["image_description"] = diagram.describe(q["diagram"]) if has_fig else None
        item.update(answer=q["answer"], solution=q["solution"])
        if has_fig:
            item["diagram"] = q["diagram"]  # the spec is kept so the figure can be re-rendered later
            item["diagram_svg"] = diagram.render_svg(q["diagram"])
        out.append(item)
    if len(out) < state["n"]:
        print(f"warning: only {len(out)} of {state['n']} questions passed verification")
    return {"accepted": out}


# ---------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("figures", figure_questions)
    g.add_node("generate", generate)
    g.add_node("verify", verify)
    g.add_node("finalize", finalize)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "figures")
    g.add_conditional_edges("figures", after_figures, {"generate": "generate", "finalize": "finalize"})
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", more_needed, {"generate": "generate", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topic")
    ap.add_argument("-n", type=int, default=5, help="how many questions to generate")
    ap.add_argument("--type", dest="question_type", help="mcq, assertion_reason, very_short_answer, short_answer, long_answer, case_study")
    ap.add_argument("--figures", action="store_true", help="only questions with a verified figure")
    ap.add_argument("--out", help="output JSON path (default: generated/<topic>.json)")
    args = ap.parse_args()

    result = build_graph().invoke({"topic": args.topic, "n": args.n, "question_type": args.question_type, "figures": args.figures})
    path = Path(args.out or Path("generated") / (re.sub(r"\W+", "_", args.topic.lower()).strip("_") + ".json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result["accepted"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(result['accepted'])} questions to {path}")
