"""Fake AWS clients so the UI can be tried without credentials (python ui_server.py --demo). Not used otherwise."""
import json
import re
from pathlib import Path

import boto3

QUESTIONS = json.loads((Path(__file__).with_name("questions.json")).read_text(encoding="utf-8"))
_last_query = {"text": ""}


class _Body:
    def __init__(self, d):
        self.d = d

    def read(self):
        return json.dumps(self.d).encode()


def _words(s):
    return set(re.findall(r"\w+", s.lower()))


class FakeBedrock:
    def invoke_model(self, modelId, body):
        _last_query["text"] = json.loads(body)["inputText"]
        return {"body": _Body({"embedding": [0.0] * 1536})}

    def converse(self, modelId, messages, inferenceConfig=None, toolConfig=None, **kw):
        prompt = messages[0]["content"][0]["text"]
        if not toolConfig:
            return {"output": {"message": {"content": [{"text": "OK"}]}}}
        tool = toolConfig["toolChoice"]["tool"]["name"]
        if tool == "submit_questions":
            topic = re.search(r"Topic requested by the user: (.*)", prompt).group(1)
            n = int(re.search(r"Write (\d+) NEW", prompt).group(1))
            qs = []
            for i in range(n):
                r = 7 * (i + 2)
                area = 22 * r * r // 7
                opts = [area, area + 22, area - 14, area * 2]
                qs.append({
                    "question_info": f"[DEMO] ({topic}) The radius of a circle is {r} cm. Its area is :\n"
                                     + "\n".join(f"({c}) {o} cm²" for c, o in zip("ABCD", opts)),
                    "question_info_hindi": f"[डेमो] ({topic}) एक वृत्त की त्रिज्या {r} cm है। इसका क्षेत्रफल है :\n"
                                           + "\n".join(f"({c}) {o} cm²" for c, o in zip("ABCD", opts)),
                    "topic": "Areas Related to Circles", "topic_hindi": "वृत्तों से संबंधित क्षेत्रफल",
                    "subtopic": "Area of a circle", "subtopic_hindi": "वृत्त का क्षेत्रफल",
                    "question_type": "mcq", "answer": "A",
                    "solution": f"Area = πr² = (22/7) × {r}² = {area} cm².",
                })
            return {"output": {"message": {"content": [{"toolUse": {"input": {"questions": qs}}}]}}}
        idxs = [int(x) for x in re.findall(r"^#(\d+)", prompt, flags=re.M)]
        results = [{"index": i, "my_answer": "A", "well_posed": True, "matches_key": i % 4 != 3,
                    "issue": "" if i % 4 != 3 else "[DEMO] my answer differs from the key"} for i in idxs]
        return {"output": {"message": {"content": [{"toolUse": {"input": {"results": results}}}]}}}


class FakeS3Vectors:
    def get_index(self, **kw):
        return {"index": {}}

    def _meta(self, i, q):
        return {"doc_id": "questions", "serial": i + 1, "question_type": q["question_type"], "topic": q["topic"],
                "subtopic": q["subtopic"], "has_figure": bool(q.get("image_description")),
                "question_info": q["question_info"], "question_info_hindi": q["question_info_hindi"],
                **({"image_description": q["image_description"]} if q.get("image_description") else {})}

    def list_vectors(self, **kw):
        return {"vectors": [{"key": f"questions-{i + 1:04d}", "metadata": self._meta(i, q)} for i, q in enumerate(QUESTIONS)]}

    def query_vectors(self, topK, **kw):
        want = kw.get("filter", {}).get("question_type")
        qw = _words(_last_query["text"])
        scored = []
        for i, q in enumerate(QUESTIONS):
            if want and q["question_type"] != want:
                continue
            doc = _words(q["question_info"] + " " + q["topic"] + " " + q["subtopic"] + " " + (q.get("image_description") or ""))
            scored.append((1 - len(qw & doc) / (len(qw) or 1) * 0.9, i, q))
        scored.sort(key=lambda t: t[0])
        return {"vectors": [{"key": f"questions-{i + 1:04d}", "distance": round(d, 4), "metadata": self._meta(i, q)}
                            for d, i, q in scored[:topK]]}


def install():
    boto3.client = lambda service, **k: FakeBedrock() if service == "bedrock-runtime" else FakeS3Vectors()
