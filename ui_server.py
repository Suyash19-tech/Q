"""Local web UI to check the whole pipeline: connection health, semantic search, question generation, vector index contents.

    python ui_server.py            # http://localhost:8000
    python ui_server.py --demo     # fake AWS responses, to try the UI without any AWS access
"""
import argparse
import json
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).parent
DEMO = "--demo" in sys.argv
if DEMO:
    import demo_stub

    demo_stub.install()

import embed_to_vectordb as ev  # noqa: E402  (must come after the optional demo patch)
import question_workflow as qw  # noqa: E402

JOBS: dict[str, dict] = {}


def timed(fn):
    t = time.time()
    try:
        fn()
        return {"ok": True, "seconds": round(time.time() - t, 1)}
    except Exception as e:  # show the real AWS error in the UI and in the terminal
        err = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"[check failed] {err}", flush=True)
        return {"ok": False, "seconds": round(time.time() - t, 1), "error": err}


def check(name: str) -> dict:
    if name == "embed":
        return timed(lambda: ev.embed(qw.bedrock, "health check"))
    if name == "index":
        return timed(lambda: qw.s3v.get_index(vectorBucketName=ev.BUCKET, indexName=ev.INDEX))

    def chat(model):
        qw.bedrock.converse(modelId=model, messages=[{"role": "user", "content": [{"text": "Reply with OK"}]}],
                            inferenceConfig={"maxTokens": 10})

    if name == "llm":
        return timed(lambda: chat(qw.LLM_MODEL_ID))
    if name == "judge":
        return timed(lambda: chat(qw.JUDGE_MODEL_ID))
    if name == "diagram":
        return timed(lambda: chat(qw.DIAGRAM_MODEL_ID))
    return {"ok": False, "error": "unknown check"}


def search(query: str, top_k: int, question_type: str | None) -> list[dict]:
    kwargs = {"filter": {"question_type": question_type}} if question_type else {}
    resp = qw.s3v.query_vectors(
        vectorBucketName=ev.BUCKET, indexName=ev.INDEX, queryVector={"float32": ev.embed(qw.bedrock, query)},
        topK=top_k, returnMetadata=True, returnDistance=True, **kwargs)
    return [{"key": v["key"], "distance": v["distance"], **v["metadata"]} for v in resp["vectors"]]


def list_index(token: str | None) -> dict:
    kwargs = {"nextToken": token} if token else {}
    resp = qw.s3v.list_vectors(vectorBucketName=ev.BUCKET, indexName=ev.INDEX, maxResults=100, returnMetadata=True, **kwargs)
    items = [{"key": v["key"], **v.get("metadata", {})} for v in resp["vectors"]]
    items.sort(key=lambda x: (x.get("doc_id", ""), x.get("serial", 0)))
    return {"items": items, "next": resp.get("nextToken")}


def run_job(job_id: str, topic: str, n: int, question_type: str | None, figures: bool = False) -> None:
    job = JOBS[job_id]
    try:
        init = {"topic": topic, "n": n, "question_type": question_type, "figures": figures}
        for update in qw.build_graph().stream(init, stream_mode="updates"):
            for node, out in update.items():
                job["events"].append({"node": node, "time": time.strftime("%H:%M:%S"), "data": out})
        job["status"] = "done"
    except Exception as e:
        traceback.print_exc()
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {str(e)[:400]}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        if "/api/job" not in self.path:  # skip the 1-second polling noise
            print(time.strftime("%H:%M:%S"), self.command, self.path.split("?")[0], flush=True)

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                body = (HERE / "ui" / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/config":
                self.send_json({"demo": DEMO, "region": ev.REGION, "bucket": ev.BUCKET, "index": ev.INDEX,
                                "embed_model": ev.MODEL_ID, "dimensions": ev.DIMENSIONS, "llm_model": qw.LLM_MODEL_ID,
                                "judge_model": qw.JUDGE_MODEL_ID, "diagram_model": qw.DIAGRAM_MODEL_ID, "top_k": qw.TOP_K})
            elif u.path == "/api/check":
                self.send_json(check(q.get("name", "")))
            elif u.path == "/api/index":
                self.send_json(list_index(q.get("token")))
            elif u.path == "/api/local":
                self.send_json(json.loads((HERE / "questions.json").read_text(encoding="utf-8")))
            elif u.path == "/api/job":
                self.send_json(JOBS.get(q.get("id"), {"status": "unknown", "events": []}))
            else:
                self.send_json({"error": "not found"}, 404)
        except Exception as e:
            self.send_json({"error": f"{type(e).__name__}: {str(e)[:400]}"}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if u.path == "/api/search":
                self.send_json({"results": search(data["query"], int(data.get("top_k", 5)), data.get("question_type") or None)})
            elif u.path == "/api/generate":
                job_id = uuid.uuid4().hex[:8]
                JOBS[job_id] = {"status": "running", "events": [], "topic": data["topic"]}
                threading.Thread(target=run_job, daemon=True, args=(
                    job_id, data["topic"], int(data.get("n", 3)), data.get("question_type") or None, bool(data.get("figures")))).start()
                self.send_json({"id": job_id})
            else:
                self.send_json({"error": "not found"}, 404)
        except Exception as e:
            self.send_json({"error": f"{type(e).__name__}: {str(e)[:400]}"}, 500)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--demo", action="store_true", help="fake AWS responses, no credentials needed")
    args = ap.parse_args()
    print(f"UI running at http://localhost:{args.port}" + ("  (DEMO mode: fake AWS)" if DEMO else ""))
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
