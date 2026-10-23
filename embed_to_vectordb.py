"""Embed questions.json into an Amazon S3 Vectors index using the Titan embedding model from .env (Bedrock).

Usage:
    python embed_to_vectordb.py                          # embed questions.json
    python embed_to_vectordb.py --file other.json --doc-id cbse_x_2025_30-1-1
    python embed_to_vectordb.py --dry-run                # build embed text only, no AWS calls
    python embed_to_vectordb.py --query "tangent to a circle, find the angle"
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET = os.getenv("S3_VECTORS_BUCKET", "quizerai-vectors")
INDEX = os.getenv("S3_VECTORS_INDEX", "questions")
MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL_ID", "apac.amazon.titan-embed-text-v1")
DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "1536"))
BATCH = 100  # put_vectors accepts up to 500 per call

# Long text goes in non-filterable metadata (filterable metadata is capped at 2 KB per vector).
# This list is fixed when the index is created.
NON_FILTERABLE = ["question_info", "question_info_hindi", "image_description"]


def embed_text(q: dict) -> str:
    """The text that gets embedded: both languages, topic, subtopic and the figure description."""
    parts = [
        q["question_info"],
        q.get("question_info_hindi"),
        f"Topic: {q['topic']} / {q.get('topic_hindi', '')}",
        f"Subtopic: {q['subtopic']} / {q.get('subtopic_hindi', '')}",
        f"Figure: {q['image_description']}" if q.get("image_description") else None,
    ]
    return "\n".join(p for p in parts if p)


def clients():
    import boto3
    from botocore.config import Config

    # Fail fast when AWS is unreachable instead of hanging for minutes on retries.
    cfg = Config(connect_timeout=10, read_timeout=120, retries={"max_attempts": 3, "mode": "standard"})
    return (
        boto3.client("bedrock-runtime", region_name=REGION, config=cfg),
        boto3.client("s3vectors", region_name=REGION, config=cfg),
    )


def embed(bedrock, text: str) -> list[float]:
    body = {"inputText": text}
    if "embed-text-v2" in MODEL_ID:  # only Titan V2 accepts these options; V1 is fixed at 1536 dims
        body.update({"dimensions": DIMENSIONS, "normalize": True})
    resp = bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
    return json.loads(resp["body"].read())["embedding"]


def ensure_bucket_and_index(s3v) -> None:
    from botocore.exceptions import ClientError

    try:
        s3v.get_vector_bucket(vectorBucketName=BUCKET)
    except ClientError as e:
        if e.response["Error"]["Code"] != "NotFoundException":
            raise
        s3v.create_vector_bucket(vectorBucketName=BUCKET)
        print(f"created vector bucket {BUCKET}")
    try:
        s3v.get_index(vectorBucketName=BUCKET, indexName=INDEX)
    except ClientError as e:
        if e.response["Error"]["Code"] != "NotFoundException":
            raise
        s3v.create_index(
            vectorBucketName=BUCKET,
            indexName=INDEX,
            dataType="float32",
            dimension=DIMENSIONS,
            distanceMetric="cosine",
            metadataConfiguration={"nonFilterableMetadataKeys": NON_FILTERABLE},
        )
        print(f"created index {INDEX} ({DIMENSIONS} dims, cosine)")


def ingest(path: Path, doc_id: str, dry_run: bool) -> None:
    questions = json.loads(path.read_text(encoding="utf-8"))
    print(f"{len(questions)} questions from {path.name}, doc_id={doc_id}")

    if dry_run:
        for n, q in enumerate(questions, 1):
            print(f"--- {doc_id}-{n:04d} ---\n{embed_text(q)}\n")
        return

    bedrock, s3v = clients()
    ensure_bucket_and_index(s3v)

    batch = []
    for n, q in enumerate(questions, 1):  # serial order in the file = question order
        metadata = {
            "doc_id": doc_id,
            "serial": n,
            "question_type": q["question_type"],
            "topic": q["topic"],
            "subtopic": q["subtopic"],
            "has_figure": bool(q.get("image_description")),
            "question_info": q["question_info"],
            "question_info_hindi": q["question_info_hindi"],
        }
        if q.get("image_description"):
            metadata["image_description"] = q["image_description"]
        batch.append({
            "key": f"{doc_id}-{n:04d}",  # same key on re-run overwrites, so re-running is safe
            "data": {"float32": embed(bedrock, embed_text(q))},
            "metadata": metadata,
        })
        if len(batch) == BATCH:
            s3v.put_vectors(vectorBucketName=BUCKET, indexName=INDEX, vectors=batch)
            batch = []
        print(f"embedded {n}/{len(questions)}", end="\r", flush=True)
    if batch:
        s3v.put_vectors(vectorBucketName=BUCKET, indexName=INDEX, vectors=batch)
    print(f"\nstored {len(questions)} vectors in s3vectors://{BUCKET}/{INDEX}")


def search(text: str, top_k: int, question_type: str | None) -> None:
    bedrock, s3v = clients()
    kwargs = {}
    if question_type:
        kwargs["filter"] = {"question_type": question_type}
    resp = s3v.query_vectors(
        vectorBucketName=BUCKET,
        indexName=INDEX,
        queryVector={"float32": embed(bedrock, text)},
        topK=top_k,
        returnMetadata=True,
        returnDistance=True,
        **kwargs,
    )
    for v in resp["vectors"]:
        m = v["metadata"]
        first_line = m["question_info"].splitlines()[0]
        print(f"{v['distance']:.4f}  {v['key']}  [{m['topic']}]  {first_line[:90]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(Path(__file__).with_name("questions.json")))
    ap.add_argument("--doc-id", help="prefix for vector keys; defaults to the JSON file name")
    ap.add_argument("--dry-run", action="store_true", help="print the text that would be embedded and exit")
    ap.add_argument("--query", help="search the index instead of ingesting")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--question-type", help="optional filter for --query, e.g. mcq")
    args = ap.parse_args()

    if args.query:
        search(args.query, args.top_k, args.question_type)
    else:
        p = Path(args.file)
        if not p.exists():
            sys.exit(f"{p} not found")
        ingest(p, args.doc_id or p.stem, args.dry_run)
