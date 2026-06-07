import argparse
import csv
import json
import sys
from pathlib import Path

from eval_bm25 import first_hit_rank, load_eval_questions, reciprocal_rank
from search_embedding import DEFAULT_INDEX_DIR, LocalEmbeddingRetriever


DEFAULT_EVAL_PATH = "data/eval/ng222_bm25_eval_questions.jsonl"
DEFAULT_JSON_OUT = "data/eval/results/embedding_ng222_results.json"
DEFAULT_CSV_OUT = "data/eval/results/embedding_ng222_results.csv"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def evaluate(
    questions: list[dict],
    retriever: LocalEmbeddingRetriever,
    top_k: int,
    recall_ks: list[int],
) -> tuple[list[dict], dict]:
    max_k = max(max(recall_ks), top_k)
    rows = []

    for question in questions:
        results = retriever.search(question["query"], top_k=max_k)
        ranked_ids = [chunk.get("chunk_id", "") for _, chunk in results]
        expected_ids = set(question["expected_chunk_ids"])
        hit_rank = first_hit_rank(ranked_ids, expected_ids)

        row = {
            "query_id": question["query_id"],
            "query": question["query"],
            "expected_chunk_ids": question["expected_chunk_ids"],
            "top_chunk_ids": ranked_ids[:top_k],
            "first_hit_rank": hit_rank,
            "reciprocal_rank": reciprocal_rank(ranked_ids, expected_ids, top_k),
            "notes": question.get("notes", ""),
        }
        for k in recall_ks:
            row[f"hit_at_{k}"] = hit_rank is not None and hit_rank <= k
        rows.append(row)

    n = len(rows)
    metrics = {
        "num_queries": n,
        "top_k": top_k,
        "recall_ks": recall_ks,
        "mrr_at_k": sum(row["reciprocal_rank"] for row in rows) / n if n else 0.0,
    }
    for k in recall_ks:
        metrics[f"recall_at_{k}"] = (
            sum(1 for row in rows if row[f"hit_at_{k}"]) / n if n else 0.0
        )

    return rows, metrics


def write_json(path: str, rows: list[dict], metrics: dict, args: argparse.Namespace, manifest: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "index_dir": args.index_dir,
            "eval": args.eval,
            "top_k": args.top_k,
            "model_name": manifest.get("model_name"),
            "num_chunks": manifest.get("num_chunks"),
            "embedding_dim": manifest.get("embedding_dim"),
            "normalize_embeddings": manifest.get("normalize_embeddings"),
        },
        "metrics": metrics,
        "results": rows,
    }
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: str, rows: list[dict], recall_ks: list[int]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query",
        "first_hit_rank",
        "reciprocal_rank",
        *[f"hit_at_{k}" for k in recall_ks],
        "expected_chunk_ids",
        "top_chunk_ids",
        "notes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["expected_chunk_ids"] = ";".join(row["expected_chunk_ids"])
            csv_row["top_chunk_ids"] = ";".join(row["top_chunk_ids"])
            writer.writerow(csv_row)


def print_summary(rows: list[dict], metrics: dict, recall_ks: list[int]) -> None:
    print("Local embedding retrieval evaluation")
    print(f"Queries: {metrics['num_queries']}")
    for k in recall_ks:
        print(f"Recall@{k}: {metrics[f'recall_at_{k}']:.3f}")
    print(f"MRR@{metrics['top_k']}: {metrics['mrr_at_k']:.3f}")

    misses = [row for row in rows if not row[f"hit_at_{metrics['top_k']}"]]
    if misses:
        print("\nMisses:")
        for row in misses:
            print(f"- {row['query_id']}: {row['query']}")
            print(f"  expected: {', '.join(row['expected_chunk_ids'])}")
            print(f"  top: {', '.join(row['top_chunk_ids'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local embedding retrieval on NG222 gold queries.")
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--eval", default=DEFAULT_EVAL_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--recall-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--csv-out", default=DEFAULT_CSV_OUT)
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    recall_ks = sorted(set(args.recall_k))
    if args.top_k < max(recall_ks):
        raise ValueError("--top-k must be greater than or equal to the largest --recall-k value")

    questions = load_eval_questions(args.eval)
    retriever = LocalEmbeddingRetriever(args.index_dir)
    rows, metrics = evaluate(questions, retriever, top_k=args.top_k, recall_ks=recall_ks)

    write_json(args.json_out, rows, metrics, args, retriever.manifest)
    write_csv(args.csv_out, rows, recall_ks)
    print_summary(rows, metrics, recall_ks)
    print(f"\nWrote JSON: {args.json_out}")
    print(f"Wrote CSV : {args.csv_out}")


if __name__ == "__main__":
    main()
