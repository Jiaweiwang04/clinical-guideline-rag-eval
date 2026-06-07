import argparse
import csv
import json
import sys
from pathlib import Path

from eval_bm25 import first_hit_rank, load_eval_questions, reciprocal_rank
from search_embedding import DEFAULT_INDEX_DIR
from search_hybrid import DEFAULT_CHUNKS_PATH, HybridRetriever


DEFAULT_EVAL_PATH = "data/eval/ng222_bm25_eval_questions.jsonl"
DEFAULT_JSON_OUT = "data/eval/results/hybrid_ng222_results.json"
DEFAULT_CSV_OUT = "data/eval/results/hybrid_ng222_results.csv"
DEFAULT_SWEEP_OUT = "data/eval/results/hybrid_ng222_sweep.json"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def evaluate(
    questions: list[dict],
    retriever: HybridRetriever,
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


def best_sweep_item(sweep_items: list[dict]) -> dict:
    return max(
        sweep_items,
        key=lambda item: (
            item["metrics"]["mrr_at_k"],
            item["metrics"].get("recall_at_5", 0.0),
            item["metrics"].get("recall_at_3", 0.0),
            item["metrics"].get("recall_at_1", 0.0),
        ),
    )


def write_json(path: str, rows: list[dict], metrics: dict, args: argparse.Namespace, alpha: float) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "chunks": args.chunks,
            "index_dir": args.index_dir,
            "eval": args.eval,
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "alpha": alpha,
            "k1": args.k1,
            "b": args.b,
        },
        "metrics": metrics,
        "results": rows,
    }
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_sweep_json(path: str, sweep_items: list[dict], args: argparse.Namespace) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "chunks": args.chunks,
            "index_dir": args.index_dir,
            "eval": args.eval,
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "alphas": args.alpha,
            "k1": args.k1,
            "b": args.b,
        },
        "sweep": [
            {
                "alpha": item["alpha"],
                "metrics": item["metrics"],
            }
            for item in sweep_items
        ],
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


def print_sweep_summary(sweep_items: list[dict], best_item: dict, recall_ks: list[int]) -> None:
    print("Hybrid retrieval alpha sweep")
    for item in sweep_items:
        metrics = item["metrics"]
        parts = [f"alpha={item['alpha']:.2f}"]
        for k in recall_ks:
            parts.append(f"Recall@{k}={metrics[f'recall_at_{k}']:.3f}")
        parts.append(f"MRR@{metrics['top_k']}={metrics['mrr_at_k']:.3f}")
        print("  " + " | ".join(parts))

    print(f"\nSelected alpha: {best_item['alpha']:.2f}")
    metrics = best_item["metrics"]
    print(f"Queries: {metrics['num_queries']}")
    for k in recall_ks:
        print(f"Recall@{k}: {metrics[f'recall_at_{k}']:.3f}")
    print(f"MRR@{metrics['top_k']}: {metrics['mrr_at_k']:.3f}")

    misses = [row for row in best_item["rows"] if not row[f"hit_at_{metrics['top_k']}"]]
    if misses:
        print("\nMisses:")
        for row in misses:
            print(f"- {row['query_id']}: {row['query']}")
            print(f"  expected: {', '.join(row['expected_chunk_ids'])}")
            print(f"  top: {', '.join(row['top_chunk_ids'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BM25 plus local embedding hybrid retrieval.")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--eval", default=DEFAULT_EVAL_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--recall-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.5, 0.6, 0.8],
        help="BM25 weight(s); embedding weight is 1-alpha.",
    )
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--csv-out", default=DEFAULT_CSV_OUT)
    parser.add_argument("--sweep-out", default=DEFAULT_SWEEP_OUT)
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()

    recall_ks = sorted(set(args.recall_k))
    if args.top_k < max(recall_ks):
        raise ValueError("--top-k must be greater than or equal to the largest --recall-k value")

    alphas = sorted(set(args.alpha))
    for alpha in alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("all alpha values must be between 0.0 and 1.0")

    questions = load_eval_questions(args.eval)
    retriever = HybridRetriever(
        chunks_path=args.chunks,
        index_dir=args.index_dir,
        alpha=alphas[0],
        candidate_k=args.candidate_k,
        k1=args.k1,
        b=args.b,
    )

    sweep_items = []
    for alpha in alphas:
        retriever.alpha = alpha
        rows, metrics = evaluate(questions, retriever, top_k=args.top_k, recall_ks=recall_ks)
        sweep_items.append({"alpha": alpha, "rows": rows, "metrics": metrics})

    selected = best_sweep_item(sweep_items)
    write_json(args.json_out, selected["rows"], selected["metrics"], args, selected["alpha"])
    write_csv(args.csv_out, selected["rows"], recall_ks)
    write_sweep_json(args.sweep_out, sweep_items, args)
    print_sweep_summary(sweep_items, selected, recall_ks)
    print(f"\nWrote JSON : {args.json_out}")
    print(f"Wrote CSV  : {args.csv_out}")
    print(f"Wrote sweep: {args.sweep_out}")


if __name__ == "__main__":
    main()
