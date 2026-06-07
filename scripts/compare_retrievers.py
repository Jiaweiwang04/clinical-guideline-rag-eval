import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_RESULTS = {
    "bm25": "data/eval/results/bm25_ng222_results.json",
    "embedding": "data/eval/results/embedding_ng222_results.json",
    "hybrid": "data/eval/results/hybrid_ng222_results.json",
}
DEFAULT_OUT = "data/eval/results/per_query_comparison.csv"
METHODS = ["bm25", "embedding", "hybrid"]


def load_result_file(path: str) -> dict[str, dict]:
    result_path = Path(path)
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    with result_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"Result file has no results list: {path}")

    return {row["query_id"]: row for row in rows}


def rank_value(row: dict | None) -> int | None:
    if row is None:
        return None
    return row.get("first_hit_rank")


def hit_at_5_value(row: dict | None) -> bool:
    if row is None:
        return False
    return bool(row.get("hit_at_5", False))


def select_best_method(ranks: dict[str, int | None]) -> str:
    non_null = {method: rank for method, rank in ranks.items() if rank is not None}
    if not non_null:
        return "none"

    best_rank = min(non_null.values())
    winners = [method for method in METHODS if non_null.get(method) == best_rank]
    return "+".join(winners)


def join_values(values: list[str]) -> str:
    return ";".join(values)


def build_comparison_rows(results_by_method: dict[str, dict[str, dict]]) -> list[dict]:
    query_ids = sorted(
        set().union(*(set(results.keys()) for results in results_by_method.values()))
    )
    rows = []

    for query_id in query_ids:
        method_rows = {
            method: results_by_method[method].get(query_id)
            for method in METHODS
        }
        reference = next((row for row in method_rows.values() if row is not None), None)
        if reference is None:
            continue

        ranks = {method: rank_value(row) for method, row in method_rows.items()}
        best_method = select_best_method(ranks)

        rows.append(
            {
                "query_id": query_id,
                "query": reference.get("query", ""),
                "expected_chunk_ids": join_values(reference.get("expected_chunk_ids", [])),
                "bm25_rank": ranks["bm25"],
                "embedding_rank": ranks["embedding"],
                "hybrid_rank": ranks["hybrid"],
                "bm25_hit_at_5": hit_at_5_value(method_rows["bm25"]),
                "embedding_hit_at_5": hit_at_5_value(method_rows["embedding"]),
                "hybrid_hit_at_5": hit_at_5_value(method_rows["hybrid"]),
                "best_method": best_method,
                "notes": reference.get("notes", ""),
            }
        )

    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query",
        "expected_chunk_ids",
        "bm25_rank",
        "embedding_rank",
        "hybrid_rank",
        "bm25_hit_at_5",
        "embedding_hit_at_5",
        "hybrid_hit_at_5",
        "best_method",
        "notes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    counts = Counter()
    for row in rows:
        best_method = row["best_method"]
        if best_method == "none":
            counts["none"] += 1
            continue
        for method in best_method.split("+"):
            counts[method] += 1

    print("Per-query best-method summary")
    for method in [*METHODS, "none"]:
        print(f"{method}: {counts[method]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare BM25, embedding, and hybrid per-query results.")
    parser.add_argument("--bm25", default=DEFAULT_RESULTS["bm25"])
    parser.add_argument("--embedding", default=DEFAULT_RESULTS["embedding"])
    parser.add_argument("--hybrid", default=DEFAULT_RESULTS["hybrid"])
    parser.add_argument("--out", default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_by_method = {
        "bm25": load_result_file(args.bm25),
        "embedding": load_result_file(args.embedding),
        "hybrid": load_result_file(args.hybrid),
    }
    rows = build_comparison_rows(results_by_method)
    write_csv(args.out, rows)
    print_summary(rows)
    print(f"\nWrote CSV: {args.out}")


if __name__ == "__main__":
    main()
