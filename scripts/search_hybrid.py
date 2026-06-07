import argparse
import sys
from pathlib import Path

from search_bm25 import SimpleBM25Retriever, load_chunks, preview_text
from search_embedding import DEFAULT_INDEX_DIR, LocalEmbeddingRetriever


DEFAULT_CHUNKS_PATH = "data/processed/ng222_chunks_with_tables.jsonl"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {chunk_id: 1.0 for chunk_id in scores}
    return {
        chunk_id: (score - min_score) / (max_score - min_score)
        for chunk_id, score in scores.items()
    }


class HybridRetriever:
    def __init__(
        self,
        chunks_path: str = DEFAULT_CHUNKS_PATH,
        index_dir: str = DEFAULT_INDEX_DIR,
        alpha: float = 0.4,
        candidate_k: int = 50,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")

        self.alpha = alpha
        self.candidate_k = candidate_k
        self.chunks = load_chunks(chunks_path)
        self.chunk_by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self.bm25 = SimpleBM25Retriever(self.chunks, k1=k1, b=b)
        self.embedding = LocalEmbeddingRetriever(index_dir)

        embedding_ids = {chunk["chunk_id"] for chunk in self.embedding.chunks}
        chunk_ids = set(self.chunk_by_id)
        if embedding_ids != chunk_ids:
            missing_in_embedding = sorted(chunk_ids - embedding_ids)
            missing_in_chunks = sorted(embedding_ids - chunk_ids)
            raise ValueError(
                "BM25 chunks and embedding index do not contain the same chunk ids. "
                f"missing_in_embedding={missing_in_embedding[:5]} "
                f"missing_in_chunks={missing_in_chunks[:5]}"
            )

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, dict]]:
        bm25_results = self.bm25.search(query, top_k=self.candidate_k)
        embedding_results = self.embedding.search(query, top_k=self.candidate_k)

        bm25_raw = {chunk["chunk_id"]: score for score, chunk in bm25_results}
        embedding_raw = {chunk["chunk_id"]: score for score, chunk in embedding_results}
        bm25_norm = normalize_scores(bm25_raw)
        embedding_norm = normalize_scores(embedding_raw)

        candidate_ids = set(bm25_norm) | set(embedding_norm)
        scored = []
        for chunk_id in candidate_ids:
            bm25_score = bm25_norm.get(chunk_id, 0.0)
            embedding_score = embedding_norm.get(chunk_id, 0.0)
            hybrid_score = self.alpha * bm25_score + (1.0 - self.alpha) * embedding_score
            scored.append(
                (
                    hybrid_score,
                    bm25_score,
                    embedding_score,
                    self.chunk_by_id[chunk_id],
                )
            )

        scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]["chunk_id"]), reverse=True)
        return [(hybrid_score, chunk) for hybrid_score, _, _, chunk in scored[:top_k]]


def print_results(results: list[tuple[float, dict]]) -> None:
    if not results:
        print("\nNo results found.\n")
        return

    print("\nTop results:\n")
    for rank, (score, chunk) in enumerate(results, start=1):
        print(f"=== Rank {rank} | Hybrid score: {score:.4f} ===")
        print(f"chunk_id     : {chunk.get('chunk_id', 'N/A')}")
        print(f"source_id    : {chunk.get('source_id', 'N/A')}")
        print(f"doc          : {chunk.get('doc', 'N/A')}")
        print(f"section_path : {chunk.get('section_path', 'N/A')}")
        print(f"anchor       : {chunk.get('anchor', 'N/A')}")
        print(f"text         : {preview_text(chunk.get('text', ''))}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search NG222 chunks with BM25 plus local embedding fusion.")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.4, help="BM25 weight; embedding weight is 1-alpha.")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    retriever = HybridRetriever(
        chunks_path=args.chunks,
        index_dir=args.index_dir,
        alpha=args.alpha,
        candidate_k=args.candidate_k,
        k1=args.k1,
        b=args.b,
    )

    if args.query:
        print_results(retriever.search(args.query, top_k=args.top_k))
        return

    print(f"Loaded hybrid retriever with alpha={args.alpha:.2f}, candidate_k={args.candidate_k}")
    print("Type your query and press Enter.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Query> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not query:
            continue
        print_results(retriever.search(query, top_k=args.top_k))


if __name__ == "__main__":
    main()
