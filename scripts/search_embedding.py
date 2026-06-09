import argparse
import json
import logging
import os
import sys
from pathlib import Path


DEFAULT_INDEX_DIR = "data/index/ng222_local_embeddings"

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_dependencies():
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging
    except ModuleNotFoundError as exc:
        missing = exc.name
        raise SystemExit(
            f"Missing local embedding dependency: {missing}\n"
            "Install the local embedding stack, then rerun this script:\n"
            "  python -m pip install sentence-transformers"
        ) from exc

    transformers_logging.set_verbosity_error()
    return np, SentenceTransformer


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


class LocalEmbeddingRetriever:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.np, SentenceTransformer = load_dependencies()

        manifest_path = self.index_dir / "manifest.json"
        chunks_path = self.index_dir / "chunks.jsonl"
        embeddings_path = self.index_dir / "embeddings.npz"

        if not manifest_path.exists() or not chunks_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError(
                f"Embedding index is incomplete: {self.index_dir}\n"
                "Build it first with:\n"
                "  python scripts\\build_embeddings_ng222.py"
            )

        self.manifest = load_json(manifest_path)
        self.chunks = load_jsonl(chunks_path)
        self.embeddings = self.np.load(embeddings_path)["embeddings"]

        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError(
                f"Index mismatch: {len(self.chunks)} chunks but "
                f"{self.embeddings.shape[0]} embedding vectors"
            )

        self.model = SentenceTransformer(self.manifest["model_name"], local_files_only=True)

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, dict]]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=bool(self.manifest.get("normalize_embeddings", True)),
            show_progress_bar=False,
        )[0]
        scores = self.embeddings @ query_embedding
        top_indices = self.np.argsort(-scores)[:top_k]
        return [(float(scores[idx]), self.chunks[int(idx)]) for idx in top_indices]


def preview_text(text: str, max_chars: int = 400) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def print_results(results: list[tuple[float, dict]]) -> None:
    if not results:
        print("\nNo results found.\n")
        return

    print("\nTop results:\n")
    for rank, (score, chunk) in enumerate(results, start=1):
        print(f"=== Rank {rank} | Score: {score:.4f} ===")
        print(f"chunk_id     : {chunk.get('chunk_id', 'N/A')}")
        print(f"source_id    : {chunk.get('source_id', 'N/A')}")
        print(f"doc          : {chunk.get('doc', 'N/A')}")
        print(f"section_path : {chunk.get('section_path', 'N/A')}")
        print(f"anchor       : {chunk.get('anchor', 'N/A')}")
        print(f"text         : {preview_text(chunk.get('text', ''))}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search NG222 chunks with a local embedding index.")
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    retriever = LocalEmbeddingRetriever(args.index_dir)

    if args.query:
        print_results(retriever.search(args.query, top_k=args.top_k))
        return

    print(f"Loaded local embedding index: {args.index_dir}")
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
