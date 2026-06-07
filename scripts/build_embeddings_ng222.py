import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from search_bm25 import load_chunks


DEFAULT_CHUNKS_PATH = "data/processed/ng222_chunks_with_tables.jsonl"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = "data/index/ng222_local_embeddings"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_dependencies():
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        missing = exc.name
        raise SystemExit(
            f"Missing local embedding dependency: {missing}\n"
            "Install the local embedding stack, then rerun this script:\n"
            "  python -m pip install sentence-transformers\n"
            "The default model is downloaded once and then loaded locally."
        ) from exc

    return np, SentenceTransformer


def write_chunks(chunks: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def write_manifest(
    path: Path,
    *,
    chunks_path: str,
    chunks: list[dict],
    model_name: str,
    embeddings_shape: tuple[int, int],
    normalize_embeddings: bool,
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_chunks": chunks_path,
        "model_name": model_name,
        "num_chunks": len(chunks),
        "embedding_dim": embeddings_shape[1],
        "normalize_embeddings": normalize_embeddings,
        "chunk_id_order": [chunk["chunk_id"] for chunk in chunks],
    }
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local sentence-transformer embeddings for NG222 chunks.")
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    np, SentenceTransformer = load_dependencies()

    chunks = load_chunks(args.chunks)
    if not chunks:
        raise SystemExit(f"No chunks found in {args.chunks}")

    texts = [chunk.get("text", "") for chunk in chunks]
    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=args.show_progress,
    )

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = index_dir / "embeddings.npz"
    chunks_path = index_dir / "chunks.jsonl"
    manifest_path = index_dir / "manifest.json"

    np.savez_compressed(embeddings_path, embeddings=embeddings)
    write_chunks(chunks, chunks_path)
    write_manifest(
        manifest_path,
        chunks_path=args.chunks,
        chunks=chunks,
        model_name=args.model,
        embeddings_shape=embeddings.shape,
        normalize_embeddings=True,
    )

    print(f"[OK] Built local embedding index -> {index_dir}")
    print(f"chunks     : {len(chunks)}")
    print(f"dimensions : {embeddings.shape[1]}")
    print(f"model      : {args.model}")


if __name__ == "__main__":
    main()
