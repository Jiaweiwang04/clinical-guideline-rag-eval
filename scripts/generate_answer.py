import argparse
import os
import sys

from openai import OpenAI

from search_hybrid import DEFAULT_CHUNKS_PATH, HybridRetriever


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_INDEX_DIR = "data/index/ng222_local_embeddings"


SYSTEM_PROMPT = """You are a cautious clinical guideline evidence summarizer.

Use only the retrieved NICE guideline evidence provided by the user.
Do not use outside medical knowledge.
Do not provide individualized medical advice.
Do not invent citations.
Cite chunk_id in square brackets for every substantive claim.
If the retrieved evidence is insufficient, say that the retrieved guideline evidence is insufficient.
Keep the answer concise and clinically careful.
End with this exact sentence: This is guideline evidence, not individualized medical advice."""


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def truncate_text(text: str, max_chars: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ... [truncated]"


def format_evidence(results: list[tuple[float, dict]], max_chars: int) -> str:
    blocks = []
    for rank, (score, chunk) in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{rank}]",
                    f"chunk_id: {chunk.get('chunk_id', 'N/A')}",
                    f"source_id: {chunk.get('source_id', 'N/A')}",
                    f"section_path: {chunk.get('section_path', 'N/A')}",
                    f"anchor: {chunk.get('anchor', 'N/A')}",
                    f"hybrid_score: {score:.4f}",
                    f"text: {truncate_text(chunk.get('text', ''), max_chars)}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_user_prompt(query: str, evidence: str) -> str:
    return f"""Question:
{query}

Retrieved evidence:
{evidence}

Write an answer grounded only in the retrieved evidence. Use chunk_id citations in square brackets, for example [NICE_NG222|ng222-1_4_16]."""


def generate_answer(model: str, user_prompt: str) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in the current shell before calling the API."
        )

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_output_tokens=700,
    )
    return response.output_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a cited answer using hybrid retrieval plus an OpenAI API model.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--evidence-max-chars", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true", help="Print retrieved evidence and prompt without calling the API.")
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()

    retriever = HybridRetriever(
        chunks_path=args.chunks,
        index_dir=args.index_dir,
        alpha=args.alpha,
        candidate_k=args.candidate_k,
    )
    results = retriever.search(args.query, top_k=args.top_k)
    evidence = format_evidence(results, max_chars=args.evidence_max_chars)
    user_prompt = build_user_prompt(args.query, evidence)

    print("Question:")
    print(args.query)
    print()
    print("Retrieved evidence:")
    print(evidence)
    print()

    if args.dry_run:
        print("Prompt preview:")
        print(user_prompt)
        return

    print(f"Model: {args.model}")
    print()
    print("Answer:")
    print(generate_answer(args.model, user_prompt))


if __name__ == "__main__":
    main()
