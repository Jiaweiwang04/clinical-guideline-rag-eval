import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def normalize_text(text: str) -> str:
    
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
  
    return normalize_text(text).split()


def load_chunks(jsonl_path: str) -> list[dict]:
   
    chunks = []
    path = Path(jsonl_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                chunks.append(obj)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_num} is invalid JSON: {e}")

    return chunks


class SimpleBM25Retriever:
    

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.doc_tokens = []
        self.doc_freq = defaultdict(int)
        self.doc_len = []
        self.avg_doc_len = 0.0
        self.N = len(chunks)

        self._build_index()

    def _build_index(self):
        
        total_len = 0

        for chunk in self.chunks:
            text = chunk.get("text", "")
            tokens = tokenize(text)
            self.doc_tokens.append(tokens)

            token_set = set(tokens)
            for token in token_set:
                self.doc_freq[token] += 1

            length = len(tokens)
            self.doc_len.append(length)
            total_len += length

        self.avg_doc_len = total_len / self.N if self.N > 0 else 0.0

    def _idf(self, term: str) -> float:
        
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_idx: int) -> float:
        
        query_tokens = tokenize(query)
        doc_tokens = self.doc_tokens[doc_idx]
        doc_counter = Counter(doc_tokens)
        doc_length = self.doc_len[doc_idx]

        score = 0.0
        for term in query_tokens:
            if term not in doc_counter:
                continue

            tf = doc_counter[term]
            idf = self._idf(term)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / self.avg_doc_len)
            )

            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, dict]]:
        
        scored = []
        for i, chunk in enumerate(self.chunks):
            s = self.score(query, i)
            if s > 0:
                scored.append((s, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


def preview_text(text: str, max_chars: int = 400) -> str:
    
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def print_results(results: list[tuple[float, dict]]):
    
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


def main():
    jsonl_path = "data/processed/ng222_chunks_with_tables.jsonl"

    print(f"Loading chunks from: {jsonl_path}")
    chunks = load_chunks(jsonl_path)
    print(f"Loaded {len(chunks)} chunks.")

    retriever = SimpleBM25Retriever(chunks)

    print("\nRetriever is ready.")
    print("Type your query and press Enter.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Query> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        if not query:
            continue

        results = retriever.search(query, top_k=5)
        print_results(results)


if __name__ == "__main__":
    main()