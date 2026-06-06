# Clinical Guideline RAG Evaluation

This project is a small retrieval evaluation pipeline for clinical guideline RAG. The current working scope is limited to one source:

- NICE NG222: Depression in adults: treatment and management

The immediate goal is to build a reliable single-guideline retrieval baseline before adding embedding-based retrieval.

## Current Status

The project has completed the first BM25 evaluation stage for NICE NG222.

Completed pieces:

- Raw NG222 HTML and PDF snapshots are stored under `data/raw/ng222/`.
- Recommendation chunks are generated from the NG222 recommendations HTML.
- Table 1 and Table 2 treatment-option rows are generated from the same HTML snapshot.
- A reproducible combined chunk file is available at `data/processed/ng222_chunks_with_tables.jsonl`.
- A BM25 retriever is implemented in `scripts/search_bm25.py`.
- A BM25 retrieval evaluation script is implemented in `scripts/eval_bm25.py`.
- A v2 gold query set is stored at `data/eval/ng222_bm25_eval_questions.jsonl`.
- Evaluation outputs are stored under `data/eval/results/`.

## Data Pipeline

Build recommendation-only chunks:

```powershell
python scripts\build_chunks_ng222.py
```

Build recommendation plus table chunks:

```powershell
python scripts\build_chunks_ng222_with_tables.py
```

The combined output contains:

- 137 recommendation chunks
- 21 table-row chunks
- 158 chunks total

The current combined corpus is:

```text
data/processed/ng222_chunks_with_tables.jsonl
```

## BM25 Evaluation

Run the BM25 evaluation:

```powershell
python scripts\eval_bm25.py
```

Default configuration:

- chunks: `data/processed/ng222_chunks_with_tables.jsonl`
- eval set: `data/eval/ng222_bm25_eval_questions.jsonl`
- top_k: `5`
- k1: `1.5`
- b: `0.75`

Current results:

```text
Queries: 29
Recall@1: 0.690
Recall@3: 0.793
Recall@5: 0.931
MRR@5: 0.768
```

Result files:

```text
data/eval/results/bm25_ng222_results.json
data/eval/results/bm25_ng222_results.csv
```

## Main Findings

BM25 is a strong first baseline for direct guideline questions when the query uses terms close to the guideline wording. It performs well on specific recommendation and table-row questions such as antidepressant review timing, psychotic depression treatment, CRHT, inpatient care, and table treatment details.

The current misses show the limits of plain lexical retrieval:

- `q007`: withdrawal symptom listing. BM25 retrieves related withdrawal duration and monitoring chunks but misses the exact symptom-list chunk.
- `q014`: relapse prevention review frequency. BM25 retrieves antidepressant review and relapse-related chunks but misses the precise 6-month review recommendation.

These failures are useful test cases for the next embedding or hybrid retrieval stage.

## Reports

The current BM25 stage report is:

```text
reports/bm25_eval_report.md
```

## Safety Note

This project evaluates retrieval over guideline evidence. It is not designed to provide individualized medical advice.
