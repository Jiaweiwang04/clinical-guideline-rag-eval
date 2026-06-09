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

## Local Embedding Evaluation

The first local embedding baseline uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Build the local embedding index:

```powershell
conda run -n ml python scripts\build_embeddings_ng222.py
```

Run embedding search:

```powershell
conda run -n ml python scripts\search_embedding.py --query "What withdrawal symptoms can happen when stopping antidepressants?" --top-k 5
```

Run embedding evaluation:

```powershell
conda run -n ml python scripts\eval_embedding.py
```

Current local embedding results:

```text
Queries: 29
Recall@1: 0.586
Recall@3: 0.862
Recall@5: 0.966
MRR@5: 0.737
```

Result files:

```text
data/eval/results/embedding_ng222_results.json
data/eval/results/embedding_ng222_results.csv
```

## Hybrid Evaluation

The hybrid retriever combines BM25 and local embedding scores:

```text
hybrid_score = alpha * normalized_bm25_score + (1 - alpha) * normalized_embedding_score
```

Run hybrid search with the selected alpha:

```powershell
conda run -n ml python scripts\search_hybrid.py --alpha 0.4 --query "When should inpatient treatment be considered for more severe depression?" --top-k 5
```

Run the hybrid alpha sweep:

```powershell
conda run -n ml python scripts\eval_hybrid.py
```

Current selected alpha:

```text
alpha: 0.40
```

Current hybrid results:

```text
Queries: 29
Recall@1: 0.759
Recall@3: 1.000
Recall@5: 1.000
MRR@5: 0.874
```

Result files:

```text
data/eval/results/hybrid_ng222_results.json
data/eval/results/hybrid_ng222_results.csv
data/eval/results/hybrid_ng222_sweep.json
```

## API Answer Generation

The first LLM stage uses the best current retriever:

```text
Hybrid retrieval with alpha = 0.40
```

The OpenAI API key and model are read from environment variables. Do not write API keys into code, JSON files, reports, or git-tracked files.

PowerShell setup for the current shell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
$env:OPENAI_MODEL="gpt-4.1-mini"
```

Dry-run without calling the API:

```powershell
conda run -n ml python scripts\generate_answer.py --query "How should antidepressant medication be tapered when stopping?" --dry-run
```

Generate an API-backed cited answer:

```powershell
conda run -n ml python scripts\generate_answer.py --query "How should antidepressant medication be tapered when stopping?"
```

The answer generator instructs the model to:

- use only retrieved NICE guideline evidence
- cite `chunk_id` for every substantive claim
- avoid individualized medical advice
- say when retrieved evidence is insufficient

## Local Web UI

Run the browser UI:

```powershell
conda run -n ml python scripts\ui_server.py --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

The UI supports:

- hybrid retrieval over NG222 chunks
- API-backed cited answer generation
- evidence inspection with chunk IDs and hybrid scores

UI screenshot:

![Clinical Guideline for Depression UI](images/UI.png)

The UI still reads API credentials only from environment variables:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
$env:OPENAI_MODEL="gpt-4.1-mini"
```

## Main Findings

BM25 is a strong first baseline for direct guideline questions when the query uses terms close to the guideline wording. It performs well on specific recommendation and table-row questions such as antidepressant review timing, psychotic depression treatment, CRHT, inpatient care, and table treatment details.

The current misses show the limits of plain lexical retrieval:

- `q007`: withdrawal symptom listing. BM25 retrieves related withdrawal duration and monitoring chunks but misses the exact symptom-list chunk.
- `q014`: relapse prevention review frequency. BM25 retrieves antidepressant review and relapse-related chunks but misses the precise 6-month review recommendation.

These failures are useful test cases for the next embedding or hybrid retrieval stage.

The local embedding baseline improves `Recall@5` and fixes the BM25 miss for withdrawal symptom listing, but it has weaker `Recall@1` and `MRR@5`. This suggests that embeddings are useful for semantic recall, while BM25 remains stronger for exact top-rank precision on this small corpus.

The hybrid retriever is currently the best method. With `alpha=0.40`, it improves over both standalone BM25 and standalone local embeddings:

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.690 | 0.793 | 0.931 | 0.768 |
| Local embedding | 0.586 | 0.862 | 0.966 | 0.737 |
| Hybrid | 0.759 | 1.000 | 1.000 | 0.874 |

## Reports

The current BM25 stage report is:

```text
reports/bm25_eval_report.md
```

The current local embedding stage report is:

```text
reports/local_embedding_eval_report.md
```

The current hybrid stage report is:

```text
reports/hybrid_eval_report.md
```

## Safety Note

This project evaluates retrieval over guideline evidence. It is not designed to provide individualized medical advice.
