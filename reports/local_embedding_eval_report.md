# Local Embedding Evaluation Report for NICE NG222

> Note: This file documents an intermediate stage of the project. The final completed retrieval summary is in `reports/retrieval_experiment_summary.md`.

## Scope

This report documents the first local embedding retrieval baseline for the NICE NG222 corpus.

The embedding baseline uses the same corpus and the same 29-query gold set as the BM25 baseline, so the comparison is based on identical document coverage and identical relevance labels.

## Model

The local embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This is a lightweight general-purpose sentence-transformer model. It is not a clinical-domain-specific model, but it is useful as a first local semantic retrieval baseline.

## Index

The local embedding index is stored at:

```text
data/index/ng222_local_embeddings
```

Index files:

```text
data/index/ng222_local_embeddings/manifest.json
data/index/ng222_local_embeddings/chunks.jsonl
data/index/ng222_local_embeddings/embeddings.npz
```

The manifest confirms:

```text
source_chunks: data/processed/ng222_chunks_with_tables.jsonl
num_chunks: 158
embedding_dim: 384
normalize_embeddings: true
```

The embedding index covers the same 158 chunks as BM25:

- 137 recommendation chunks
- 21 table-row chunks

## Commands

Build the index:

```powershell
conda run -n ml python scripts\build_embeddings_ng222.py
```

Run a single search:

```powershell
conda run -n ml python scripts\search_embedding.py --query "What withdrawal symptoms can happen when stopping antidepressants?" --top-k 5
```

Run the evaluation:

```powershell
conda run -n ml python scripts\eval_embedding.py
```

## Results

Current local embedding results:

```text
Queries: 29
Recall@1: 0.586
Recall@3: 0.862
Recall@5: 0.966
MRR@5: 0.737
```

Output files:

```text
data/eval/results/embedding_ng222_results.json
data/eval/results/embedding_ng222_results.csv
```

## Comparison With BM25

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.690 | 0.793 | 0.931 | 0.768 |
| Local embedding | 0.586 | 0.862 | 0.966 | 0.737 |

The local embedding retriever improves top-5 recall but ranks the exact expected chunk first less often than BM25.

This suggests that local embeddings are useful for semantic candidate generation, while BM25 is still stronger for top-rank exact evidence retrieval in this small single-guideline corpus.

## BM25 Miss Fixed By Embeddings

BM25 missed:

```text
q007: What withdrawal symptoms can happen when stopping antidepressants?
expected: NICE_NG222|ng222-1_4_13
```

The local embedding retriever returns the expected symptom-list chunk at rank 1:

```text
NICE_NG222|ng222-1_4_13
```

This is the clearest benefit of the embedding baseline so far.

## Embedding Miss

The embedding retriever misses:

```text
q022: When should inpatient treatment be considered for more severe depression?
expected: NICE_NG222|ng222-1_16_12
```

The returned top results are semantically related to treatment and severe depression, but they do not retrieve the specific inpatient-treatment recommendation.

This is a useful case for hybrid retrieval, because the exact terms "inpatient treatment" are likely better handled by lexical matching.

## Conclusion

The local embedding baseline is complete and usable. It should not replace BM25 on its own, because its top-rank precision is weaker. However, it improves top-5 recall and fixes at least one BM25 miss.

The next recommended stage is hybrid retrieval:

```text
BM25 candidates + local embedding candidates
```

or score fusion:

```text
hybrid_score = normalized_bm25_score + normalized_embedding_score
```

The goal of hybrid retrieval should be to keep BM25's stronger top-rank precision while preserving the embedding model's better semantic recall.
