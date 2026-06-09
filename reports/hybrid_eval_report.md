# Hybrid Retrieval Evaluation Report for NICE NG222

> Note: This file documents an intermediate stage of the project. The final completed retrieval summary is in `reports/retrieval_experiment_summary.md`.

## Scope

This report documents the first hybrid retrieval experiment for the NICE NG222 corpus.

The hybrid retriever combines:

- BM25 lexical retrieval
- local embedding retrieval with `sentence-transformers/all-MiniLM-L6-v2`

The evaluation uses the same 29-query gold set as the BM25 and local embedding baselines.

## Method

The hybrid retriever uses score fusion over the union of BM25 and embedding candidates.

For each query:

1. Retrieve top candidates from BM25.
2. Retrieve top candidates from the local embedding index.
3. Normalize each score set with min-max normalization.
4. Fuse scores with:

```text
hybrid_score = alpha * normalized_bm25_score + (1 - alpha) * normalized_embedding_score
```

In this setup, `alpha` is the BM25 weight. A larger alpha gives more weight to lexical matching. A smaller alpha gives more weight to semantic embedding similarity.

The default candidate pool is:

```text
candidate_k: 50
```

## Commands

Run a single hybrid search:

```powershell
conda run -n ml python scripts\search_hybrid.py --alpha 0.4 --query "When should inpatient treatment be considered for more severe depression?" --top-k 5
```

Run the hybrid evaluation and alpha sweep:

```powershell
conda run -n ml python scripts\eval_hybrid.py
```

## Alpha Sweep

The sweep tested:

```text
alpha = 0.20, 0.40, 0.50, 0.60, 0.80
```

Results:

| Alpha | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0.759 | 1.000 | 1.000 | 0.862 |
| 0.40 | 0.759 | 1.000 | 1.000 | 0.874 |
| 0.50 | 0.759 | 0.966 | 1.000 | 0.871 |
| 0.60 | 0.655 | 0.966 | 1.000 | 0.807 |
| 0.80 | 0.724 | 0.828 | 0.966 | 0.810 |

The selected alpha is:

```text
alpha = 0.40
```

This gives the best `MRR@5` while preserving perfect `Recall@3` and `Recall@5` on the current gold set.

## Best Hybrid Results

With `alpha=0.40`:

```text
Queries: 29
Recall@1: 0.759
Recall@3: 1.000
Recall@5: 1.000
MRR@5: 0.874
```

Output files:

```text
data/eval/results/hybrid_ng222_results.json
data/eval/results/hybrid_ng222_results.csv
data/eval/results/hybrid_ng222_sweep.json
```

## Comparison With Previous Baselines

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.690 | 0.793 | 0.931 | 0.768 |
| Local embedding | 0.586 | 0.862 | 0.966 | 0.737 |
| Hybrid | 0.759 | 1.000 | 1.000 | 0.874 |

The hybrid retriever is currently the strongest retrieval method in this project.

## Key Cases

### BM25 Miss: Withdrawal Symptoms

BM25 missed:

```text
q007: What withdrawal symptoms can happen when stopping antidepressants?
expected: NICE_NG222|ng222-1_4_13
```

The hybrid retriever with `alpha=0.40` retrieves the expected chunk in the top 3.

This shows that embedding similarity helps recover the specific symptom-list evidence that BM25 ranked below related withdrawal chunks.

### Embedding Miss: Inpatient Treatment

The local embedding retriever missed:

```text
q022: When should inpatient treatment be considered for more severe depression?
expected: NICE_NG222|ng222-1_16_12
```

The hybrid retriever with `alpha=0.40` retrieves the expected chunk at rank 1.

This shows that BM25 helps preserve exact lexical matches for specific clinical service terms such as "inpatient treatment".

## Conclusion

The hybrid stage is successful. It combines the stronger top-rank precision of BM25 with the broader semantic recall of local embeddings.

For the current NG222 single-guideline corpus, the recommended retriever is:

```text
Hybrid retrieval with alpha = 0.40 and candidate_k = 50
```

This should be treated as the current best retrieval baseline before moving to answer generation or reranking.
