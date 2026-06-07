# Retrieval Experiment Summary for NICE NG222

## Scope

This summary describes the retrieval evaluation over the NICE NG222 guideline corpus:

```text
NICE NG222: Depression in adults: treatment and management
```

The corpus contains recommendation chunks and structured table-row chunks from the NG222 recommendations HTML snapshot. All methods were evaluated on the same manually created 29-query gold set.

## Methods Compared

Three retrieval methods were evaluated:

- BM25 lexical retrieval
- local embedding retrieval with `sentence-transformers/all-MiniLM-L6-v2`
- hybrid retrieval combining normalized BM25 and embedding scores

The hybrid retriever uses:

```text
hybrid_score = alpha * normalized_bm25_score + (1 - alpha) * normalized_embedding_score
```

The selected hybrid setting is:

```text
alpha = 0.40
```

## Results

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.690 | 0.793 | 0.931 | 0.768 |
| Local embedding | 0.586 | 0.862 | 0.966 | 0.737 |
| Hybrid | 0.759 | 1.000 | 1.000 | 0.874 |

## Main Finding

BM25 is strong for exact guideline wording and clinical terms. It performs well when the query uses vocabulary close to the NICE source text, especially for precise recommendation IDs, clinical service terms, and table-specific treatment details.

Local embedding retrieval improves semantic recall. It can recover relevant evidence when the query wording differs from the guideline wording, but it has weaker top-rank precision than BM25 on this small corpus.

Hybrid retrieval gives the best overall result. It combines BM25's exact lexical matching with embedding-based semantic recall, achieving the strongest scores across all reported metrics:

```text
Recall@1: 0.759
Recall@3: 1.000
Recall@5: 1.000
MRR@5: 0.874
```

## Interpretation

The results suggest that BM25 alone is a strong baseline for this guideline corpus, but it can miss evidence when query wording is semantically related rather than lexically close. Local embeddings help with these semantic cases but can rank broad or related chunks above the exact evidence.

The hybrid approach is the most balanced method. It improves top-k recall while preserving better top-rank behavior than embedding-only retrieval.

## Limitations

This evaluation has several limitations:

- It uses a single guideline source, NICE NG222.
- The query set is small, with 29 evaluation questions.
- Gold labels were manually created.
- The evaluation is retrieval-only.
- No answer generation has been evaluated yet.
- The local embedding model is a general-purpose model, not a clinical-domain-specific embedding model.

## Next Step

The next stage should keep the hybrid retriever as the current retrieval baseline and evaluate answer generation with grounded citations. The same query set can be reused initially, but answer-level evaluation will need additional criteria such as citation correctness, faithfulness to the retrieved evidence, and safety around individualized medical advice.
