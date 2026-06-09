# BM25 Evaluation Report for NICE NG222

> Note: This file documents an intermediate stage of the project. The final completed retrieval summary is in `reports/retrieval_experiment_summary.md`.

## Scope

This report documents the first retrieval baseline for the project. The evaluation is limited to one guideline source:

- NICE NG222: Depression in adults: treatment and management

The purpose of this stage is to establish a reproducible lexical retrieval baseline before starting embedding-based retrieval.

## Corpus Preparation

The current retrieval corpus is:

```text
data/processed/ng222_chunks_with_tables.jsonl
```

It contains:

- 137 recommendation chunks from the NG222 recommendations HTML
- 21 table-row chunks from Table 1 and Table 2 in the same HTML snapshot
- 158 chunks total

Two build scripts are used:

```text
scripts/build_chunks_ng222.py
scripts/build_chunks_ng222_with_tables.py
```

The recommendation chunk builder now parses each `article.recommendation` block directly. This avoids the earlier issue where sibling traversal pulled text from following recommendation blocks and created duplicated or overlong chunks.

The table chunk builder now extracts Table 1 and Table 2 from the HTML snapshot rather than PDF text. This avoids PDF extraction encoding problems such as mojibake and malformed bullet characters.

## Debugging Notes

Several issues were found and fixed before the BM25 evaluation was treated as a usable baseline.

First, recommendation chunks contained repeated text. The previous extraction logic started from anchors and walked through following elements, which sometimes included later recommendations or repeated list text. The builder was rewritten to extract only the recommendation number and body from the current `article` element.

Second, table chunks previously contained encoding artifacts from PDF-derived text, including malformed bullet characters. Since the NG222 HTML snapshot already contains the treatment tables, the table rows are now parsed from HTML. The output is normalized to plain English text without non-ASCII artifacts.

Third, an early broad gold set made the evaluation hard to interpret. Questions such as "What treatment options are recommended for more severe depression?" could reasonably match a general recommendation, a table pointer, or any row in Table 2. The gold set was revised into more precise queries with clearer expected evidence.

Fourth, expected evidence was tightened. For example, the antidepressant first-line question now expects only `NICE_NG222|ng222-1_5_3`, because that chunk directly states not to routinely offer antidepressant medication as first-line treatment for less severe depression.

## Evaluation Design

The BM25 evaluation file is:

```text
data/eval/ng222_bm25_eval_questions.jsonl
```

The current v2 set contains 29 English queries. Each query has:

- `query_id`
- `query`
- `expected_chunk_ids`
- `notes`

The evaluation uses exact chunk-id matching. A query is counted as a hit at `k` if any returned chunk in the top `k` appears in `expected_chunk_ids`.

This keeps the first evaluation deterministic and easy to inspect. It does not attempt semantic answer grading.

## Evaluation Command

```powershell
python scripts\eval_bm25.py
```

Default settings:

```text
chunks: data/processed/ng222_chunks_with_tables.jsonl
eval: data/eval/ng222_bm25_eval_questions.jsonl
top_k: 5
k1: 1.5
b: 0.75
```

## Results

Current BM25 results:

```text
Queries: 29
Recall@1: 0.690
Recall@3: 0.793
Recall@5: 0.931
MRR@5: 0.768
```

Output files:

```text
data/eval/results/bm25_ng222_results.json
data/eval/results/bm25_ng222_results.csv
```

## Interpretation

BM25 performs well for direct lexical queries that use terms close to NG222 wording. It is especially effective for:

- table-row treatment details
- antidepressant medication review timing
- stopping and tapering antidepressants
- psychotic depression care
- crisis resolution and home treatment
- inpatient treatment criteria

The high `Recall@5` indicates that BM25 is a useful first-stage retriever for this single-guideline corpus.

The lower `Recall@1` and `MRR@5` show that ranking is still imperfect. BM25 often retrieves a related but broader chunk above the most precise chunk.

## Remaining Misses

Two queries remain misses at top 5.

### q007

Query:

```text
What withdrawal symptoms can happen when stopping antidepressants?
```

Expected:

```text
NICE_NG222|ng222-1_4_13
```

Observed top results retrieve related withdrawal chunks, especially `ng222-1_4_14`, which discusses withdrawal duration and severity. The exact symptom-list chunk is missed. This is a useful case where lexical overlap around "withdrawal" is not enough to rank the symptom-list evidence.

### q014

Query:

```text
How often should antidepressant medication be reviewed when used to prevent relapse?
```

Expected:

```text
NICE_NG222|ng222-1_8_11
```

Observed top results retrieve antidepressant review and relapse-related chunks but miss the precise recommendation that states review should happen at least every 6 months.

## Baseline Conclusion

The BM25 evaluation stage is complete enough to act as the lexical baseline for the next retrieval stage.

The current baseline is:

```text
BM25 over NG222 recommendation and table chunks
Recall@5 = 0.931
MRR@5 = 0.768
```

This should be used as the comparison point for embedding retrieval and hybrid retrieval.

## Next Stage

The next stage should introduce embedding-based retrieval while keeping this BM25 setup unchanged as a baseline.

Recommended next comparisons:

- BM25 only
- embedding retrieval only
- hybrid retrieval with BM25 plus embeddings

The same v2 gold set should be reused first. New queries can be added after the embedding pipeline is stable.
