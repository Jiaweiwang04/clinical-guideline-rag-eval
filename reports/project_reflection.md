# Project Reflection

This project was intended as a learning-oriented RAG experiment over a small, controlled clinical guideline corpus. The goal was to understand the core mechanics of retrieval-augmented generation rather than to build a production clinical assistant.

## What the Project Was Intended to Teach

### Chunking

The project shows that chunking is not just a preprocessing detail. Retrieval quality depends heavily on whether chunks map cleanly to the source structure.

For NICE NG222, recommendation chunks and table-row chunks were built separately because they represent different kinds of evidence. Recommendation chunks preserve guideline statements, while table-row chunks preserve structured treatment-option information from the HTML tables.

The debugging process also showed why chunk boundaries matter. Earlier extraction logic produced repeated or overlong text, which made evaluation harder to interpret. Rebuilding chunks from more precise HTML elements created cleaner retrieval units.

### BM25

BM25 provided the first retrieval baseline. It was useful because it is deterministic, interpretable, and strong when user queries share terms with the source guideline.

In this project, BM25 performed well for exact guideline wording, clinical terms, recommendation IDs, and table-specific phrases. It also gave a practical benchmark that later semantic methods had to beat.

### Local Embeddings

Local embedding retrieval introduced semantic matching without relying on an external embedding API. The project used `sentence-transformers/all-MiniLM-L6-v2` as a lightweight general-purpose model.

Embedding retrieval improved semantic recall, especially when queries used wording that differed from the guideline text. However, it had weaker top-rank precision than BM25 on this small corpus, because semantically related but less exact chunks could be ranked above the target evidence.

### Hybrid Retrieval

Hybrid retrieval combined BM25 and local embedding scores. This was the strongest retrieval setup in the project.

The experiment demonstrated why hybrid retrieval is often practical for RAG systems: BM25 contributes exact lexical precision, while embeddings contribute semantic recall. On the shared 29-query evaluation set, the selected hybrid configuration produced the best overall retrieval metrics.

### Recall@k and MRR

The project used Recall@k and MRR@5 to evaluate retrieval.

Recall@k measures whether at least one expected evidence chunk appears in the top `k` results. It is useful for RAG because answer generation can only use evidence that was retrieved.

MRR@5 measures how highly the first relevant chunk is ranked. This matters because higher-ranked chunks are more likely to influence the answer and are easier for users to inspect.

Together, these metrics showed the tradeoff between broad evidence recovery and top-rank precision.

### Evidence-Grounded Answer Generation

API answer generation was added as a qualitative demonstration after retrieval evaluation. The generator retrieves evidence with the selected hybrid retriever, builds a prompt from the retrieved chunks, and asks an API model to answer using only that evidence.

This step demonstrates the connection between retrieval and generation, including the importance of citations and evidence constraints. It was not formally evaluated with answer-level metrics.

### Limitations and Non-Goals

The project has clear limitations:

- It uses a single guideline source.
- The evaluation set has only 29 queries.
- Gold chunks were manually labelled.
- Quantitative evaluation covers retrieval only.
- Answer generation was not formally evaluated.
- The system is not clinically validated.

The project was not intended to provide individualized medical advice, support clinical decision-making, or replace professional clinical judgement. Its purpose was to learn and document the main steps of a small RAG pipeline from corpus preparation through retrieval evaluation and qualitative answer generation.
