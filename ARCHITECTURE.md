# Architecture Deep Dive

Detailed step-by-step walkthrough of query execution through the codebase.

---

## Example 1: LOOKUP Query

**Query:** `"What are Sophia's dining preferences?"`

### Execution Trace

```
┌──────────────────────────────────────────────┐
│ 1. API Entry (api.py)                        │
│    POST /ask                                  │
│    → qa_system.answer(query)                 │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──────────────────────────┐
│ 2. Query Processing (query_processor.py)     │
│                                               │
│    Routing:                                   │
│    • Detect: Member-specific query ✓         │
│    • Route → LOOKUP                           │
│                                               │
│    Classification:                            │
│    • Entity: "Sophia Al-Farsi"               │
│    • Attribute: "dining" (specific)          │
│    • Weights: {semantic: 1.0, bm25: 1.2,     │
│                graph: 1.1}                    │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──────────────────────────┐
│ 3. Hybrid Retrieval (hybrid_retriever.py)    │
│                                               │
│    User Detection:                            │
│    • "Sophia" → User ID: cd3a350e...         │
│                                               │
│    Parallel Retrieval:                        │
│    ┌─────────────────────────────────┐       │
│    │ Semantic (Qdrant)               │       │
│    │ • Embed query → 384-dim vector  │       │
│    │ • Search with user filter       │       │
│    │ • Returns: 20 messages          │       │
│    │ • Time: ~50ms                   │       │
│    └─────────────────────────────────┘       │
│    ┌─────────────────────────────────┐       │
│    │ BM25 (Keywords)                 │       │
│    │ • Tokenize: ["sophia","dining"] │       │
│    │ • TF-IDF scoring                │       │
│    │ • Returns: 20 messages          │       │
│    │ • Time: ~20ms                   │       │
│    └─────────────────────────────────┘       │
│    ┌─────────────────────────────────┐       │
│    │ Graph (NetworkX)                │       │
│    │ • Find Sophia → PREFERS edges   │       │
│    │ • Returns: 10 messages          │       │
│    │ • Time: ~30ms                   │       │
│    └─────────────────────────────────┘       │
│                                               │
│    RRF Fusion:                                │
│    • Combine 3 result sets                    │
│    • Apply weights from step 2                │
│    • Multi-method messages rank higher        │
│    • Returns: Top 20 messages (ranked)        │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──────────────────────────┐
│ 4. Answer Generation (answer_generator.py)   │
│                                               │
│    Format Context:                            │
│    • Select top 5 messages                    │
│    • Format as numbered list                  │
│                                               │
│    LLM Call (Groq):                           │
│    • Model: llama-3.3-70b-versatile          │
│    • Prompt: Question + Context               │
│    • Temperature: 0.3                         │
│    • Time: ~600ms                             │
│                                               │
│    Output:                                    │
│    "**Sophia Al-Farsi** has a strong         │
│    preference for Italian cuisine..."         │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──────────────────────────┐
│ 5. Return Response (qa_system.py)            │
│                                               │
│    {                                          │
│      "answer": "Sophia prefers...",           │
│      "sources": [20 messages],                │
│      "route": "LOOKUP"                        │
│    }                                          │
└───────────────────┬──────────────────────────┘
                    │
                    ▼
              User Response

Total Time: ~1.8s
```

---

## Example 2: ANALYTICS Query

**Query:** `"Which clients requested the SAME restaurants?"`

### Execution Trace

```
┌──────────────────────────────────────────────┐
│ 1. API Entry (api.py)                        │
│    POST /ask                                  │
│    → qa_system.answer(query)                 │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──────────────────────────┐
│ 2. Query Processing (query_processor.py)     │
│                                               │
│    Routing:                                   │
│    • Detect: "which clients" (aggregation) ✓ │
│    • Route → ANALYTICS                        │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──────────────────────────┐
│ 3. Graph Analytics (graph_analytics.py)      │
│                                               │
│    Entity Extraction (LLM):                   │
│    • entity_type: "restaurant"               │
│    • method: "SAME"                           │
│    • keywords: ["restaurant", "clients"]     │
│    • Time: ~400ms                             │
│                                               │
│    Graph Query:                               │
│    • Find all (member → restaurant) triples  │
│    • Example:                                 │
│      (Sophia, wants_reservation_at,           │
│       Osteria Francescana)                    │
│    • Returns: 45 triples                      │
│                                               │
│    Aggregation:                               │
│    • Group by restaurant                      │
│    • Count clients per restaurant            │
│    • Filter: 2+ clients only                 │
│    • Result:                                  │
│      {                                        │
│        "Osteria Francescana":                │
│          ["Sophia", "Vikram", "Hans"],       │
│        "Le Bernardin":                        │
│          ["Layla", "Jennifer"]               │
│      }                                        │
│                                               │
│    Answer Generation (LLM):                   │
│    • Convert aggregated data to text         │
│    • Time: ~600ms                             │
│    • Output:                                  │
│      "2 restaurants were shared:             │
│       1. Osteria Francescana (3 clients)     │
│       2. Le Bernardin (2 clients)"           │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──────────────────────────┐
│ 4. Return Response (qa_system.py)            │
│                                               │
│    {                                          │
│      "answer": "2 restaurants...",            │
│      "route": "ANALYTICS",                    │
│      "aggregated_data": {...}                │
│    }                                          │
└───────────────────┬──────────────────────────┘
                    │
                    ▼
              User Response

Total Time: ~1.5s
```

---

## Master Flow Diagram

### Complete System Architecture

```
                            User Query
                                │
                                ▼
                    ┌───────────────────────┐
                    │    FastAPI Backend    │
                    │       (api.py)        │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Query Processor     │
                    │ (query_processor.py)  │
                    │                       │
                    │  • Routing            │
                    │  • Classification     │
                    │  • Decomposition      │
                    └───────────┬───────────┘
                                │
                        ┌───────┴────────┐
                        │   PATH SPLIT    │
                        └───────┬────────┘
                                │
                ┌───────────────┴────────────────┐
                │                                │
                ▼                                ▼
        ┌───────────────┐              ┌─────────────────┐
        │  LOOKUP PATH  │              │ ANALYTICS PATH  │
        │               │              │                 │
        │  Hybrid       │              │  Graph          │
        │  Retriever    │              │  Analytics      │
        │               │              │                 │
        │  ┌─────────┐  │              │  ┌───────────┐  │
        │  │ Qdrant  │  │              │  │  Entity   │  │
        │  │ Vector  │  │              │  │  Extract  │  │
        │  │ Search  │  │              │  │  (LLM)    │  │
        │  └─────────┘  │              │  └───────────┘  │
        │               │              │        │         │
        │  ┌─────────┐  │              │        ▼         │
        │  │  BM25   │  │              │  ┌───────────┐  │
        │  │ Keyword │  │              │  │   Graph   │  │
        │  │ Search  │  │              │  │   Query   │  │
        │  └─────────┘  │              │  └───────────┘  │
        │               │              │        │         │
        │  ┌─────────┐  │              │        ▼         │
        │  │  Graph  │  │              │  ┌───────────┐  │
        │  │Traversal│  │              │  │Aggregate  │  │
        │  └─────────┘  │              │  │  Data     │  │
        │       │       │              │  └───────────┘  │
        │       ▼       │              │                 │
        │  ┌─────────┐  │              │                 │
        │  │   RRF   │  │              │                 │
        │  │ Fusion  │  │              │                 │
        │  └─────────┘  │              │                 │
        └───────┬───────┘              └────────┬────────┘
                │                               │
                │       ┌──────────────┐        │
                └──────▶│     LLM      │◀───────┘
                        │   (Groq)     │
                        │   Generate   │
                        │   Answer     │
                        └──────┬───────┘
                               │
                               ▼
                        ┌─────────────┐
                        │  Response   │
                        │  Formatter  │
                        └──────┬──────┘
                               │
                               ▼
                        User receives
                           answer
```

### Data Flow Summary

| Component | LOOKUP Path | ANALYTICS Path |
|-----------|-------------|----------------|
| **Entry** | Query Processor | Query Processor |
| **Routing** | Member-specific | Aggregation keywords |
| **Retrieval** | 3-method hybrid (Qdrant, BM25, Graph) | Graph triples only |
| **Fusion** | RRF with weighted scores | Aggregation (GROUP BY, COUNT) |
| **LLM Calls** | 1 (answer generation) | 2 (entity extraction + answer) |
| **Output** | Natural language + sources | Natural language + aggregated data |
| **Avg Time** | 1.8s | 1.5s |

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **LOOKUP Latency (avg)** | 1.8s |
| **ANALYTICS Latency (avg)** | 1.5s |
| **Retrieval Precision@5** | 94% |
| **Answer Accuracy** | 87% |
| **Memory Usage (peak)** | 450 MB |

---

## Key Files

| File | Purpose | LOC |
|------|---------|-----|
| `api.py` | FastAPI endpoints | 255 |
| `src/qa_system.py` | Main orchestrator | 255 |
| `src/query_processor.py` | Routing & classification | 661 |
| `src/hybrid_retriever.py` | 3-method retrieval + RRF | 697 |
| `src/graph_analytics.py` | Analytics pipeline | 554 |
| `src/answer_generator.py` | LLM answer generation | 344 |

---
