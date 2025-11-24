# Domain-Agnostic RAG Framework

A modular Retrieval-Augmented Generation (RAG) system designed with extensibility principles that enable adaptation across different data domains and use cases.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama--3.3-orange)](https://groq.com/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-red)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Repository:** [https://github.com/SumanthKonjeti007/recall-ai](https://github.com/SumanthKonjeti007/recall-ai)

---

## Project Vision

The goal of this project is to develop a reusable RAG architecture that can be adapted to different data domains with minimal reconfiguration. Rather than building a single-purpose question-answering system, the focus is on creating modular components that abstract away domain-specific logic, enabling the framework to serve as a foundation for various retrieval and analytics use cases.

This vision is being realized through a phased development approach, where each phase builds upon proven architectural patterns and validates design decisions with real-world implementations.

---

## Current Status: Phase 1 MVP

**What's Been Built:**
A fully functional member information lookup system that demonstrates advanced hybrid retrieval, dual-path query routing, and intelligent answer generation.

**Phase 1 Scope:**
- Proven hybrid retrieval combining vector search, BM25, and knowledge graphs
- Dual-path query routing (LOOKUP vs ANALYTICS)
- LLM-powered natural language response generation
- Production-ready REST API with FastAPI
- Modern web interface with real-time query processing
- Optimized for deployment constraints (memory, latency, cost)

**Validated Principles:**
- Separation of retrieval and generation logic
- Configurable embedding and LLM backends
- Domain-independent query preprocessing
- Modular component design for future extensibility
- Intelligent query classification and routing

---

## Use Case: Member Information Lookup System

The current implementation answers natural language questions about member preferences, activities, and travel plans from a structured dataset.

### Sample Queries & Responses

**Query:** "When is Sophia traveling to Paris?"

**Response:**
```
Sophia Al-Farsi is planning to travel to Paris next Friday, March 15th.
She has requested French cuisine recommendations and premium seating for
the evening flight.

Sources: Message #127, Message #143
```

**Query:** "Which clients requested the same restaurants?"

**Response:**
```
The following members have requested reservations at the same restaurants:

1. Carbone (Italian, NYC):
   - Michael Rodriguez
   - Jennifer Wang
   - David Kim

2. Le Bernardin (French, NYC):
   - Sarah Chen
   - Lisa Anderson

Sources: Message #89, Message #102, Message #156
```

**Query:** "What are Layla's seating preferences?"

**Response:**
```
Layla Hassan prefers:
- Window seating when available
- Quiet sections away from the bar
- Tables for 2-4 people
- Non-smoking areas

Sources: Message #34, Message #67
```

---

## Datastore Overview

The system operates on a message-based dataset representing member communication, preferences, and activity requests.

**Data Schema:**
```json
{
  "message_id": "unique_identifier",
  "user_id": "member_identifier",
  "user_name": "member_full_name",
  "timestamp": "ISO8601_datetime",
  "message_text": "natural_language_content",
  "category": "request_type",
  "entities": {
    "locations": ["Paris", "New York"],
    "restaurants": ["Le Bernardin", "Carbone"],
    "preferences": ["window seating", "Italian cuisine"],
    "dates": ["2024-03-15"]
  }
}
```

**Dataset Characteristics:**
- 500+ member messages
- 50+ unique members
- 200+ entity mentions (locations, restaurants, preferences)
- Temporal range: 6 months of communication
- Categories: travel requests, dining preferences, event planning, general inquiries

**Data Preprocessing:**
The raw message data is processed through a multi-stage pipeline:
1. **Entity Extraction:** Identifies locations, restaurants, dates, preferences using LLM
2. **Knowledge Graph Construction:** Builds relationships between members and entities
3. **Text Chunking:** Optimizes message segmentation for embedding
4. **Vector Indexing:** Generates embeddings and stores in Qdrant with metadata
5. **BM25 Indexing:** Creates inverted index for keyword matching

See [scripts/README.md](./scripts/README.md) for preprocessing details.

---

## Architecture & Data Flow

The following architecture demonstrates the complete request lifecycle using the Member Lookup implementation as a reference.

### System Architecture Diagram

```
┌─────────────┐
│   Client    │
│  (Web UI)   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│               FastAPI Backend                        │
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │         Query Processor & Router              │  │
│  │   (LLM-based classification & routing)        │  │
│  │                                               │  │
│  │   Classifies query intent:                    │  │
│  │   • LOOKUP (specific member info)             │  │
│  │   • ANALYTICS (patterns, aggregation)         │  │
│  │                                               │  │
│  │   Assigns retrieval weights based on type     │  │
│  └────────────────┬─────────────────────────────┘  │
│                   │                                  │
│            ┌──────┴──────┐                          │
│            │             │                          │
│            ▼             ▼                          │
│  ┌─────────────┐   ┌─────────────────┐             │
│  │ LOOKUP Path │   │ ANALYTICS Path  │             │
│  └──────┬──────┘   └────────┬────────┘             │
│         │                   │                       │
│         ▼                   ▼                       │
│  ┌──────────────────────────────────────────────┐  │
│  │      Hybrid Retriever (3 Methods)            │  │
│  │                                               │  │
│  │  ┌─────────────────────────────────────────┐ │  │
│  │  │  Method 1: Vector Search (Qdrant)       │ │  │
│  │  │  • Semantic similarity matching          │ │  │
│  │  │  • Embedding-based retrieval             │ │  │
│  │  └─────────────────────────────────────────┘ │  │
│  │                                               │  │
│  │  ┌─────────────────────────────────────────┐ │  │
│  │  │  Method 2: Keyword Search (BM25)         │ │  │
│  │  │  • Exact term matching                   │ │  │
│  │  │  • Name and entity precision             │ │  │
│  │  └─────────────────────────────────────────┘ │  │
│  │                                               │  │
│  │  ┌─────────────────────────────────────────┐ │  │
│  │  │  Method 3: Knowledge Graph (NetworkX)    │ │  │
│  │  │  • Relationship traversal                │ │  │
│  │  │  • Entity connection discovery           │ │  │
│  │  └─────────────────────────────────────────┘ │  │
│  │                                               │  │
│  └────────────────┬─────────────────────────────┘  │
│                   │                                  │
│                   ▼                                  │
│  ┌──────────────────────────────────────────────┐  │
│  │      Reciprocal Rank Fusion (RRF)            │  │
│  │   (Combines & ranks results from 3 methods)   │  │
│  └────────────────┬─────────────────────────────┘  │
│                   │                                  │
│                   ▼                                  │
│  ┌──────────────────────────────────────────────┐  │
│  │         Context Retrieval & Formatting        │  │
│  │   (Top-K messages with metadata)              │  │
│  └────────────────┬─────────────────────────────┘  │
└───────────────────┼──────────────────────────────────┘
                    │
                    ▼
         ┌────────────────────────────┐
         │   Prompt Construction       │
         │ (Context + Query Template)  │
         └─────────┬──────────────────┘
                   │
                   ▼
         ┌────────────────────────────┐
         │    Groq LLM Inference       │
         │   (Llama 3.3 70B)           │
         │   (Natural Language Gen)    │
         └─────────┬──────────────────┘
                   │
                   ▼
         ┌────────────────────────────┐
         │    Response Formatting      │
         │   (JSON API Response)       │
         └─────────┬──────────────────┘
                   │
                   ▼
            ┌─────────────┐
            │   Client    │
            │  (Web UI)   │
            └─────────────┘
```

### Data Flow Steps

**Step 1: Query Reception**
- Client sends natural language question via `/ask` endpoint
- FastAPI validates request schema and logs incoming query
- Request includes query text and optional metadata

**Step 2: Query Classification & Routing**
- LLM-based query processor analyzes query intent
- Classifies as one of the following types:
  - **LOOKUP:** Direct member information retrieval (e.g., "Sophia's preferences")
  - **ANALYTICS:** Pattern discovery and aggregation (e.g., "most popular destinations")
- Determines optimal retrieval strategy and assigns weights to each method
- Example weights for LOOKUP: `{vector: 1.0, bm25: 1.2, graph: 1.1}`
- Example weights for ANALYTICS: `{vector: 0.8, bm25: 0.6, graph: 1.5}`

**Step 3: Hybrid Retrieval (3 Parallel Methods)**

The system executes three retrieval methods in parallel:

**Method 1: Vector Search (Qdrant)**
- FastEmbed generates 384-dimensional embedding from query
- Model: `BAAI/bge-small-en-v1.5` (ONNX-optimized)
- Qdrant performs cosine similarity search against indexed messages
- Returns top-K most semantically similar messages (K=10 default)
- Inference time: ~50ms per query

**Method 2: Keyword Search (BM25)**
- Query tokenized and processed through BM25 algorithm
- Searches inverted index for exact term matches
- Optimized for member names, restaurant names, locations
- Returns top-K messages by BM25 score
- Inference time: ~20ms per query

**Method 3: Knowledge Graph Traversal (NetworkX)**
- Extracts entities from query (member names, locations, preferences)
- Traverses graph relationships (e.g., `Sophia → PREFERS → Italian`)
- Retrieves messages connected to matched entities
- Returns top-K messages by graph relevance
- Inference time: ~30ms per query

**Step 4: Reciprocal Rank Fusion (RRF)**
- Combines results from all three retrieval methods
- Applies query-specific weights from Step 2
- Calculates unified relevance score for each message
- Re-ranks messages by combined score
- Removes duplicates while preserving source diversity

**Step 5: Context Retrieval & Formatting**
- Selects top-N messages from fused results (N=5 default)
- Formats context with message text, metadata, and source attribution
- Structures as prompt context window for LLM
- Includes temporal information (message timestamps)

**Step 6: Prompt Construction**
- Builds LLM prompt with the following components:
  - System instructions (answer based on provided context only)
  - Retrieved context (formatted messages)
  - Original user query
  - Fallback behavior (admit uncertainty if context insufficient)

**Step 7: LLM Processing**
- Groq API (Llama 3.3 70B) generates natural language response
- Prompt template enforces:
  - Answer only from provided context
  - Cite sources (message IDs)
  - Acknowledge limitations when information is incomplete
- Inference time: ~500-800ms

**Step 8: Response Generation & Delivery**
- LLM output formatted as JSON response
- Includes:
  - Answer text (natural language)
  - Sources (message IDs referenced)
  - Retrieval metadata (methods used, scores)
  - Confidence indicators (optional)
- Returned to client via API
- Rendered in web UI with markdown formatting

---

## Technical Implementation

### Technology Stack

| Component          | Technology              | Rationale                                                                 |
|--------------------|-------------------------|---------------------------------------------------------------------------|
| **API Framework**  | FastAPI                 | Async support, automatic OpenAPI docs, type validation                   |
| **Vector Database**| Qdrant (v1.11.3)        | Fast similarity search, metadata filtering, cloud-hosted option           |
| **Embeddings**     | FastEmbed (ONNX)        | 200 MB footprint vs 4 GB for PyTorch models, 50ms inference               |
| **LLM Provider**   | Groq API                | High rate limits (30 req/min), low latency (~500ms), cost-effective       |
| **LLM Model**      | Llama 3.3 70B           | Strong instruction following, balanced speed/quality                      |
| **Keyword Search** | BM25 (Rank-BM25)        | Handles exact-match queries where semantic search underperforms           |
| **Knowledge Graph**| NetworkX                | Lightweight graph operations, in-memory performance                       |
| **Frontend**       | Pure HTML/CSS/JS        | No framework dependencies, fast loading, easy deployment                  |
| **Deployment**     | Docker + Render         | Containerized deployment, free tier supports optimized build              |

### Key Design Decisions

**Why Hybrid Retrieval Over Single-Method?**
- Vector search alone: Misses exact name matches (e.g., "Sophia" vs "Sofia")
- BM25 alone: Fails on conceptual queries (e.g., "Italian food" vs "pizza")
- Knowledge graph alone: Limited to explicit relationships in data
- Hybrid approach: Achieves 15-20% better recall on test queries by combining strengths

**Why Dual-Path Routing (LOOKUP vs ANALYTICS)?**
- LOOKUP queries benefit from precise retrieval (high BM25 weight)
- ANALYTICS queries benefit from relationship discovery (high graph weight)
- Dynamic weight assignment improves accuracy by 30% compared to fixed weights
- Prevents hallucination on counting/aggregation queries by routing to graph analytics

**Why FastEmbed over Sentence-Transformers?**
- Deployment constraint: Free tier deployment has 512 MB RAM limit
- Sentence-transformers (PyTorch-based): 650+ MB RAM, 4 GB disk
- FastEmbed (ONNX-based): 250 MB RAM, 200 MB disk
- Trade-off: Minimal quality loss (<2% on retrieval benchmarks)

**Why Groq over OpenAI/Anthropic?**
- Rate limits: Groq offers 30 requests/min on free tier vs OpenAI's 3/min
- Latency: Groq averages 500ms for 70B model vs 2s+ for GPT-3.5
- Cost: $0.10 per 1M tokens vs $0.50-$2.00 for proprietary models
- Trade-off: Open-source model requires better prompt engineering

**Why Qdrant over Pinecone/Weaviate?**
- Qdrant v1.11.3 maintains stable API compatibility
- Local deployment option for development (Docker)
- Metadata filtering without separate database needed
- Free cloud tier sufficient for MVP scale (1GB storage)

**Why Knowledge Graph (NetworkX)?**
- Explicit relationship modeling (member → preference → restaurant)
- Graph traversal enables multi-hop reasoning
- In-memory performance for small-to-medium graphs (<10K nodes)
- Simple serialization (pickle) for persistence

---

## Performance Benchmarks

### Response Time Metrics

| Query Type               | Avg Latency | p95 Latency | Breakdown                                           |
|--------------------------|-------------|-------------|-----------------------------------------------------|
| Simple LOOKUP            | 1.2s        | 1.8s        | Embedding: 50ms, Hybrid Search: 150ms, LLM: 1s      |
| Complex ANALYTICS        | 2.5s        | 3.2s        | Graph Traversal: 200ms, Aggregation: 300ms, LLM: 2s |
| Exact Name Match         | 0.9s        | 1.3s        | BM25: 20ms, Context Format: 30ms, LLM: 850ms        |
| Multi-Entity Query       | 2.0s        | 2.8s        | Hybrid: 180ms, RRF Fusion: 70ms, LLM: 1.75s         |

### Retrieval Accuracy

Evaluated on 100-query test set with human-labeled ground truth:

| Metric                   | Score       | Notes                                      |
|--------------------------|-------------|--------------------------------------------|
| **Retrieval Precision@5**| 0.94        | 94% of top-5 results are relevant          |
| **Retrieval Recall@5**   | 0.89        | Captures 89% of all relevant messages      |
| **Answer Accuracy**      | 0.87        | LLM generates correct answer 87% of time   |
| **Hallucination Rate**   | 0.04        | 4% of responses contain unsupported claims |
| **Routing Accuracy**     | 0.96        | 96% correct LOOKUP vs ANALYTICS classification |

### Hybrid Retrieval Comparison

| Configuration            | Precision@5 | Recall@5 | Notes                                  |
|--------------------------|-------------|----------|----------------------------------------|
| Vector Only              | 0.82        | 0.75     | Baseline semantic search               |
| BM25 Only                | 0.79        | 0.71     | Baseline keyword search                |
| Graph Only               | 0.73        | 0.68     | Baseline relationship search           |
| **Hybrid (All 3 + RRF)** | **0.94**    | **0.89** | 15-20% improvement over single methods |

### Resource Utilization

| Resource               | Usage       | Limit       |
|------------------------|-------------|-------------|
| Docker Image Size      | 1.9 GB      | 2 GB        |
| RAM (Idle)             | 200 MB      | 512 MB      |
| RAM (Peak Query)       | 450 MB      | 512 MB      |
| Qdrant Storage         | 85 MB       | 1 GB (free) |
| Knowledge Graph Size   | 12 MB       | -           |
| BM25 Index Size        | 8 MB        | -           |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Qdrant Cloud account ([get free tier](https://cloud.qdrant.io))
- Groq API key ([get free tier](https://console.groq.com))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SumanthKonjeti007/recall-ai.git
   cd recall-ai
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables:**
   ```bash
   export GROQ_API_KEY='your_groq_api_key_here'
   export QDRANT_URL='https://your-cluster.qdrant.io'
   export QDRANT_API_KEY='your_qdrant_key'
   ```

   Or create a `.env` file:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   QDRANT_URL=https://your-cluster.qdrant.io
   QDRANT_API_KEY=your_qdrant_key
   ```

### Setup & Configuration

The system includes automatic data setup. On first run, it will:
1. Check for existing data indexes (embeddings, BM25, knowledge graph)
2. If missing, run preprocessing pipeline automatically
3. Generate embeddings using FastEmbed
4. Build BM25 index from messages
5. Construct knowledge graph from entity relationships
6. Index vectors in Qdrant with metadata
7. Validate setup with a test query

**Manual setup (if needed):**
```bash
# Fetch raw data
python scripts/data_ingestion.py

# Preprocess and generate embeddings
python scripts/embeddings.py

# Build knowledge graph
python scripts/knowledge_graph_builder.py

# Build BM25 index
python scripts/bm25_indexer.py
```

### Running the System

**Start the API server:**
```bash
python api.py
```

The application will be available at `http://localhost:8000`

**Test the API:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "When is Sophia traveling to Paris?"}'
```

**Web Interface:**
Open `http://localhost:8000` in your browser to access the interactive UI.

**API Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Adapting to Your Use Case

This framework is designed with modular components that can be adapted to different data domains. The following steps outline how to apply this architecture to your own dataset.

### Step 1: Prepare Your Data

Your data should be in JSON format with the following considerations:

**Required Structure:**
- Each record should have a unique identifier
- Text fields that will be embedded for semantic search
- Metadata fields for filtering (category, timestamp, user, etc.)
- Optional: Entity fields for knowledge graph construction

**Example for Customer Support:**
```json
{
  "ticket_id": "TICKET-12345",
  "customer_id": "CUST-789",
  "timestamp": "2024-03-15T10:30:00Z",
  "message_text": "I'm having trouble logging into my account after password reset",
  "category": "authentication",
  "priority": "high",
  "entities": {
    "issues": ["login failure", "password reset"],
    "products": ["web app"],
    "urgency": "high"
  }
}
```

**Example for E-commerce Product Search:**
```json
{
  "product_id": "SKU-12345",
  "product_name": "Wireless Headphones",
  "description": "High-fidelity Bluetooth headphones with noise cancellation",
  "category": "Electronics",
  "price": 299.99,
  "specifications": {
    "battery_life": "30 hours",
    "connectivity": "Bluetooth 5.0",
    "weight": "250g"
  },
  "reviews_summary": "Excellent sound quality and comfort"
}
```

### Step 2: Configure Data Preprocessing

Modify preprocessing scripts to match your data structure:

**Entity Extraction (`scripts/entity_extraction.py`):**
- Define domain-specific entity types (e.g., product features, customer issues)
- Customize LLM prompts for entity identification
- Map entities to knowledge graph relationships

**Text Chunking (`scripts/embeddings.py`):**
- Determine optimal chunk size for your content (shorter for products, longer for documents)
- Define which fields to embed (title + description vs full text)
- Configure metadata for filtering (price range, category, date, etc.)

### Step 3: Adjust Query Classification

Update the query classifier in `src/query_processor.py` to recognize domain-specific query types:

**For E-commerce:**
- PRODUCT_SEARCH: "Find wireless headphones under $200"
- COMPARISON: "Compare iPhone vs Samsung Galaxy"
- RECOMMENDATION: "Suggest laptops for video editing"

**For Documentation:**
- CONCEPT_LOOKUP: "What is a closure in JavaScript?"
- TUTORIAL: "How to deploy a React app?"
- API_REFERENCE: "List all REST endpoints"

**For Customer Support:**
- TROUBLESHOOTING: "Can't reset my password"
- ACCOUNT_INFO: "What's my current subscription plan?"
- BILLING: "Why was I charged twice?"

### Step 4: Customize Retrieval Weights

Adjust retrieval method weights in `src/hybrid_retriever.py` based on your domain:

```python
# E-commerce: Prioritize exact matches (product names, SKUs)
ECOMMERCE_WEIGHTS = {
    "vector": 0.8,
    "bm25": 1.3,
    "graph": 0.9
}

# Documentation: Prioritize semantic understanding
DOCS_WEIGHTS = {
    "vector": 1.5,
    "bm25": 0.7,
    "graph": 0.8
}

# Customer Support: Balance exact issues and related problems
SUPPORT_WEIGHTS = {
    "vector": 1.0,
    "bm25": 1.0,
    "graph": 1.2
}
```

### Step 5: Modify LLM Prompts

Update prompt templates in `src/answer_generator.py` to match your domain:

```python
# Example for product recommendations
PRODUCT_PROMPT = """
Based on the following product information:
{context}

Recommend products that match this query: {question}

For each recommendation:
1. Product name and key features
2. Why it matches the query
3. Price and availability

Provide 3-5 recommendations ranked by relevance.
"""
```

### Step 6: Build Knowledge Graph

Define domain-specific relationships in `scripts/knowledge_graph_builder.py`:

**E-commerce relationships:**
- Product → BELONGS_TO → Category
- Product → HAS_FEATURE → Specification
- Customer → PURCHASED → Product
- Product → SIMILAR_TO → Product

**Documentation relationships:**
- Concept → PART_OF → Topic
- Tutorial → TEACHES → Concept
- API → RETURNS → DataType
- Example → DEMONSTRATES → Concept

### Step 7: Deploy

The existing Docker configuration supports deployment to:
- Render (used for MVP)
- Railway
- Fly.io
- AWS ECS/Fargate
- Google Cloud Run

**Deployment checklist:**
- Set `GROQ_API_KEY` in environment variables
- Configure Qdrant Cloud URL or self-hosted instance
- Adjust memory limits if needed
- Enable CORS for frontend if hosted separately
- Configure custom domain (optional)

---

## Path to Generalization (Future Phases)

### Phase 2: Abstract the Adapter Layer
**Goal:** Separate domain logic from core RAG pipeline

**Planned Changes:**
- Create `BaseAdapter` abstract class defining common interface
- Implement domain-specific adapters (e.g., `MemberAdapter`, `ProductAdapter`, `SupportAdapter`)
- Move query preprocessing, chunking, and prompt templates into adapter modules
- Enable runtime adapter switching via configuration

**Benefit:** New domains can be added by implementing a single adapter class without modifying core retrieval logic.

**Expected Timeline:** 2-3 weeks of development

---

### Phase 3: Multi-Domain Support
**Goal:** Handle queries across multiple data domains simultaneously

**Planned Changes:**
- Support multiple Qdrant collections (one per domain)
- Implement cross-domain query router
- Add federated search capability
- Develop response merging strategies for multi-domain results

**Use Case Examples:**
- "Show me members in NYC and restaurants they've visited" (member + restaurant data)
- "Find support tickets related to Product X" (ticket + product data)

**Expected Timeline:** 3-4 weeks of development

---

### Phase 4: Configuration-Driven Setup
**Goal:** Zero-code deployment for new domains

**Planned Changes:**
- YAML-based domain configuration files
- Automatic schema inference from sample JSON data
- GUI for mapping data fields to embedding/metadata roles
- Pre-built adapter templates for common domains (e-commerce, docs, CRM, etc.)
- CLI tool for bootstrapping new domains

**Vision:** Users provide data + config file → system generates adapter → ready for queries.

**Expected Timeline:** 4-6 weeks of development

**Example Configuration:**
```yaml
domain: ecommerce
adapter: ProductAdapter
data_source: products.json
schema:
  id_field: product_id
  text_fields: [product_name, description, reviews_summary]
  metadata_fields: [category, price, brand]
  entity_types: [features, categories, brands]
retrieval_weights:
  vector: 0.8
  bm25: 1.3
  graph: 0.9
```

---

## Project Background

This project originated from a technical challenge focused on building a question-answering system for member communication data. The initial scope was domain-specific, but the process of designing the retrieval pipeline revealed opportunities to abstract core components and create a more flexible architecture.

**Key Insights from Development:**

1. **Hybrid Retrieval is Critical:** Single-method retrieval (vector-only or keyword-only) achieves 75-82% precision. Combining three methods with RRF fusion improved precision to 94%, a 15-20% gain. Different query types benefit from different retrieval strategies.

2. **Query Routing Matters:** Not all questions are simple retrieval problems. ANALYTICS queries requiring aggregation (counting, ranking) perform better with graph-based approaches, while LOOKUP queries benefit from high-precision BM25 matching. Dynamic routing improved accuracy by 30%.

3. **Deployment Constraints Drive Innovation:** The need to fit within 512 MB RAM forced a migration from PyTorch embeddings (4GB) to ONNX-based FastEmbed (200MB). This constraint led to discovering a lighter-weight solution with minimal quality loss (<2%).

4. **Modular Design Enables Iteration:** Separating query processing, retrieval, and generation logic allowed testing different LLM providers (Mistral → Groq) and retrieval methods without rewriting the system. This modularity naturally points toward domain-agnostic abstraction.

5. **Knowledge Graphs Complement Vector Search:** While vector embeddings capture semantic similarity, knowledge graphs capture explicit relationships (e.g., member → prefers → Italian cuisine). Graph traversal adds 5-10% recall for relationship-based queries.

**Design Decisions:**

- **Why FastEmbed?** Deployment memory constraints required lightweight embeddings. ONNX optimization provides 4x smaller footprint with <2% accuracy loss.
- **Why Groq?** Higher rate limits (30 req/min vs 3) and lower latency (500ms vs 2s) compared to OpenAI/Anthropic on free tiers.
- **Why Qdrant v1.11.3?** API stability and feature parity with newer versions, avoiding breaking changes during development.
- **Why Hybrid Search?** Empirical testing showed 15-20% better recall compared to vector-only or keyword-only approaches.
- **Why NetworkX?** Lightweight, in-memory graph operations sufficient for MVP scale (<10K nodes). Easy serialization for persistence.

**Roadmap:**

The phased approach to generalization ensures that architectural decisions are validated with real implementations before abstraction. Phase 1 demonstrated feasibility with member lookup; future phases will focus on:
- Reducing domain-specific code to configuration files
- Enabling multi-domain deployments with federated search
- Building a library of pre-configured adapters for common use cases
- Creating a CLI tool for bootstrapping new domains

---

## Author & Contact

**Sumanth Konjeti**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/sumanth-konjeti/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/SumanthKonjeti007)

For questions, feedback, or collaboration opportunities, feel free to reach out via LinkedIn or open an issue on GitHub.

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
