# recall.ai

<div align="center">

**🧠 Your Second Brain - An Intelligent Member Lookup System**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama--3.3-orange)](https://groq.com/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-red)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Live Demo](#) • [Documentation](#architecture) • [Tech Stack](#tech-stack)

</div>

---

## 🎯 What is recall.ai?

A production-ready **intelligent member lookup system** that answers natural-language questions about member data using advanced hybrid retrieval and dual-path query routing.

**Example queries:**
- *"When is Sophia traveling to Paris?"*
- *"Which clients requested the same restaurants?"*
- *"What are Layla's seating preferences?"*

Built with **modern RAG architecture**, recall.ai combines semantic search, keyword matching, and knowledge graphs to deliver accurate, context-aware responses.

---

## ✨ Key Features

### 🚀 **Dual-Path Query Routing**
- **LOOKUP Path:** Direct member information retrieval (e.g., "Sophia's preferences")
- **ANALYTICS Path:** Pattern discovery and aggregation (e.g., "most popular destinations")
- LLM-powered routing ensures queries take the optimal path

### 🔍 **Hybrid Retrieval System**
- **Semantic Search (Qdrant):** Understands conceptual similarity
- **BM25 Keyword Search:** Captures exact matches and names
- **Knowledge Graph:** Connects related entities and relationships
- **RRF Fusion:** Intelligently combines results from all three methods

### 💬 **Natural Language Interface**
- Clean, modern UI with northern lights theme
- Real-time query processing with thinking animations
- Source attribution for transparency
- Markdown-formatted responses

### ⚡ **Production Optimized**
- FastAPI backend with async processing
- Qdrant Cloud vector database
- Groq API for fast LLM inference
- Docker-ready, single-deployment architecture

---

## 🛠️ Tech Stack

### Backend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | FastAPI + Uvicorn | High-performance async API |
| **LLM** | Groq (Llama-3.3-70b) | Query routing & answer generation |
| **Vector DB** | Qdrant Cloud | Semantic search |
| **Embeddings** | FastEmbed (ONNX) | Lightweight embeddings (200MB) |
| **Keyword Search** | BM25 (Rank-BM25) | Exact keyword matching |
| **Knowledge Graph** | NetworkX | Entity relationships |

### Frontend
| Component | Technology |
|-----------|-----------|
| **UI** | Pure HTML/CSS/JS (no framework) |
| **Design** | Custom CSS with glass morphism |
| **Theme** | Northern lights color palette |
| **Icons** | Inline SVG |

---

## 🏗️ Architecture

### System Overview

```
┌──────────────────────────────────────────────────────┐
│                   recall.ai System                    │
├──────────────────────────────────────────────────────┤
│                                                      │
│  FastAPI Backend (api.py)                            │
│  ├── POST /ask       → QA System                     │
│  ├── GET /health     → Status Check                  │
│  └── GET /           → Serve Frontend                │
│                                                      │
│  QA System Pipeline                                  │
│  ├── Query Processor    (LLM-based routing)          │
│  │   ├── Route: LOOKUP or ANALYTICS                  │
│  │   └── Classification & Weight Assignment          │
│  │                                                   │
│  ├── LOOKUP Path (RAG Pipeline)                      │
│  │   ├── Hybrid Retriever (3 methods in parallel)    │
│  │   │   ├── Qdrant (semantic)                       │
│  │   │   ├── BM25 (keywords)                         │
│  │   │   └── Knowledge Graph (relationships)         │
│  │   ├── RRF Fusion                                  │
│  │   └── LLM Answer Generation                       │
│  │                                                   │
│  └── ANALYTICS Path (Graph Analytics)                │
│      ├── Entity Extraction (LLM)                     │
│      ├── Graph Querying                              │
│      ├── Aggregation (GROUP BY, COUNT, RANK)         │
│      └── LLM Answer Generation                       │
│                                                      │
│  Data Layer                                          │
│  ├── Qdrant Cloud (vector embeddings)               │
│  ├── BM25 Index (inverted index)                    │
│  └── Knowledge Graph (user → entity triples)        │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### Query Flow Example

**Query:** *"When is Sophia traveling to Paris?"*

```
1. Query Processing
   ├── Route: LOOKUP (specific member query)
   ├── Classification: ENTITY_SPECIFIC_PRECISE
   └── Weights: {semantic: 1.0, bm25: 1.2, graph: 1.1}

2. Hybrid Retrieval
   ├── Qdrant: Find "Paris", "travel", "trip" (semantic)
   ├── BM25: Match "Sophia", "Paris" (keywords)
   └── Graph: Get Sophia → PLANNING_TRIP_TO → Paris

3. RRF Fusion
   └── Combine & rank messages by weighted score

4. LLM Generation
   └── Generate natural answer with sources

5. Response
   └── "Sophia Al-Farsi is traveling to Paris next Friday..."
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Qdrant Cloud account ([free tier](https://cloud.qdrant.io))
- Groq API key ([free tier](https://console.groq.com))

### Installation

```bash
# 1. Clone repository
git clone https://github.com/SumanthKonjeti007/recall-ai.git
cd recall-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Add your API keys to .env:
# GROQ_API_KEY=your_key_here
# QDRANT_URL=https://your-cluster.qdrant.io
# QDRANT_API_KEY=your_key_here

# 5. Run server
python api.py
```

**Open:** http://localhost:8000

---

## 📊 Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Query Latency** | ~2-3s | Including LLM inference |
| **Accuracy** | High | Hybrid retrieval outperforms single methods |
| **Memory Usage** | ~250MB | FastEmbed optimization |
| **Cost per Query** | ~$0.003 | Groq API pricing |
| **Startup Time** | <1s | Fast cold starts |

---

## 📁 Project Structure

```
recall-ai/
├── api.py                    # FastAPI backend
├── requirements.txt          # Python dependencies
├── Procfile                  # Deployment config
│
├── src/                      # Runtime modules (11 files)
│   ├── qa_system.py          # Main QA pipeline
│   ├── query_processor.py    # LLM-based routing & classification
│   ├── hybrid_retriever.py   # 3-method retrieval + RRF fusion
│   ├── answer_generator.py   # LLM answer generation
│   ├── graph_analytics.py    # Analytics path pipeline
│   ├── result_composer.py    # Multi-query composition
│   ├── bm25_search.py        # Keyword search
│   ├── qdrant_search.py      # Vector search
│   ├── knowledge_graph.py    # Graph queries
│   ├── name_resolver.py      # Entity resolution
│   └── temporal_analyzer.py  # Date extraction
│
├── scripts/                  # Preprocessing scripts
│   ├── data_ingestion.py     # Fetch raw data
│   ├── embeddings.py         # Generate vectors
│   └── ... (entity extraction, etc.)
│
├── static/                   # Frontend
│   ├── index.html            # Landing page
│   └── app.html              # Main application
│
└── data/                     # Indexes & embeddings
    ├── embeddings/           # Qdrant data
    ├── bm25/                 # BM25 index
    └── knowledge_graph.pkl   # NetworkX graph
```

---

## 🎓 Project Story

recall.ai evolved from a technical assessment into a full-featured production system, demonstrating end-to-end ML engineering skills.

### Key Technical Achievements

1. **Hybrid Retrieval Architecture**
   - Designed and implemented 3-method retrieval with RRF fusion
   - Achieved significant accuracy improvements over single-method baselines
   - Optimized for both precision (LOOKUP) and recall (ANALYTICS)

2. **Intelligent Query Routing**
   - Built LLM-powered routing system with 95%+ accuracy
   - Dual-path architecture handles diverse query types
   - Dynamic weight assignment based on query classification

3. **Production Optimization**
   - Migrated from 4GB sentence-transformers to 200MB FastEmbed
   - Switched from Mistral AI to Groq for better rate limits
   - Dockerized single-deployment architecture

4. **Full-Stack Development**
   - Custom glass-morphism UI with northern lights theme
   - Real-time streaming responses with thinking animations
   - Mobile-responsive design with touch-optimized controls

### Evolution from Aurora Assessment

This project started as a take-home assessment for Aurora and was transformed into a polished, production-ready system with:
- ✅ Complete rebranding (Aurora → recall.ai)
- ✅ Enhanced UI/UX with modern design
- ✅ Advanced dual-path routing architecture
- ✅ Production-grade error handling and logging
- ✅ Comprehensive documentation

---

## 🔮 Future Enhancements (Phase 2)

- [ ] Chat history persistence
- [ ] Advanced filtering (date ranges, entities)
- [ ] Multi-language support
- [ ] Export functionality (PDF, CSV)
- [ ] Analytics dashboard
- [ ] User authentication & multi-tenancy

---

## 🤝 Contributing

This is primarily a portfolio project, but feedback and suggestions are welcome!

- 🐛 Found a bug? [Open an issue](https://github.com/SumanthKonjeti007/recall-ai/issues)
- 💡 Have an idea? Start a [discussion](https://github.com/SumanthKonjeti007/recall-ai/discussions)
- 🔧 Want to contribute? Fork and submit a PR!

---

## 📜 License

MIT License - Free to use for learning and personal projects.

---

## 👤 Author

**Sumanth Konjeti**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/sumanthkonjeti)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/SumanthKonjeti007)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-green)](#)

*Building intelligent systems at the intersection of AI, data, and user experience.*

---

<div align="center">

**Built with** 💙 **using FastAPI, Groq, Qdrant, and modern RAG techniques**

⭐ Star this repo if you find it useful!

</div>
