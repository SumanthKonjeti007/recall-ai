# Recall - Personal Memory Assistant

> Query your personal activity data using natural language, powered by hybrid RAG

![Recall](https://img.shields.io/badge/Status-Live-brightgreen)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-blue)
![React](https://img.shields.io/badge/React-18+-61DAFB)

---

## What is Recall?

Recall is an AI-powered personal memory assistant that lets you query your activity data using natural language. Ask questions like "What Italian restaurants did I visit?" or "How many messages about family last month?" and get accurate, source-cited answers instantly.

**Live Demo:** [Coming Soon]

---

## Features

- **Natural Language Queries** - Ask questions in plain English
- **Hybrid Retrieval** - Combines vector search (Qdrant), BM25 keyword matching, and knowledge graphs
- **Source Attribution** - Every answer includes citations to source data
- **Modern UI** - Built with React, Tailwind CSS, and shadcn/ui components
- **Production-Ready** - Optimized for deployment with FastEmbed (200MB vs 4GB alternatives)

---

## Tech Stack

### Backend
- **Framework:** FastAPI + Uvicorn
- **LLM:** Groq (Llama-3.3-70b-versatile)
- **Vector Database:** Qdrant Cloud
- **Embeddings:** FastEmbed (ONNX-based, lightweight)
- **Keyword Search:** BM25
- **Knowledge Graph:** NetworkX

### Frontend
- **Framework:** React 18 + Vite
- **Styling:** Tailwind CSS
- **Components:** shadcn/ui
- **Icons:** Lucide React

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Qdrant Cloud account
- Groq API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SumanthKonjeti007/recall-ai.git
   cd recall-ai
   ```

2. **Set up backend**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt

   # Create .env file
   cp .env.example .env
   # Add your GROQ_API_KEY, QDRANT_URL, and QDRANT_API_KEY
   ```

3. **Set up frontend**
   ```bash
   cd frontend
   npm install
   ```

4. **Run locally**
   ```bash
   # Terminal 1: Start backend
   python api.py

   # Terminal 2: Start frontend
   cd frontend
   npm run dev
   ```

5. **Open** `http://localhost:5173`

---

## Architecture

### Hybrid Retrieval Pipeline

```
User Query
    ↓
Query Classifier (LLM)
    ↓
┌──────────┬──────────┬──────────┐
│  Vector  │   BM25   │   Graph  │
│  Search  │  Search  │ Traversal│
└──────────┴──────────┴──────────┘
    ↓
Cross-Encoder Reranking
    ↓
Context Assembly
    ↓
LLM Answer Generation
    ↓
Formatted Response + Sources
```

### Why Hybrid?
- **Vector Search:** Handles semantic similarity ("Italian restaurants" → "trattoria", "pizzeria")
- **BM25:** Captures exact keyword matches ("policy 4352")
- **Knowledge Graph:** Connects related activities and entities

This combination dramatically improves accuracy over single-method retrieval.

---

## Deployment

### Build Frontend
```bash
cd frontend
npm run build
```

Built files go to `dist/` and are served by FastAPI.

### Deploy to Render

1. Push to GitHub
2. Create new Web Service on Render
3. Connect your GitHub repo
4. Set environment variables:
   - `GROQ_API_KEY`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
5. Deploy!

**Build Command:** `pip install -r requirements.txt`
**Start Command:** `uvicorn api:app --host 0.0.0.0 --port $PORT`

---

## Example Queries

Try these:
- "What Italian restaurants did I visit?"
- "How many messages about family?"
- "Show me activities from last month"
- "What did I do related to travel?"
- "Find messages mentioning work deadlines"

---

## Performance

- **Query Latency:** ~2 seconds average
- **Memory Usage:** ~250MB RAM (thanks to FastEmbed optimization)
- **Cost per Query:** ~$0.003 (Groq API)
- **Accuracy:** Hybrid retrieval significantly outperforms single-method approaches

---

## Project Story

This project evolved from a take-home assessment into a full-fledged production system. Key learnings:

1. **Model size matters** - Switched from sentence-transformers (4GB) to FastEmbed (200MB) to fit deployment constraints
2. **Hybrid retrieval isn't optional** - Pure vector search failed on counting queries; pure keyword search missed semantic variations
3. **Rate limits bite** - Migrated from Mistral to Groq for better free-tier limits

---

## Contributing

This is a portfolio project, but suggestions and feedback are welcome! Open an issue or reach out.

---

## License

MIT License - feel free to use this as learning material or fork for your own projects.

---

## Author

**Sumanth Konjeti**
[LinkedIn](https://linkedin.com/in/sumanthkonjeti) | [GitHub](https://github.com/SumanthKonjeti007)

Built with ❤️ using FastAPI, React, Qdrant, and Groq.
