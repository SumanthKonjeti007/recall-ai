# Data Preprocessing Scripts

These scripts are used for **one-time data generation** and are not part of the runtime API.

## Scripts Overview

| Script | Purpose | Output |
|--------|---------|--------|
| `data_ingestion.py` | Fetch raw messages from API | `data/raw_messages.json` |
| `embeddings.py` | Generate vector embeddings (FAISS) | `data/embeddings/` |
| `entity_extraction.py` | Extract entities from messages | Knowledge graph data |
| `entity_extraction_gliner.py` | GLiNER-based entity extraction | Alternative extraction |
| `hybrid_extractor.py` | Hybrid extraction approach | Combined extraction |
| `llm_extractor.py` | LLM-based entity extraction | LLM-powered extraction |
| `rule_based_extractor.py` | Rule-based extraction | Pattern-based extraction |

## Usage

These scripts were used during initial setup to process the data. They are included for reference and reproducibility.

```bash
# Example: Fetch messages from API
python scripts/data_ingestion.py

# Example: Generate embeddings
python scripts/embeddings.py
```

## Note

The generated data files are already included in `data/` directory. You don't need to run these scripts unless you want to regenerate the data from scratch.
