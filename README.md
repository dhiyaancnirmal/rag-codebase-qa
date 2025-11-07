# RAG-Codebase-QA

A RAG-based Q&A system for C++ codebases using Google Gemini.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your Google API key:
   ```bash
   cp .env.example .env
   ```

3. Edit `.env` and add your API key.

## Usage

Ingest a C++ repository:
```bash
python main.py ingest /path/to/cpp/repo
```

Query the codebase:
```bash
python main.py query "How does the memory allocation work?"
```

## Configuration

Edit `config.yaml` to adjust:
- LLM model and temperature
- Embedding model
- Chunk size and overlap
- Database path
