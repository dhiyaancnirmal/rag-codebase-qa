# RAG-Codebase-QA

Q&A system for C++ codebases using retrieval-augmented generation.

## Setup

Install dependencies and configure your API key:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your Google API key.

## Usage

Ingest a C++ repository:
```bash
python main.py ingest /path/to/cpp/repo
```

Query the codebase:
```bash
python main.py query "your question here"
```

## Configuration

The `config.yaml` file controls chunk size, model selection, and database settings. The architecture uses abstract provider interfaces, making it easy to swap LLM or embedding providers.
