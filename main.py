import argparse
import sys

from config import Config
from rag_engine import RAGEngine


def main():
    parser = argparse.ArgumentParser(
        description="RAG-based Q&A for C++ codebases"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a C++ repository")
    ingest_parser.add_argument("path", help="Path to C++ repository")

    query_parser = subparsers.add_parser("query", help="Query the codebase")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument(
        "-k",
        type=int,
        default=4,
        help="Number of chunks to retrieve (default: 4)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        config = Config()
        llm_provider = config.get_llm_provider()
        embedding_provider = config.get_embedding_provider()
        rag_config = config.get_rag_config()

        engine = RAGEngine(llm_provider, embedding_provider, rag_config)

        if args.command == "ingest":
            engine.ingest_repository(args.path)

        elif args.command == "query":
            answer = engine.query(args.question, args.k)
            print(f"\nAnswer:\n{answer}\n")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
