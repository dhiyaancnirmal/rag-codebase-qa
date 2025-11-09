import os
from pathlib import Path
from typing import Dict, List

import chromadb
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import LLMProvider, EmbeddingProvider


class RAGEngine:
    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        rag_config: Dict
    ):
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.rag_config = rag_config
        self.vectorstore = None

        self.chunk_size = rag_config.get("chunk_size", 1000)
        self.chunk_overlap = rag_config.get("chunk_overlap", 200)
        self.db_path = rag_config.get("db_path", "./chroma_db")
        self.collection_name = rag_config.get("collection_name", "codebase")

        self.cpp_extensions = {".cpp", ".h", ".hpp", ".cc", ".cxx", ".c"}

    def ingest_repository(self, repo_path: str) -> None:
        print(f"Ingesting repository: {repo_path}")

        documents = self._load_cpp_files(repo_path)
        if not documents:
            print("No C++ files found in repository")
            return

        print(f"Found {len(documents)} C++ files")

        chunks = self._split_documents(documents)
        print(f"Split into {len(chunks)} chunks")

        self._store_in_vectordb(chunks)
        print("Ingestion complete")

    def _load_cpp_files(self, repo_path: str) -> List[Document]:
        documents = []
        repo_path_obj = Path(repo_path)

        for file_path in repo_path_obj.rglob("*"):
            if file_path.suffix in self.cpp_extensions and file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    doc = Document(
                        page_content=content,
                        metadata={"source": str(file_path)}
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        return documents

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.CPP,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(documents)

    def _store_in_vectordb(self, chunks: List[Document]) -> None:
        embeddings = self.embedding_provider.get_embeddings()

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=self.collection_name,
            persist_directory=self.db_path
        )

    def _load_vectorstore(self) -> None:
        if self.vectorstore is None:
            embeddings = self.embedding_provider.get_embeddings()

            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=self.db_path
            )

    def query(self, question: str, k: int = 4) -> str:
        try:
            self._load_vectorstore()

            if self.vectorstore is None:
                return "Error: No codebase ingested yet. Run ingest first."

            llm = self.llm_provider.get_llm()
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

            template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer:"""

            prompt = ChatPromptTemplate.from_template(template)

            def format_docs(docs):
                return "\n\n".join([d.page_content for d in docs])

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            result = chain.invoke(question)
            return result

        except Exception as e:
            return f"Error during query: {e}"
