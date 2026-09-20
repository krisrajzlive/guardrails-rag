"""Builds and loads the LlamaIndex vector store over the vendor NDA PDF."""
from __future__ import annotations

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader

from . import config


def _configure_embeddings() -> None:
    Settings.embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)


def build_index(force: bool = False) -> VectorStoreIndex:
    _configure_embeddings()
    if config.STORAGE_DIR.exists() and not force:
        storage_context = StorageContext.from_defaults(persist_dir=str(config.STORAGE_DIR))
        return load_index_from_storage(storage_context)

    documents = PDFReader().load_data(file=config.SOURCE_PDF)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=str(config.STORAGE_DIR))
    return index


def get_retriever(top_k: int = config.TOP_K) -> BaseRetriever:
    index = build_index()
    return index.as_retriever(similarity_top_k=top_k)


def retrieve_context(query: str, top_k: int = config.TOP_K) -> list[str]:
    retriever = get_retriever(top_k=top_k)
    nodes = retriever.retrieve(query)
    return [n.get_content() for n in nodes]


if __name__ == "__main__":
    build_index(force=True)
    print(f"Index persisted to {config.STORAGE_DIR}")
