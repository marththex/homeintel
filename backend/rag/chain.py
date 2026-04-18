"""
rag/chain.py — LangChain RAG chain: retrieve → format context → generate.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import settings
from rag.prompts import SYSTEM_TEMPLATE
from rag.retriever import retrieve

logger = logging.getLogger(__name__)


class RAGChain:
    """
    Stateless RAG chain.

    run() retrieves relevant chunks from ChromaDB, formats them as context,
    calls the Ollama LLM, and returns the answer alongside the source docs.
    """

    def __init__(self) -> None:
        self._llm = ChatOllama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url_str,
        )
        logger.info("RAGChain ready — model=%s", settings.ollama_llm_model)

    def run(
        self,
        question: str,
        modality_filter: Optional[str] = None,
    ) -> dict:
        """
        Execute the full RAG pipeline.

        Returns:
            {
                "answer": str,
                "docs":   list[Document],
            }
        """
        docs: list[Document] = retrieve(question, modality_filter)

        if docs:
            context_parts = [
                f"[Source: {doc.metadata.get('file_name', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            ]
            context = "\n\n---\n\n".join(context_parts)
        else:
            context = "(no relevant context found in your files)"

        messages = [
            SystemMessage(content=SYSTEM_TEMPLATE.format(context=context)),
            HumanMessage(content=question),
        ]

        logger.info(
            "Calling LLM — model=%s docs=%d question=%r",
            settings.ollama_llm_model,
            len(docs),
            question[:80],
        )
        response = self._llm.invoke(messages)

        return {
            "answer": response.content,
            "docs": docs,
        }


_chain: Optional[RAGChain] = None


def get_chain() -> RAGChain:
    """Return the module-level RAGChain singleton."""
    global _chain
    if _chain is None:
        _chain = RAGChain()
    return _chain
