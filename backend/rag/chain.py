"""
rag/chain.py — LangChain RAG chain: retrieve → format context → generate.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import settings
from rag.prompts import SYSTEM_TEMPLATE, IMAGE_SEARCH_TEMPLATE
from rag.retriever import retrieve
from vectorstore.qdrant import Modality
from security import redact_secrets

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
        top_k: Optional[int] = None,
    ) -> dict:
        """
        Execute the full RAG pipeline.

        Returns:
            {
                "answer": str,
                "docs":   list[Document],
            }
        """
        docs: list[Document] = retrieve(question, modality_filter, top_k)

        if docs:
            def _content(doc: Document) -> str:
                text = doc.page_content
                # Defense in depth: scrub secrets from already-indexed chunks
                # too, so the LLM never even sees a raw credential.
                return redact_secrets(text) if settings.redact_secrets else text

            context_parts = [
                f"[Source: {doc.metadata.get('file_name', 'unknown')}]\n{_content(doc)}"
                for doc in docs
            ]
            context = "\n\n---\n\n".join(context_parts)
        else:
            context = "(no relevant context found in your files)"

        # Image search is a "show me matching photos" task, not strict Q&A.
        # Use the photo-search prompt when the query targets images (explicit
        # filter, or every retrieved doc is an image) so the model presents the
        # matches instead of refusing.
        is_image_query = bool(docs) and (
            modality_filter == Modality.IMAGE.value
            or all(d.metadata.get("modality") == Modality.IMAGE.value for d in docs)
        )
        template = IMAGE_SEARCH_TEMPLATE if is_image_query else SYSTEM_TEMPLATE

        messages = [
            SystemMessage(content=template.format(context=context)),
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
