"""Document ingestion pipeline — load, chunk, embed, store."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from l90 import config
from l90.vectordb.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

# ── Supported file types ───────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def _load_pdf(path: Path) -> str:
    """Load text from a PDF file."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _load_txt(path: Path) -> str:
    """Load text from a plain text or markdown file."""
    return path.read_text(encoding="utf-8")


def _load_docx(path: Path) -> str:
    """Load text from a DOCX file."""
    from docx import Document  # python-docx

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


# ── Loader dispatch ────────────────────────────────────────────
_LOADERS = {
    ".pdf": _load_pdf,
    ".txt": _load_txt,
    ".md": _load_txt,
    ".docx": _load_docx,
}


class IngestionPipeline:
    """End-to-end document ingestion: load → chunk → store in ChromaDB.

    Embedding is handled by the ChromaDB collection's embedding function.
    """

    def __init__(self, store: ChromaStore | None = None) -> None:
        self._store = store or ChromaStore()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def ingest(
        self,
        file_path: str | Path,
        collection_name: str,
        *,
        owner: str = "",
        workspace_id: str = "",
        security_level: str = "standard",
        domain: str = "",
        source_type: str = "user_upload",
        extra_metadata: dict[str, Any] | None = None,
    ) -> int:
        """Ingest a single document into the specified collection.

        Args:
            file_path: Path to the document.
            collection_name: Target ChromaDB collection.
            owner: User ID of the document owner.
            workspace_id: Workspace ID (for shared collections).
            security_level: Classification level.
            domain: Subject domain.
            source_type: e.g., "user_upload", "approved_library".
            extra_metadata: Additional metadata to attach.

        Returns:
            Number of chunks stored.
        """
        path = Path(file_path)

        # Validate file type
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Supported: {SUPPORTED_EXTENSIONS}"
            )

        # Load document text
        loader = _LOADERS[path.suffix.lower()]
        raw_text = loader(path)

        if not raw_text.strip():
            logger.warning("Empty document: %s", path)
            return 0

        # Generate document ID (deterministic hash of content)
        doc_id = hashlib.sha256(raw_text.encode()).hexdigest()[:16]

        # Split into chunks
        chunks = self._splitter.split_text(raw_text)
        logger.info("Split '%s' into %d chunks", path.name, len(chunks))

        # Build metadata for each chunk
        base_meta: dict[str, Any] = {
            "source": path.name,
            "source_type": source_type,
            "owner": owner,
            "workspace_id": workspace_id,
            "security_level": security_level,
            "domain": domain,
            "document_id": doc_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if extra_metadata:
            base_meta.update(extra_metadata)

        metadatas = []
        ids = []
        for i, _chunk in enumerate(chunks):
            chunk_meta = {**base_meta, "chunk_index": i}
            metadatas.append(chunk_meta)
            ids.append(f"{doc_id}_chunk_{i}")

        # Store in ChromaDB (embedding handled by collection's embedding function)
        await self._store.add_documents(
            collection_name=collection_name,
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(
            "Ingested '%s' → %d chunks into '%s'",
            path.name,
            len(chunks),
            collection_name,
        )
        return len(chunks)

    async def ingest_approved_library_document(
        self,
        file_path: str | Path,
        *,
        document_name: str = "",
        version: str = "1.0",
        approval_authority: str = "",
        domain: str = "",
        classification: str = "internal",
    ) -> int:
        """Convenience method for ingesting into the Approved Library collection.

        Only system administrators should call this.
        """
        path = Path(file_path)
        return await self.ingest(
            file_path=path,
            collection_name=config.COLLECTION_APPROVED_LIBRARY,
            source_type="approved_library",
            domain=domain,
            security_level=classification,
            extra_metadata={
                "document_name": document_name or path.stem,
                "version": version,
                "approval_authority": approval_authority,
                "classification": classification,
            },
        )
