"""
PDF Knowledge Loader for CognitiveAI

Uses pdfplumber for PDF text extraction, chunks content, generates embeddings,
and stores in the long-term memory system.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import pdfplumber
import logging
from .memory.ltm import LTMManager

logger = logging.getLogger(__name__)


class PDFLoader:
    """
    PDF Knowledge Loader that extracts, chunks, and stores PDF content.

    Uses Unstructured for robust PDF parsing and integrates with LTM for storage.
    """

    def __init__(self, ltm_manager: LTMManager, chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize PDF Loader.

        Args:
            ltm_manager: Long-term memory manager instance
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
        """
        self.ltm_manager = ltm_manager
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdf(self, pdf_path: str, metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
        """
        Load and process a PDF file.

        Args:
            pdf_path: Path to the PDF file
            metadata: Additional metadata for the PDF

        Returns:
            PDF document ID
        """
        pdf_file_path = Path(pdf_path)
        if not pdf_file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_file_path}")

        logger.info(f"Loading PDF: {pdf_path}")

        
        doc_id = doc_id or str(uuid.uuid4())

        try:
            
            full_text = self._extract_text_pdfplumber(pdf_path)

            
            if user_id:
                try:
                    from backend.redis_client import get_redis
                    import os
                    r = get_redis()
                    ttl = int(os.getenv("STM_TTL", "1800"))
                    key = f"pdf:{user_id}:{doc_id}"
                    
                    max_bytes = 200 * 1024
                    raw_bytes = full_text.encode("utf-8")
                    if len(raw_bytes) > max_bytes:
                        
                        raise ValueError("Extracted PDF text exceeds allowed size of 200 KB")
                    
                    r.set(key, full_text)
                    r.expire(key, ttl)
                except Exception:
                    
                    logger.warning("Failed to store extracted PDF text in Redis for user %s doc %s", user_id, doc_id)

            
            if self.ltm_manager is None:
                logger.warning("LTM manager not configured: skipping storage, returning extracted text only")
                return doc_id

            
            doc_metadata = self._extract_pdf_metadata(pdf_file_path, full_text, metadata or {})

            
            chunks = self._chunk_text(full_text)

            
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = doc_metadata.copy()
                chunk_metadata.update({
                    "document_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_type": "pdf_content",
                    "user_id": user_id
                })

                chunk_id = self.ltm_manager.add_memory(
                    content=chunk,
                    memory_type="pdf_knowledge",
                    metadata=chunk_metadata,
                    importance=0.6,  
                    user_id=user_id
                )
                chunk_ids.append(chunk_id)

            
            summary_content = self._generate_document_summary(full_text, doc_metadata)
            self.ltm_manager.add_memory(
                content=summary_content,
                memory_type="pdf_summary",
                metadata={
                    "document_id": doc_id,
                    "chunk_ids": chunk_ids,
                    "total_chunks": len(chunks),
                    "document_type": "pdf",
                    "user_id": user_id
                },
                importance=0.8,  
                user_id=user_id
            )

            logger.info(f"Successfully loaded PDF {pdf_file_path.name} with {len(chunks)} chunks")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to load PDF {pdf_path}: {e}")
            raise

    def _extract_text_pdfplumber(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfplumber."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
                full_text = "\n\n".join(text_parts)
                return full_text
        except Exception as e:
            logger.error(f"Failed to extract text with pdfplumber: {e}")
            raise

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract raw text from a PDF without storing it.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text string
        """
        pdf_file_path = Path(pdf_path)
        if not pdf_file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_file_path}")

        try:
            return self._extract_text_pdfplumber(pdf_path)
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
            raise

    def _elements_to_text(self, elements) -> str:
        """Convert Unstructured elements to plain text."""
        text_parts = []

        for element in elements:
            element_text = str(element)

            if hasattr(element, "category"):
                if element.category == "Title":
                    element_text = f"\n{element_text}\n"
                elif element.category == "Header":
                    element_text = f"\n{element_text}\n"
                elif element.category == "Table":
                    element_text = f"\n[TABLE]\n{element_text}\n[/TABLE]\n"

            text_parts.append(element_text)

        return "\n".join(text_parts)


    def _extract_pdf_metadata(self, pdf_path: Path, full_text: str, user_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from PDF."""
        metadata = {
            "filename": pdf_path.name,
            "filepath": str(pdf_path),
            "file_size": pdf_path.stat().st_size,
            "source_type": "pdf",
            "upload_timestamp": self._get_timestamp()
        }

        
        lines = full_text.split('\n')
        for line in lines[:5]:
            if line.strip() and len(line.strip()) > 5:
                metadata["title"] = line.strip()[:100]
                break

        
        metadata.update(user_metadata)

        return metadata

    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text into manageable pieces."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append(text[start:])
                break

            
            chunk_end = self._find_chunk_boundary(text, start, end)
            chunk = text[start:chunk_end]

            
            if len(chunk) < 100 and end < len(text):
                chunk_end = min(end + 200, len(text))
                chunk = text[start:chunk_end]

            chunks.append(chunk.strip())
            start = chunk_end - self.chunk_overlap

            
            if start >= len(text):
                break

        return chunks

    def _find_chunk_boundary(self, text: str, start: int, end: int) -> int:
        """Find a good boundary to break the chunk (sentence or word)."""
        
        search_start = max(start, end - 100)

        for i in range(end - 1, search_start - 1, -1):
            if text[i] in '.!?\n':
                return i + 1

        
        for i in range(end - 1, search_start - 1, -1):
            if text[i] in ' \t\n':
                return i + 1

        
        return end

    from typing import Dict, Any

    def _generate_document_summary(self, full_text: str, metadata: Dict[str, Any]) -> str:
        """Generate a summary of the document."""

        title = metadata.get("title") or metadata.get("filename") or "Unknown Document"

        # Hard cutoff summary preview
        summary_text = full_text[:500].strip()

        # Extract probable section headers from the first 20 lines
        lines = full_text.split("\n")
        sections = []

        for line in lines[:20]:
            line = line.strip()
            if not line:
                continue

            # Common section header patterns
            if (
                line.isupper()
                or line.startswith(("#", "##", "Section", "Chapter"))
            ):
                sections.append(line)

        summary = f"Document: {title}\n\nSummary: {summary_text}..."

        if sections:
            summary += f"\n\nKey Sections: {', '.join(sections[:3])}"

        summary += f"\n\nTotal length: {len(full_text)} characters"

        return summary


    def search_pdf_knowledge(self, query: str, document_id: Optional[str] = None,
                           limit: int = 10, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant PDF knowledge.

        Args:
            query: Search query
            document_id: Optional document ID filter
            limit: Maximum results

        Returns:
            Search results
        """
        
        metadata_filters = {}
        if document_id:
            metadata_filters["document_id"] = document_id

        return self.ltm_manager.search_memories(
            query=query or (document_id or ""),
            memory_type="pdf_knowledge",
            limit=limit,
            user_id=user_id,
            metadata_filters=metadata_filters if metadata_filters else None
        )

    def get_pdf_documents(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of uploaded PDF documents for a user (if provided)."""
        
        results = self.ltm_manager.search_memories(
            query="pdf document summary",
            memory_type="pdf_summary",
            limit=100,  
            user_id=user_id
        )

        documents = []
        for result in results:
            if result.get("memory_type") == "pdf_summary":
                metadata = result.get("metadata", {})
                
                if user_id and metadata.get("user_id") != user_id:
                    continue
                documents.append({
                    "id": metadata.get("document_id"),
                    "title": metadata.get("title", metadata.get("filename", "Unknown")),
                    "filename": metadata.get("filename"),
                    "chunk_count": metadata.get("total_chunks", 0),
                    "upload_timestamp": metadata.get("upload_timestamp")
                })

        return documents

    def delete_pdf(self, document_id: str, user_id: Optional[str] = None):
        """
        Delete a PDF document and all its chunks.

        Args:
            document_id: Document ID to delete
        """
        
        
        try:
            
            results = self.ltm_manager.search_memories(
                query=f"document_id:{document_id}",
                limit=1000,
                user_id=user_id
            )

            
            for result in results:
                if result.get("metadata", {}).get("document_id") == document_id:
                    if user_id and result.get("metadata", {}).get("user_id") != user_id:
                        continue
                    self.ltm_manager.delete_memory(result["id"])

            
            summary_results = self.ltm_manager.search_memories(
                query=f"document_id:{document_id}",
                memory_type="pdf_summary",
                limit=10,
                user_id=user_id
            )

            for result in summary_results:
                if result.get("metadata", {}).get("document_id") == document_id:
                    if user_id and result.get("metadata", {}).get("user_id") != user_id:
                        continue
                    self.ltm_manager.delete_memory(result["id"])

            logger.info(f"Deleted PDF document: {document_id}")

        except Exception as e:
            logger.error(f"Failed to delete PDF {document_id}: {e}")
            raise

    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
