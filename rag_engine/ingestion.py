"""
Document ingestion and chunking for the RAG pipeline.

Loads regulatory documents (PDF, TXT, MD) and splits them
into overlapping chunks for embedding and retrieval.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)
config = Config()


class DocumentChunk:
    """
    Represents a single chunk of a document with metadata.

    Attributes:
        text: The chunk text content.
        metadata: Source info (filename, page, chunk index).
        chunk_id: Unique identifier for deduplication.
    """

    def __init__(self, text: str, metadata: Dict[str, str]) -> None:
        self.text = text
        self.metadata = metadata
        self.chunk_id = self._compute_id()

    def _compute_id(self) -> str:
        """Compute deterministic ID from content hash."""
        content = f"{self.metadata.get('source', '')}__{self.text[:200]}"
        return hashlib.md5(content.encode()).hexdigest()

    def __repr__(self) -> str:
        return f"DocumentChunk(source={self.metadata.get('source', '?')}, len={len(self.text)})"


class TextSplitter:
    """
    Recursive character text splitter with overlap.

    Strategy: Split on paragraph breaks first, then sentences,
    then spaces. Overlap ensures context continuity across chunks.

    Design decision: chunk_size=512 with overlap=64
    - 512 tokens ≈ 1 dense paragraph. Optimal for MiniLM embeddings
      which have a 256-token sweet spot but handle 512 well.
    - 64 overlap prevents losing context at chunk boundaries,
      critical for regulatory text where clauses span paragraphs.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Input text to split.

        Returns:
            List of text chunks.
        """
        chunks: List[str] = []
        current_chunks = [text]

        for separator in self.separators:
            new_chunks: List[str] = []
            for chunk in current_chunks:
                if len(chunk) <= self.chunk_size:
                    new_chunks.append(chunk)
                else:
                    parts = chunk.split(separator)
                    new_chunks.extend(parts)
            current_chunks = new_chunks

        # Merge small chunks and apply overlap
        return self._merge_with_overlap(current_chunks)

    def _merge_with_overlap(self, parts: List[str]) -> List[str]:
        """Merge small parts into chunks with overlap."""
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if current_len + len(part) > self.chunk_size and current:
                chunk_text = " ".join(current)
                chunks.append(chunk_text)

                # Keep overlap from end of current chunk
                overlap_parts: List[str] = []
                overlap_len = 0
                for p in reversed(current):
                    if overlap_len + len(p) > self.chunk_overlap:
                        break
                    overlap_parts.insert(0, p)
                    overlap_len += len(p)

                current = overlap_parts
                current_len = overlap_len

            current.append(part)
            current_len += len(part)

        if current:
            chunks.append(" ".join(current))

        return chunks


class DocumentIngester:
    """
    Loads and chunks regulatory documents for the RAG pipeline.

    Supports PDF, TXT, and MD formats. Each document is split
    into overlapping chunks with source metadata preserved.
    """

    def __init__(self) -> None:
        rag_config = config.get_section("rag_engine")
        chunking_config = rag_config.get("chunking", {})

        self.source_dir = Path(
            rag_config.get("documents", {}).get("source_dir", "data/regulatory_docs")
        )
        self.supported_extensions = rag_config.get("documents", {}).get(
            "supported_extensions", [".pdf", ".txt", ".md"]
        )
        self.splitter = TextSplitter(
            chunk_size=chunking_config.get("chunk_size", 512),
            chunk_overlap=chunking_config.get("chunk_overlap", 64),
            separators=chunking_config.get("separators"),
        )

    def ingest_all(self) -> List[DocumentChunk]:
        """
        Ingest all documents from the source directory.

        Returns:
            List of DocumentChunk objects.
        """
        if not self.source_dir.exists():
            logger.warning(f"Document directory not found: {self.source_dir}")
            self.source_dir.mkdir(parents=True, exist_ok=True)
            return []

        all_chunks: List[DocumentChunk] = []
        files = [
            f for f in self.source_dir.iterdir()
            if f.suffix.lower() in self.supported_extensions
        ]

        logger.info(f"Found {len(files)} documents to ingest")

        for file_path in sorted(files):
            try:
                chunks = self._ingest_file(file_path)
                all_chunks.extend(chunks)
                logger.info(f"  {file_path.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to ingest {file_path.name}: {e}")

        logger.info(f"Total chunks ingested: {len(all_chunks)}")
        return all_chunks

    def _ingest_file(self, file_path: Path) -> List[DocumentChunk]:
        """
        Ingest a single file.

        Args:
            file_path: Path to the document file.

        Returns:
            List of DocumentChunk objects from the file.
        """
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            text = self._read_pdf(file_path)
        elif suffix in (".txt", ".md"):
            text = self._read_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Split into chunks
        chunk_texts = self.splitter.split_text(text)

        # Create DocumentChunk objects with metadata
        chunks: List[DocumentChunk] = []
        for i, chunk_text in enumerate(chunk_texts):
            if chunk_text.strip():
                chunks.append(
                    DocumentChunk(
                        text=chunk_text.strip(),
                        metadata={
                            "source": file_path.name,
                            "source_path": str(file_path),
                            "chunk_index": str(i),
                            "total_chunks": str(len(chunk_texts)),
                        },
                    )
                )
        return chunks

    @staticmethod
    def _read_pdf(file_path: Path) -> str:
        """Read text from a PDF file using PyPDF2."""
        try:
            import PyPDF2

            text_parts: List[str] = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except ImportError:
            logger.warning("PyPDF2 not installed. Skipping PDF.")
            return ""

    @staticmethod
    def _read_text(file_path: Path) -> str:
        """Read text from a plain text or markdown file."""
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def ingest_text(self, text: str, source_name: str = "direct_input") -> List[DocumentChunk]:
        """
        Ingest raw text directly (useful for API uploads).

        Args:
            text: Raw text to ingest.
            source_name: Name label for the source.

        Returns:
            List of DocumentChunk objects.
        """
        chunk_texts = self.splitter.split_text(text)
        chunks: List[DocumentChunk] = []
        for i, chunk_text in enumerate(chunk_texts):
            if chunk_text.strip():
                chunks.append(
                    DocumentChunk(
                        text=chunk_text.strip(),
                        metadata={
                            "source": source_name,
                            "chunk_index": str(i),
                            "total_chunks": str(len(chunk_texts)),
                        },
                    )
                )
        return chunks
