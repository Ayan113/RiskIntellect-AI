"""
FAISS vector store for dense retrieval.

Manages index creation, persistence, and similarity search
over document embeddings.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from rag_engine.ingestion import DocumentChunk
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)
config = Config()


class VectorStore:
    """
    FAISS-based vector store for document chunk retrieval.

    Uses IndexFlatIP (Inner Product) with L2-normalized vectors,
    which is equivalent to cosine similarity but faster.

    Design decision: FAISS in-process vs. managed vector DB
    - FAISS: Zero infra, fast, no network latency. Ideal for <1M docs.
    - Pinecone/Weaviate: Managed, scalable, but adds network round-trip
      and cost. Required when index exceeds single-machine RAM.
    
    For production with >1M documents, migrate to IndexIVFFlat for
    approximate search (10x faster, ~95% recall) or a managed DB.
    """

    def __init__(self) -> None:
        rag_config = config.get_section("rag_engine")
        self.index_path = Path(
            rag_config.get("vector_store", {}).get("index_path", "artifacts/faiss_index")
        )
        self.metadata_path = Path(
            rag_config.get("vector_store", {}).get("metadata_path", "artifacts/faiss_metadata.json")
        )
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, str]] = []
        self.texts: List[str] = []

    def build_index(
        self, chunks: List[DocumentChunk], embeddings: np.ndarray
    ) -> None:
        """
        Build the FAISS index from document chunks and their embeddings.

        Args:
            chunks: List of DocumentChunk objects.
            embeddings: numpy array of shape (n_chunks, dim).
        """
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vecs)
        self.index.add(embeddings.astype(np.float32))

        self.texts = [chunk.text for chunk in chunks]
        self.metadata = [chunk.metadata for chunk in chunks]

        logger.info(
            f"FAISS index built: {self.index.ntotal} vectors, {dim} dimensions"
        )

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[Tuple[str, Dict[str, str], float]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector (1D numpy array).
            top_k: Number of results to return.

        Returns:
            List of (text, metadata, score) tuples, sorted by relevance.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty or not loaded")
            return []

        # Reshape for FAISS
        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results: List[Tuple[str, Dict[str, str], float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for missing results
                results.append((
                    self.texts[idx],
                    self.metadata[idx],
                    float(score),
                ))

        return results

    def save(self) -> None:
        """Save the FAISS index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("No index to save. Call build_index() first.")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        metadata_store = {
            "texts": self.texts,
            "metadata": self.metadata,
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata_store, f)

        logger.info(f"FAISS index saved to {self.index_path}")

    def load(self) -> None:
        """Load the FAISS index and metadata from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")

        self.index = faiss.read_index(str(self.index_path))

        with open(self.metadata_path, "r") as f:
            metadata_store = json.load(f)

        self.texts = metadata_store["texts"]
        self.metadata = metadata_store["metadata"]

        logger.info(
            f"FAISS index loaded: {self.index.ntotal} vectors"
        )

    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        return self.index.ntotal if self.index else 0
