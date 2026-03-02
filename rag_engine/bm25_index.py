"""
BM25 sparse retrieval index for the RAG pipeline.

Implements BM25Okapi over chunked document text for
keyword-based retrieval complementing dense vector search.
"""

import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)
config = Config()


class BM25Index:
    """
    BM25 sparse retrieval over document chunks.

    BM25Okapi captures exact keyword matches (e.g., regulation IDs
    like "RBI/2024-25/42") that dense embeddings may miss. Combined
    with dense retrieval, it forms a hybrid system with superior recall.

    Design decision: BM25 vs. Elasticsearch
    - BM25Okapi: In-process, no infra, fast for <100K docs.
    - Elasticsearch: Distributed, supports faceted search, but requires
      separate cluster. Required for >100K docs or complex queries.
    """

    def __init__(self) -> None:
        rag_config = config.get_section("rag_engine")
        self.index_path = Path(
            rag_config.get("bm25", {}).get("index_path", "artifacts/bm25_index.pkl")
        )
        self._bm25 = None
        self._texts: List[str] = []
        self._metadata: List[Dict[str, str]] = []
        self._tokenized_corpus: List[List[str]] = []

    def build_index(
        self, texts: List[str], metadata: List[Dict[str, str]]
    ) -> None:
        """
        Build BM25 index from document texts.

        Args:
            texts: List of chunk text strings.
            metadata: List of metadata dicts corresponding to each text.
        """
        from rank_bm25 import BM25Okapi

        self._texts = texts
        self._metadata = metadata
        self._tokenized_corpus = [self._tokenize(text) for text in texts]

        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 index built: {len(texts)} documents")

    def search(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[str, Dict[str, str], float]]:
        """
        Search the BM25 index.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of (text, metadata, score) tuples.
        """
        if self._bm25 is None:
            logger.warning("BM25 index not built or loaded")
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[::-1][:top_k]

        results: List[Tuple[str, Dict[str, str], float]] = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append((
                    self._texts[idx],
                    self._metadata[idx],
                    float(scores[idx]),
                ))

        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Simple tokenization for BM25.

        Lowercases, splits on non-alphanumeric, removes short tokens.
        """
        tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        return [t for t in tokens if len(t) > 1]

    def save(self) -> None:
        """Save the BM25 index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "texts": self._texts,
            "metadata": self._metadata,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"BM25 index saved to {self.index_path}")

    def load(self) -> None:
        """Load the BM25 index from disk."""
        from rank_bm25 import BM25Okapi

        if not self.index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {self.index_path}")

        with open(self.index_path, "rb") as f:
            data = pickle.load(f)

        self._texts = data["texts"]
        self._metadata = data["metadata"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info(f"BM25 index loaded: {len(self._texts)} documents")

    @property
    def size(self) -> int:
        """Return the number of documents in the index."""
        return len(self._texts)
