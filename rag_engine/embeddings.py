"""
Embedding generation for the RAG pipeline.

Wraps HuggingFace sentence-transformers for dense vector generation.
"""

from typing import List

import numpy as np

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)
config = Config()


class EmbeddingGenerator:
    """
    Dense embedding generator using HuggingFace sentence-transformers.

    Model: all-MiniLM-L6-v2
    - 384-dimensional embeddings
    - 5x faster than larger models
    - Good quality for passage retrieval
    - Max sequence length: 256 tokens (truncates longer inputs)

    Tradeoff: MiniLM vs. BGE-large vs. OpenAI embeddings
    - MiniLM: Free, fast, 384-dim. Good for prototyping and moderate-scale.
    - BGE-large: Higher quality, 1024-dim. Better retrieval, 3x slower.
    - OpenAI text-embedding-3-small: Best quality, costs $0.02/1M tokens.
    We default to MiniLM for zero-cost local execution.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        Initialize the embedding generator.

        Args:
            model_name: HuggingFace model name. Defaults to config.
        """
        self.model_name = model_name or config.get(
            "rag_engine.embeddings.model_name",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.device = config.get("rag_engine.embeddings.device", "cpu")
        self.batch_size = config.get("rag_engine.embeddings.batch_size", 64)
        self._model = None
        self._fallback_mode = False

    def _load_model(self) -> None:
        """Lazy-load the sentence transformer model with robustness."""
        if self._model is not None:
            return

        try:
            # Set environment variables to prevent common native library crashes on Mac
            import os
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name, device=self.device
            )
            logger.info(
                f"Embedding model loaded. Dimension: {self._model.get_sentence_embedding_dimension()}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}. Switching to fallback mode.")
            self._fallback_mode = True

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        """
        self._load_model()
        
        if self._fallback_mode:
            logger.warning("Using mock embeddings (random normalized vectors)")
            total_dims = 384
            rng = np.random.default_rng(seed=42)
            vecs = rng.standard_normal((len(texts), total_dims))
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return (vecs / (norms + 1e-10)).astype(np.float32)

        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}. Using fallback.")
            self._fallback_mode = True
            return self.embed_texts(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Args:
            query: Query string.

        Returns:
            1D numpy array of embedding values.
        """
        self._load_model()
        assert self._model is not None
        return self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        self._load_model()
        assert self._model is not None
        return self._model.get_sentence_embedding_dimension()
