#!/usr/bin/env python3
"""
Bootstrap script — generates synthetic data, trains the ML model,
and builds RAG indices so the full pipeline runs end-to-end.

Usage:
    python bootstrap.py
"""

import sys
from pathlib import Path

# Project root imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger("bootstrap")


def generate_synthetic_data():
    """Generate synthetic credit card fraud data matching the Kaggle schema."""
    logger.info("=" * 60)
    logger.info("Step 1/3: Generating synthetic credit card data")
    logger.info("=" * 60)

    np.random.seed(42)
    n_legit = 5000
    n_fraud = 50  # ~1% fraud rate for training purposes

    # Generate legitimate transactions
    legit_v = np.random.randn(n_legit, 28)  # V1-V28 (PCA features)
    legit_amount = np.abs(np.random.lognormal(3.5, 1.5, n_legit))
    legit_time = np.sort(np.random.uniform(0, 172800, n_legit))  # 48 hours

    # Generate fraudulent transactions (shifted distribution)
    fraud_v = np.random.randn(n_fraud, 28)
    # Shift some V-features to mimic fraud patterns
    fraud_v[:, 0] -= 3.0   # V1 lower
    fraud_v[:, 1] -= 2.5   # V2 lower
    fraud_v[:, 2] += 1.5   # V3 higher
    fraud_v[:, 3] += 2.0   # V4 higher
    fraud_v[:, 4] -= 2.0   # V5 lower
    fraud_v[:, 9] -= 4.0   # V10 lower
    fraud_v[:, 13] -= 6.0  # V14 lower — strong fraud signal
    fraud_v[:, 16] -= 4.0  # V17 lower

    fraud_amount = np.abs(np.random.lognormal(6.0, 2.0, n_fraud))  # Higher amounts
    fraud_time = np.random.uniform(0, 172800, n_fraud)

    # Combine
    v_cols = [f"V{i}" for i in range(1, 29)]
    legit_df = pd.DataFrame(legit_v, columns=v_cols)
    legit_df["Amount"] = legit_amount
    legit_df["Time"] = legit_time
    legit_df["Class"] = 0

    fraud_df = pd.DataFrame(fraud_v, columns=v_cols)
    fraud_df["Amount"] = fraud_amount
    fraud_df["Time"] = fraud_time
    fraud_df["Class"] = 1

    df = pd.concat([legit_df, fraud_df], ignore_index=True).sample(frac=1, random_state=42)
    df = df.reset_index(drop=True)

    # Save
    out_path = ROOT / "data" / "raw" / "creditcard.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    logger.info(f"Synthetic dataset saved: {out_path}")
    logger.info(f"  Total: {len(df)} | Legit: {n_legit} | Fraud: {n_fraud}")
    return out_path


def train_model():
    """Train the XGBoost fraud model and save artifacts."""
    logger.info("=" * 60)
    logger.info("Step 2/3: Training XGBoost fraud detection model")
    logger.info("=" * 60)

    from ml_engine.data_loader import DataLoader
    from ml_engine.feature_engineering import FeatureEngineer
    from ml_engine.trainer import FraudModelTrainer

    # Load data
    loader = DataLoader()
    loader.load()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.get_splits()

    # Feature engineering
    fe = FeatureEngineer()
    X_train_t = fe.fit_transform(X_train)
    X_val_t = fe.transform(X_val)
    X_test_t = fe.transform(X_test)

    # Train
    trainer = FraudModelTrainer()
    trainer.train(X_train_t, y_train, X_val_t, y_val)

    # Save artifacts
    trainer.save_model()
    fe.save()

    # Quick test score
    from sklearn.metrics import roc_auc_score
    test_proba = trainer.model.predict_proba(X_test_t)[:, 1]
    test_auc = roc_auc_score(y_test, test_proba)
    logger.info(f"Test ROC-AUC: {test_auc:.4f}")

    return trainer, fe


def build_rag_indices():
    """Ingest regulatory documents and build FAISS + BM25 indices."""
    logger.info("=" * 60)
    logger.info("Step 3/3: Building RAG indices from regulatory docs")
    logger.info("=" * 60)

    from rag_engine.ingestion import DocumentIngester
    from rag_engine.embeddings import EmbeddingGenerator
    from rag_engine.vector_store import VectorStore
    from rag_engine.bm25_index import BM25Index

    # Ingest documents
    ingester = DocumentIngester()
    chunks = ingester.ingest_all()  # Returns List[DocumentChunk]

    logger.info(f"Ingested {len(chunks)} chunks from regulatory documents")

    if not chunks:
        logger.error("No document chunks found! Check data/regulatory_docs/")
        return

    texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # Build FAISS index
    logger.info("Building FAISS vector index...")
    embedder = EmbeddingGenerator()
    embeddings = embedder.embed_texts(texts)

    vs = VectorStore()
    vs.build_index(chunks, embeddings)  # Takes (DocumentChunk list, embeddings)
    vs.save()
    logger.info(f"FAISS index built and saved")

    # Build BM25 index
    logger.info("Building BM25 sparse index...")
    bm25 = BM25Index()
    bm25.build_index(texts, metadatas)  # Takes (texts, metadata)
    bm25.save()
    logger.info(f"BM25 index built and saved")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🛡️  RiskIntellect-AI — Bootstrap")
    print("=" * 60 + "\n")

    try:
        # Step 1: Generate data
        generate_synthetic_data()

        # Step 2: Train ML model
        train_model()

        # Step 3: Build RAG indices
        build_rag_indices()

        print("\n" + "=" * 60)
        print("✅ Bootstrap complete! All components are ready.")
        print("=" * 60)
        print("\nStart the server with:")
        print("  PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
        print("\nDashboard: http://localhost:8000/dashboard")

    except Exception as e:
        logger.error(f"Bootstrap failed: {e}", exc_info=True)
        sys.exit(1)
