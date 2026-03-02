# 🛡️ RiskIntellect-AI

> **Made by Ayan**
>
> A production-grade Intelligence system combining **supervised ML fraud detection**, **RAG-based regulatory document search**, and **LLM-grounded reasoning** for financial risk assessment.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-orange.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

---

## 🔗 Live Links
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Ayan113/RiskIntellect-AI)

- **Live Demo**: [https://riskintellect-ai.onrender.com](https://riskintellect-ai.onrender.com/dashboard)
- **API Documentation**: [https://riskintellect-ai.onrender.com/docs](https://riskintellect-ai.onrender.com/docs)
- **GitHub Repository**: [https://github.com/Ayan113/RiskIntellect-AI](https://github.com/Ayan113/RiskIntellect-AI)

> [!TIP]
> **To Deploy**: Click the "Deploy to Render" button above. Render will use the included `render.yaml` to set up the service.

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Gateway                          │
│  /api/v1/fraud/predict  │  /api/v1/rag/query  │  /copilot/assess│
└────────┬────────────────┴─────────┬───────────┴────────┬───────┘
         │                          │                      │
    ┌────▼────┐              ┌──────▼──────┐        ┌─────▼──────┐
    │ ML Engine│              │  RAG Engine  │        │ LLM Layer  │
    │         │              │              │        │            │
    │ Scikit  │              │ FAISS + BM25 │───────►│ Prompt     │
    │ SHAP    │──────────────► Hybrid RRF   │        │ Builder    │
    │ MLflow  │              │ Reranker     │        │ OpenAI     │
    └────┬────┘              └──────┬───────┘        │ Guardrails │
         │                          │                └─────┬──────┘
         │                          │                      │
    ┌────▼──────────────────────────▼──────────────────────▼─────┐
    │                    Evaluation Framework                     │
    │  ML Metrics │ RAG Metrics │ Adversarial Tests │ Reports    │
    └────────────────────────────────────────────────────────────┘
```

### Data Flow — Full Pipeline
1. **Transaction Data** is analyzed by the **ML Engine (HistGradientBoosting)** to calculate a fraud probability.
2. **SHAP Explainer** identifies the top features driving the risk score.
3. **RAG Engine** retrieves relevant regulatory context from indexed compliance documents (RBI circulars, PMLA).
4. **LLM Layer** orchestrates the scores, explanations, and context to provide a final human-readable risk assessment.

---

## 🗂️ Repository Structure

```
RiskIntellect-AI/
├── config/             # Centralized YAML configuration
├── data/               # Raw & processed data + Regulatory documents
├── ml_engine/          # ML Training, inference & explainability (SHAP)
├── rag_engine/         # Retrieval-Augmented Generation (FAISS + BM25)
├── llm_layer/          # LLM Reasoning & Prompt Orchestration
├── evaluation/         # Automated evaluation for ML & RAG components
├── api/                # FastAPI application & REST endpoints
├── frontend/           # Premium Glassmorphism Dashboard (HTML/CSS/JS)
├── utils/              # Shared utilities (logging, security, config)
├── tests/              # Comprehensive test suite
├── artifacts/          # Serialized models and indices (Gitignored)
├── Dockerfile          # Production Docker configuration
└── requirements.txt    # Python dependencies
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Ayan113/RiskIntellect-AI.git
cd RiskIntellect-AI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file from the example:
```bash
cp .env.example .env
# Add your OPENAI_API_KEY
```

### 3. Bootstrap the System
Generate synthetic data, train the model, and build RAG indices:
```bash
PYTHONPATH=. python bootstrap.py
```

### 4. Run the API & Dashboard
```bash
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to `http://localhost:8000/dashboard` to access the interactive dashboard.

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | `GET` | System health check |
| `/api/v1/fraud/predict` | `POST` | Get fraud score & reasoning |
| `/api/v1/rag/search` | `POST` | Search regulatory documents |
| `/api/v1/copilot/assess` | `POST` | **Full Assessment** (ML + RAG + LLM) |

---

## 🏗️ Technical Decisions

- **ML Model**: `HistGradientBoostingClassifier` chosen for its speed, accuracy, and native support for missing values.
- **Explainability**: SHAP (Shapley Additive Explanations) for local model transparency.
- **RAG Strategy**: Hybrid Search (FAISS for semantics + BM25 for keywords) with Reciprocal Rank Fusion (RRF).
- **Frontend**: Vanilla JS/CSS for performance, featuring a premium Glassmorphism design system.

---

## 🔐 Security & Guardrails

- **Prompt Injection Defense**: Multi-layer semantic check for malicious instructions.
- **Hallucination Guardrails**: Cross-references LLM output with retrieved regulatory facts.
- **Input Sanitization**: Strict Pydantic validation for all transaction features.

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
