# AI-Powered Real Estate Property Valuation Assistant

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B.svg)](https://streamlit.io/)
[![DeepEval](https://img.shields.io/badge/DeepEval-4.2.8-purple.svg)](https://deepeval.com/)
[![ML Test R2](https://img.shields.io/badge/Model%20R%C2%B2-0.9031-success.svg)](https://scikit-learn.org/)

An autonomous multi-agent real estate valuation intelligence platform built on the **Ames Housing Dataset** (OpenML 43926). Combines true hybrid retrieval (dense semantic vectors + TF-IDF keyword search over the full filtered candidate set), supervised Gradient Boosting regression, physical risk quantification, and an independent LLM-as-Judge appraisal auditor.

---

## 1. Project Overview & Business Objective
### Problem Statement
Real estate professionals regularly handle recurring workflows involving property appraisal, comparable market analysis (CMA), condition adjustments, and risk underwriting. Manual searching across static property records containing thousands of past transactions is slow, subjective, and prone to uncalibrated appraisal risk.

### Business Objective
To deliver an end-to-end, production-ready microservice and user-friendly interface that allows real estate professionals to enter natural language queries or structured property attributes to:
1. Retrieve prioritized comparable historical properties using true hybrid keyword and semantic search.
2. Accurately predict fair market property values using supervised machine learning ($R^2 = 0.9031$).
3. Quantify multi-dimensional physical age and market volatility risks.
4. Validate valuation consistency via an independent LLM-as-Judge auditor.
5. Capture appraiser feedback in a continuous learning loop to dynamically eliminate pricing bias.

---

## 2. Solution Summary
The system integrates an autonomous 8-step agentic workflow:
- **Input Guardrails**: Validates 14 physical parameter fields including `total_rooms` (TotRmsAbvGrd) and `garage_yr_blt` (GarageYrBlt) against physical boundaries ($100 \le \text{SqFt} \le 15,000$, $1 \le \text{Quality} \le 10$) and detects prompt injection attacks. Metadata filters that match zero properties return an empty list `[]` — there is no silent fallback to all properties.
- **True Hybrid Search & Prioritization Engine**: TF-IDF keyword scoring runs over the **full metadata-filtered candidate set** (not only the semantic top-k), then semantic and keyword scores are fused over their union. Applies a multi-factor physical comparability formula (Location, SqFt, Quality, Age, Rooms).
- **Classification Agent**: Automatically segments properties into Market Tiers, Architectural Typologies, Condition Grades, and Investment Profiles.
- **Predictive ML Service**: Trained Gradient Boosting regressor with 90% confidence prediction intervals. Accepts `total_rooms` and `garage_yr_blt` as direct inputs.
- **Basic Valuation Agent**: Calculates comp-adjusted median, mean, and size-weighted fair market values. Uses a live LLM narrative when `OPENAI_API_KEY` is set; falls back to rule-based rationale otherwise.
- **Property Value Risk Agent**: Computes a 0–100 risk score across physical depreciation, condition discrepancies, neighborhood price volatility, and layout liquidity.
- **Feature & Location Impact Service**: Analyzes neighborhood price premiums vs. city baseline and powers what-if renovation simulations.
- **LLM-as-Judge Agent**: Audits the valuation against market comps and ML predictions to issue an official audit verdict (`APPROVED`, `FLAGGED_OVERVALUED`, `FLAGGED_UNDERVALUED`, `NEEDS_REVIEW`). Uses live LLM when `OPENAI_API_KEY` is set; falls back to deterministic multi-criterion rules otherwise.
- **Continuous Feedback Loop**: Captures appraiser ratings and transaction outcomes. Error and bias are always measured against the **uncalibrated** model prediction so the calibration factor does not cancel itself in successive rounds.
- **DeepEval Quality Benchmark**: Measures Faithfulness, Answer Relevancy, Contextual Precision, and Contextual Recall using real DeepEval metrics. In **offline mode** (no `OPENAI_API_KEY`), these four LLM-backed metrics are skipped (`SKIPPED_NO_LLM_KEY`) and `benchmark_status` is `NOT_EVALUATED`. Rule-based valuation accuracy is computed in all modes. Live scores are only reported when `OPENAI_API_KEY` is configured.

---

## 3. Technology Stack

- **Programming Language**: Python 3.13
- **LLM Used**: Dual-Mode Architecture:
  - Live Provider: OpenAI `gpt-4o-mini` (configurable via `OPENAI_API_KEY`)
  - Offline Fallback: Deterministic Real Estate Valuation Reasoning Engine (guaranteed offline operation for all endpoints; DeepEval LLM-backed metrics require a live key)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, cached locally)
- **Vector Database**: In-Memory Normalized Dense Cosine Index with cached `.npy` vectors
- **Lexical Keyword Search**: Scikit-learn TF-IDF Vectorizer (keyword scoring over full filtered candidate set)
- **Supervised ML Models**: Scikit-learn Gradient Boosting Regressor (Best model, $R^2 = 0.9031$, $\text{MAPE} = 8.46\%$) & Random Forest Regressor
- **Backend API Framework**: FastAPI + Uvicorn
- **Frontend Framework**: Streamlit
- **Evaluation Framework**: DeepEval (v4.2.8, pinned in `requirements.txt`)
- **Documentation & Presentation**: ReportLab (PDF generation) & python-pptx (PowerPoint deck)

---

## 4. Repository Structure

```
.
├── architecture/
│   ├── Architecture_Diagram.pdf       # High-level architecture diagram (PDF)
│   └── Architecture_Diagram.png       # Architecture diagram (PNG/JPEG)
├── design/
│   ├── Design_Document.pdf            # Formal design document (PDF)
│   └── Design_Document.md             # System design specifications (Markdown)
├── docs/
│   ├── API_Documentation.pdf          # REST API specifications (PDF)
│   ├── API_Documentation.md           # API reference documentation (Markdown)
│   ├── Evaluation_Report.pdf          # DeepEval quality evaluation report (PDF)
│   ├── Evaluation_Report.md           # Evaluation report (Markdown)
│   ├── Dataset_Details.pdf            # Ames housing dataset details (PDF)
│   └── Dataset_Details.md             # Dataset documentation (Markdown)
├── presentation/
│   ├── Project_Presentation.pptx      # 10-minute panel presentation deck (PPTX)
│   └── Project_Presentation.pdf       # Presentation deck (PDF)
├── data/
│   ├── sample_data/
│   │   ├── ames_housing.csv           # Raw Ames Housing dataset (OpenML 43926)
│   │   ├── ames_housing_cleaned.csv   # Standardized dataset (15 key fields)
│   │   ├── property_embeddings.npy    # Precomputed dense vector index
│   │   └── feedback.json              # Continuous improvement feedback log
│   └── models/
│       └── property_price_model.joblib# Trained Gradient Boosting model artifact
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py                  # 14 REST API endpoints
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── classification_agent.py    # Property classification agent
│   │   ├── valuation_agent.py         # Basic valuation recommendation agent
│   │   ├── risk_agent.py              # Property value risk prediction agent
│   │   ├── judge_agent.py             # LLM-as-Judge valuation audit agent
│   │   └── orchestrator.py            # Multi-agent collaboration orchestrator
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── data_pipeline.py           # Preprocessing & searchable text generator
│   │   ├── vector_store.py            # Dense embedding vector store & filters
│   │   ├── hybrid_retriever.py        # True hybrid TF-IDF + Dense search & ranking
│   │   └── guardrails.py              # 14-field bounds validation & injection defense
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ml_prediction.py           # Supervised ML price regression service
│   │   ├── feature_impact.py          # Location premiums & what-if simulator
│   │   ├── evaluation_service.py      # DeepEval benchmark runner (honest offline/live mode)
│   │   └── feedback_service.py        # Appraiser feedback & bias calibration
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py                  # Environment & project configuration
│   │   ├── llm_client.py              # Unified LLM client with offline fallback
│   │   └── logger.py                  # Structured console logger
│   ├── main.py                        # FastAPI entrypoint
│   └── frontend.py                    # Streamlit web application
├── tests/
│   ├── test_guardrails.py             # Unit tests for guardrails (incl. new fields)
│   ├── test_retrieval.py              # Unit tests for hybrid search & ranking
│   ├── test_ml_models.py              # Unit tests for ML price prediction
│   ├── test_agents.py                 # Unit tests for multi-agent workflows
│   ├── test_api.py                    # Integration tests for FastAPI endpoints
│   └── test_evaluation.py             # Tests for DeepEval benchmark runner (offline+live)
├── requirements.txt                   # Complete project dependencies (deepeval==4.2.8)
├── README.md                          # Project documentation & instructions
└── .gitignore                         # Git exclusion rules (.deepeval/, Icon, __pycache__)
```

---

## 5. Installation Instructions

### Step 1: Environment Setup
Ensure Python 3.10+ (Python 3.13 recommended) is installed:
```bash
python3 --version
```

### Step 2: Clone & Navigate to Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: API Key Configuration (Optional)
The system operates in offline mode out-of-the-box using local models (`all-MiniLM-L6-v2`, trained Gradient Boosting regressor, and deterministic real estate reasoning fallback).

> **Note**: The four LLM-backed DeepEval metrics (Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall) require a live OpenAI API key. Without it, the evaluation endpoint returns `evaluation_mode = "offline_no_llm"` and `benchmark_status = "NOT_EVALUATED"` with null scores for those metrics. Valuation accuracy is always computed.

To connect an external live LLM (e.g. OpenAI GPT-4o-mini), set the environment variable:
```bash
export OPENAI_API_KEY="your-openai-api-key"
# Optional custom base URL:
# export OPENAI_BASE_URL="https://api.openai.com/v1"
```

---

## 6. Local Deployment Instructions

### Running the Backend API (FastAPI)
Launch the FastAPI microservice using Uvicorn:
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Base URL: `http://localhost:8000`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/health`

### Running the Front-End UI (Streamlit)
Launch the Streamlit web dashboard in a separate terminal:
```bash
streamlit run src/frontend.py
```
- Browser Access: `http://localhost:8501`

### Running the Test Suite
Execute the 32 automated unit and integration tests:
```bash
PYTHONPATH=. pytest tests/
```

---

## 7. Sample Queries & Expected Outputs

### Query 1: Suburban Single-Family Valuation (North Ames)
- **Natural Language Query**:
  > *"A property in North Ames has 1,800 sq ft, 3 bedrooms, 2 bathrooms, built in 1995 with overall quality 7. What is the expected property value and similar properties?"*
- **Sample Request Payload**:
  ```json
  POST /api/v1/valuation/collaborate
  {
    "property_features": {
      "neighborhood": "North Ames",
      "gr_liv_area": 1800,
      "bedrooms": 3,
      "bathrooms": 2,
      "year_built": 1995,
      "overall_qual": 7,
      "overall_cond": 6,
      "garage_cars": 2,
      "total_rooms": 7,
      "garage_yr_blt": 1995
    },
    "query": "A property in North Ames has 1,800 sq ft, 3 bedrooms, 2 bathrooms, built in 1995 with overall quality 7",
    "top_k_comps": 5
  }
  ```
- **Expected Output**:
  - **Reconciled Fair Market Value**: varies by calibration factor applied
  - **ML Statistical Point Estimate**: ~ \$200,000–\$215,000
  - **Market Tier**: Upper-Middle Market (Suburban Single-Family Residence)
  - **Risk Profile**: Low Risk (< 35 / 100)
  - **Judge Audit Verdict**: `APPROVED` / `NEEDS_REVIEW`
  - **Evaluation Mode**: `offline_no_llm` if no API key is set; `live_llm_judge` otherwise.

---

### Query 2: Metadata Filter That Matches Nothing
When filters match zero properties the API returns an empty comparable list rather than falling back to all properties:
```json
POST /api/v1/search
{
  "query": "family home",
  "top_k": 5,
  "filters": {"neighborhood": "NonExistentXYZ", "min_bedrooms": 15}
}
```
**Response**: `{"query": "...", "count": 0, "comparables": []}`

---

## 8. Evaluation Mode

The system supports two evaluation modes controlled by `OPENAI_API_KEY`:

| Mode | `evaluation_mode` | DeepEval Metrics | `benchmark_status` |
|---|---|---|---|
| No API key | `offline_no_llm` | `SKIPPED_NO_LLM_KEY` (null scores) | `NOT_EVALUATED` |
| API key set | `live_llm_judge` | Real scores (0.0–1.0) | `PASSED` / `NEEDS_OPTIMIZATION` |

- Rule-based valuation accuracy is computed in **both** modes.
- The DeepEval version is read dynamically from `deepeval.__version__` (pinned to `4.2.8` in `requirements.txt`).

---

## 9. Suggestions (NOT implemented)
*In strict accordance with the project scope guidelines, the following potential enhancements were noted but intentionally NOT implemented to maintain exact specification adherence:*
- Multi-region expansion beyond Ames, Iowa (e.g., California or Florida housing data).
- Automated aerial satellite imagery ingestion via computer vision.
- Mortgage payment and amortization schedule calculator.
- Distributed vector databases (e.g., Pinecone or Milvus cloud clusters).
- Real-time MLS webhook streaming listeners.

---

## 10. Compliance Checklist

| Item # | Specification Requirement | Status | Satisfying File(s) / Endpoints |
|---|---|---|---|
| **1.1** | Basic RAG for property similarity search | **Done** | `src/rag/vector_store.py`, `POST /api/v1/search` |
| **1.2** | True hybrid search (full-candidate-set TF-IDF + semantic fusion) | **Done** | `src/rag/hybrid_retriever.py`, `POST /api/v1/search` |
| **1.3** | Property classification agent | **Done** | `src/agents/classification_agent.py`, `POST /api/v1/classify` |
| **1.4** | Metadata filtering; empty result when filters match nothing | **Done** | `src/rag/vector_store.py`, `src/rag/hybrid_retriever.py` |
| **1.5** | Basic property valuation recommendation | **Done** | `src/agents/valuation_agent.py`, `POST /api/v1/valuation/basic` |
| **1.6** | Input validation guardrails (14 fields, injection, off-topic) | **Done** | `src/rag/guardrails.py`, `POST /api/v1/guardrails/validate` |
| **1.7** | Comparable-property prioritization | **Done** | `src/rag/hybrid_retriever.py` (`_calculate_comparability`) |
| **1.8** | Expose core functionality through API endpoints | **Done** | `src/api/routes.py`, `src/main.py` (14 REST endpoints) |
| **2.1** | DeepEval honest evaluation (offline/live mode) | **Done** | `src/services/evaluation_service.py`, `GET /api/v1/evaluation/run` |
| **2.2** | ML price prediction (TotRmsAbvGrd & GarageYrBlt passed through) | **Done** | `src/services/ml_prediction.py`, `POST /api/v1/valuation/predict-price` |
| **2.3** | LLM-as-judge with live LLM + fallback | **Done** | `src/agents/judge_agent.py`, `POST /api/v1/valuation/judge-validate` |
| **2.4** | Multi-agent property valuation collaboration | **Done** | `src/agents/orchestrator.py`, `POST /api/v1/valuation/collaborate` |
| **2.5** | Property value risk prediction | **Done** | `src/agents/risk_agent.py`, `POST /api/v1/valuation/predict-risk` |
| **2.6** | Property feature and location impact analysis | **Done** | `src/services/feature_impact.py`, `POST /api/v1/valuation/impact-analysis` |
| **2.7** | Feedback loop with uncalibrated-bias calibration factor | **Done** | `src/services/feedback_service.py`, `POST /api/v1/feedback/submit` |
| **2.8** | Build a simple front-end interface | **Done** | `src/frontend.py` (Streamlit 6-tab dashboard) |
| **4.1** | Architecture Diagram (PDF/JPEG) | **Done** | `architecture/Architecture_Diagram.pdf`, `.png` |
| **4.2** | Design Document (PDF/MD) | **Done** | `design/Design_Document.pdf`, `design/Design_Document.md` |
| **4.3** | Full Executable Code – Microservice & README | **Done** | `src/`, `README.md`, `requirements.txt` |
| **4.4** | Panel Presentation – 10 Minutes (PPTX/PDF) | **Done** | `presentation/Project_Presentation.pptx`, `.pdf` |
| **5.0** | Dataset Documentation & 15 Key Fields | **Done** | `docs/Dataset_Details.pdf`, `data/sample_data/ames_housing.csv` |
| **Doc** | API Documentation (PDF/MD) | **Done** | `docs/API_Documentation.pdf`, `docs/API_Documentation.md` |
| **Doc** | Evaluation Report (DeepEval PDF/MD) | **Done** | `docs/Evaluation_Report.pdf`, `docs/Evaluation_Report.md` |
| **Test**| Test Suite (pytest) | **Done** | `tests/` (32 passing unit & integration tests) |
# KunjalAhuja_RealEstate
