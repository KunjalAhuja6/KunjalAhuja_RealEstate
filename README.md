# AI-Powered Real Estate Property Valuation Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3.1-61DAFB.svg)](https://react.dev/)
[![DeepEval](https://img.shields.io/badge/DeepEval-4.2.8-purple.svg)](https://confident-ai.com/)
[![ML Model R2](https://img.shields.io/badge/Model%20R%C2%B2-0.9198-success.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/pytest-37%20passed-brightgreen.svg)](https://pytest.org/)

An autonomous multi-agent real estate valuation platform combining supervised machine learning (`GradientBoostingRegressor`), hybrid Retrieval-Augmented Generation (dense cosine embeddings + lexical TF-IDF keyword search over candidate sets), physical risk quantification, appraiser feedback calibration, and an independent LLM-as-Judge appraisal auditor.

**Presenter / Author**: Kunjal Ahuja  
**Domain**: Real Estate & PropTech Valuation Intelligence  
**Institution**: Prodapt Capstone Project  

---

## 1. Project Overview & Business Objective

### Problem Statement
Real estate appraisal, comparative market analysis (CMA), and property underwriting are traditionally manual, labor-intensive processes. Appraisers must manually filter hundreds of historical MLS records, make subjective condition and age adjustments, and reconcile conflicting figures without quantitative risk bounds. This results in slow turnaround times (often 3–7 business days), inconsistent valuations, and vulnerability to uncalibrated systematic pricing bias.

### Business Objective
The objective is to deliver an end-to-end, production-grade valuation intelligence platform that empowers real estate professionals to enter structured property parameters or natural language queries and immediately obtain:
1. **Prioritized Comparable Sales**: Grounded historical comparable sales retrieved via true hybrid retrieval (TF-IDF keyword + dense semantic vector search) with dynamic neighborhood widening when local comps are sparse.
2. **High-Precision Supervised Valuation**: Machine-learned point estimates using a Gradient Boosting Regressor ($R^2 = 0.9198$, $\text{RMSE} = \$25,361.84$, $\text{MAPE} = 8.87\%$ evaluated on strictly real test records) with 90% empirical prediction intervals.
3. **Multi-Dimensional Risk Assessment**: A composite 0–100 physical and market risk index quantifying physical age depreciation, quality-condition gaps, neighborhood price volatility, and layout liquidity.
4. **Independent LLM-as-Judge Audit**: An appraisal audit checking statistical alignment between comp evidence and ML models, producing an audit verdict (`APPROVED`, `FLAGGED_OVERVALUED`, `FLAGGED_UNDERVALUED`, `NEEDS_REVIEW`).
5. **Continuous Calibration Loop**: An automated feedback loop that captures real closing prices and adjusts future reconciled valuations via an uncalibrated bias factor.

---

## 2. Solution Summary

The platform operates an autonomous multi-stage workflow:

```
[Property Input / Query]
           │
           ▼
[Input Guardrails & Security]  ──► (Validates 14 fields: living area, quality, rooms; prevents injection)
           │
           ▼
[Classification Agent]          ──► (Market Tier, Typology, Condition Grade, Target Demographic)
           │
           ▼
[Hybrid RAG Engine]             ──► (Dense all-MiniLM-L6-v2 + TF-IDF lexical search, alpha=0.55)
           │                        (Physical comparability scoring; dynamic neighborhood widening)
           ▼
[Supervised ML Regressor]       ──► (GradientBoostingRegressor with 90% prediction interval)
           │
           ▼
[Valuation Reconciliation]      ──► (Dynamic comp_fit weighting: comp median vs. ML point estimate)
           │
           ▼
[Property Value Risk Agent]     ──► (0-100 Risk Score: physical, condition, volatility, liquidity)
           │
           ▼
[LLM-as-Judge Auditor]          ──► (Multi-criterion audit: checks comp-ML divergence, issues verdict)
           │
           ▼
[Feedback & Continuous Learning]──► (Logs appraiser feedback; tracks uncalibrated pricing bias)
```

### Key Workflow Capabilities:
- **Input Guardrails**: Validates 14 property fields (`gr_liv_area`, `overall_qual`, `overall_cond`, `year_built`, `total_rooms`, `garage_yr_blt`, etc.) against physical bounds ($100 \le \text{SqFt} \le 15,000$, $1 \le \text{Quality} \le 10$). Metadata filters matching zero records return an empty list `[]` defensively rather than returning unconstrained properties.
- **True Hybrid Search & Prioritization**: Dense embeddings (`all-MiniLM-L6-v2`, 384 dimensions) combined with TF-IDF keyword matching across the full candidate set ($\alpha = 0.55$). Candidate prioritization applies a 50/50 blend of hybrid score and physical comparability (Location 30%, Living Area 25%, Quality 20%, Age 15%, Room Count 10%).
- **Comp Neighborhood Widening**: When $< 3$ comparable properties match in the subject neighborhood within $\pm 25\%$ living area and $\pm 1$ quality grade, the engine automatically widens search city-wide to prevent stale or unrepresentative comp sets.
- **Dynamic Valuation Reconciliation**: Reconciled fair market value dynamically weights comp median and ML prediction based on `comp_fit` (the proportion of high-quality matching comps), rather than an arbitrary static split:
  $$\text{comp\_weight} = \text{clip}(0.5 \times \text{comp\_fit} + 0.1, 0.1, 0.5)$$
  $$\text{ml\_weight} = 1.0 - \text{comp\_weight}$$
- **Continuous Feedback Calibration**: Tracks user-submitted closing prices and star ratings. Bias is measured against uncalibrated estimates ($\text{recommended} / \text{applied\_factor}$) to prevent compounding calibration drift.
- **DeepEval Quality Benchmark**: Evaluates retrieval and generation quality across 5 comprehensive benchmark scenarios measuring Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall, and Valuation Accuracy.

---

## 3. Technology Stack

- **Author / Presenter**: Kunjal Ahuja
- **Programming Language**: Python 3.10+ (tested on Python 3.13)
- **Backend API Framework**: FastAPI 0.115+, Uvicorn (16 REST endpoints)
- **Frontend Framework**: React 18.3.1 + Vite 5.4.11 (7 accessible tabs: Valuation, Comparable sales, ML prediction, What-if, Risk and audit, Feedback, Evaluation)
- **Supervised ML Regressor**: Scikit-learn `GradientBoostingRegressor` ($R^2 = 0.9198$, $\text{RMSE} = \$25,361.84$, $\text{MAE} = \$16,161.76$, $\text{MAPE} = 8.87\%$)
- **Dense Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, local NumPy cosine dot-product index)
- **Lexical Keyword Engine**: Scikit-learn `TfidfVectorizer` (unigrams + bigrams over full candidate set)
- **LLM Engine**: Dual-Mode Architecture:
  - Live Provider: OpenAI-compatible API (`gpt-4o-mini`, configured via `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL_NAME`)
  - Offline Fallback: Deterministic domain-specific heuristic valuation reasoning and rule-based judge auditing
- **Evaluation Framework**: DeepEval v4.2.8 (`LLMTestCase`, `FaithfulnessMetric`, `AnswerRelevancyMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric`)
- **Documentation & Presentation**: ReportLab (PDF publication) & `python-pptx` (PowerPoint generation)

---

## 4. Repository Structure

```
.
├── README.md                          # Master project documentation
├── requirements.txt                   # Pinned Python dependencies
├── .env.example                       # Environment configuration template
├── architecture/
│   ├── Architecture_Diagram.png       # Complete system architecture diagram
│   └── Architecture_Diagram.pdf       # Printable vector architecture diagram
├── design/
│   ├── Design_Document.md             # Formal architectural design document
│   └── Design_Document.pdf            # PDF version of design document
├── docs/
│   ├── API_Documentation.md           # Complete REST API reference (16 endpoints)
│   ├── API_Documentation.pdf          # PDF version of API documentation
│   ├── Dataset_Details.md             # Ames housing & synthetic data specifications
│   ├── Dataset_Details.pdf            # PDF version of dataset documentation
│   ├── Evaluation_Report.md           # DeepEval benchmark results & analysis
│   ├── Evaluation_Report.pdf          # PDF version of evaluation report
│   ├── eval_results.json              # Measured DeepEval benchmark run output
│   ├── sample_test_cases.md           # Curated test cases and benchmark scenarios
│   └── synthetic_data_report.json     # Synthetic generation validation metrics
├── presentation/
│   ├── Project_Presentation.pptx      # Capstone panel presentation deck
│   └── Project_Presentation.pdf       # PDF presentation export
├── data/
│   ├── sample_data/
│   │   ├── ames_housing.csv           # Raw Ames Housing dataset (2,930 real records)
│   │   ├── ames_housing_cleaned.csv   # Standardized 10,000-record dataset
│   │   ├── property_embeddings.npy    # Precomputed dense embeddings (10000x384)
│   │   └── feedback.json              # Feedback repository for dynamic calibration
│   └── models/
│       └── property_price_model.joblib# Serialized trained model & pipeline metadata
├── frontend/                          # Production React 18 frontend
│   ├── README.md                      # Frontend architecture & guide
│   ├── package.json                   # Dependencies (React 18.3.1, Vite 5.4.11)
│   ├── vite.config.js                 # Dev server proxy (/api -> http://localhost:8000)
│   ├── src/
│   │   ├── main.jsx                   # React application mount
│   │   ├── App.jsx                    # Root state container & tab coordination
│   │   ├── api.js                     # Defensive REST client
│   │   ├── theme.js                   # Light/dark theme management
│   │   ├── styles.css                 # Clean CSS with dark mode variables
│   │   ├── components/                # Modular reusable components
│   │   │   ├── Header.jsx             # Top bar, status indicator, theme toggle
│   │   │   ├── SidebarForm.jsx        # Shared property input form
│   │   │   ├── RangeBar.jsx           # Clean valuation track & marker legend
│   │   │   ├── ScoreBars.jsx          # Visual risk factor indicators
│   │   │   ├── EmptyState.jsx         # Empty state component
│   │   │   ├── ErrorState.jsx         # Error alert component
│   │   │   ├── LoadingState.jsx       # Working indicator component
│   │   │   └── OutOfDateNotice.jsx    # Form modification stale warning
│   │   └── tabs/                      # 7 accessible workflow tabs
│   │       ├── ValuationTab.jsx       # Reconciled value, range, plain-English summary
│   │       ├── ComparableSalesTab.jsx # Real comparable sales table & similarity scores
│   │       ├── MLPredictionTab.jsx    # Model estimate, 90% interval, top features
│   │       ├── WhatIfTab.jsx          # Interactive renovation simulator
│   │       ├── RiskAndAuditTab.jsx    # Risk breakdown & LLM judge audit verdict
│   │       ├── FeedbackTab.jsx        # Submit feedback & view calibration metrics
│   │       └── EvaluationTab.jsx      # Run/view DeepEval quality benchmarks
├── src/                               # Backend application microservice
│   ├── main.py                        # FastAPI application entrypoint
│   ├── frontend.py                    # Legacy Streamlit prototype (not used)
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py                  # 16 REST API endpoint controllers
│   ├── agents/
│   │   ├── classification_agent.py    # Typology, tier, and demographic classification
│   │   ├── valuation_agent.py         # Comparable valuation calculations & narrative
│   │   ├── risk_agent.py              # Multi-factor property risk assessment
│   │   ├── judge_agent.py             # LLM-as-judge appraisal auditor
│   │   └── orchestrator.py            # Multi-agent collaboration orchestrator
│   ├── rag/
│   │   ├── data_pipeline.py           # Text serialization & preprocessing
│   │   ├── vector_store.py            # Cosine similarity index over dense vectors
│   │   ├── hybrid_retriever.py        # True hybrid TF-IDF + dense search & widening
│   │   └── guardrails.py              # Input parameter validation & security
│   ├── services/
│   │   ├── ml_prediction.py           # Gradient Boosting Regressor training & inference
│   │   ├── feature_impact.py          # Feature importance & renovation simulator
│   │   ├── feedback_service.py        # Appraiser feedback & bias calibration factor
│   │   └── evaluation_service.py      # DeepEval benchmark runner & history persistence
│   └── utils/
│       ├── config.py                  # Environment variable configuration
│       ├── llm_client.py              # LLM wrapper with deterministic fallback
│       ├── logger.py                  # Structured console logging
│       └── augment_dataset.py         # Synthetic augmentation utilities
└── tests/                             # Automated test suite (37 tests)
    ├── test_guardrails.py             # Guardrail boundaries & injection filtering
    ├── test_retrieval.py              # Hybrid search, filtering, and empty-set behavior
    ├── test_ml_models.py              # Model inference, metrics, and prediction intervals
    ├── test_agents.py                 # Multi-agent collaboration & orchestrator
    ├── test_api.py                    # REST API integration tests (including history)
    └── test_evaluation.py             # DeepEval benchmark runner tests
```

---

## 5. Dataset & Synthetic Augmentation

The project dataset (`data/sample_data/ames_housing_cleaned.csv`) contains **10,000 records**:
- **Real Historical Transactions**: **2,930 records** sourced from the Ames Housing Dataset (Dean De Cock, 2011; OpenML 43926).
- **Synthetic Records**: **7,070 records** generated via a Gaussian Copula fitted per neighborhood on real marginal distributions. A novelty filter enforces minimum Euclidean distance to real nearest neighbors at the 25th percentile, preventing duplicates while expanding coverage.
- **Neighborhoods**: 28 distinct Ames neighborhoods.
- **Year Built Range**: 1872 to 2010.
- **15 Standardized Key Fields**:
  `SalePrice`, `OverallQual`, `OverallCond`, `GrLivArea`, `TotalBsmtSF`, `GarageCars`, `GarageArea`, `YearBuilt`, `YearRemodAdd`, `FullBath`, `BedroomAbvGr`, `TotRmsAbvGrd`, `Neighborhood`, `LotArea`, `GarageYrBlt`.

### Honest Train / Test Split & Comparable Protection:
1. **Real-Only Test Evaluation**: The test set ($n = 586$) consists strictly of 20% of the **real** historical sales (`random_state=42`). Synthetic records are **never** present in the test set.
2. **Training Set**: 2,344 real records + 7,070 synthetic records ($n = 9,414$).
3. **Comparable Sales Isolation**: All comparable property searches exclude synthetic records by default (`include_synthetic=False`). Only genuine historical transactions are presented as CMA evidence to clients.

### Model Performance Metrics (Evaluated on Real Test Set $n = 586$):
| Model | Training Data | Test Set | $R^2$ Score | RMSE | MAE | MAPE |
|---|---|---|---|---|---|---|
| **Active Gradient Boosting** | Real + Synthetic ($n=9,414$) | Real Only ($n=586$) | **0.9198** | **\$25,361.84** | **\$16,161.76** | **8.87%** |
| Baseline Gradient Boosting | Real Only ($n=2,344$) | Real Only ($n=586$) | 0.9031 | \$27,870.39 | \$15,889.09 | 8.46% |

---

## 6. Installation Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Node.js 18+ (tested on Node v23.11.0) and npm

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd "Prodapt Capstone"
```

### Step 2: Set Up Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### Step 4: Environment Configuration (Optional for Live LLM)
Copy the template configuration file:
```bash
cp .env.example .env
```
Edit `.env` to configure an OpenAI-compatible API key. If left unset, the system runs seamlessly in offline fallback mode.

---

## 7. Local Deployment Instructions

### Starting the Backend Microservice (FastAPI)
Activate the virtual environment and start Uvicorn:
```bash
source venv/bin/activate
uvicorn src.main:app --reload
```
- **API Base URL**: `http://localhost:8000`
- **Interactive Swagger Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

### Starting the Frontend UI (React 18 + Vite)
In a separate terminal, install dependencies and start the Vite development server:
```bash
cd frontend
npm install
npm run dev
```
- **User Interface**: `http://localhost:5173`
- The dev server automatically proxies `/api` requests to `http://localhost:8000`.

*(Note: `src/frontend.py` is a legacy Streamlit prototype and is not used in production).*

### Running the Automated Test Suite
Run the 37 unit and integration tests:
```bash
PYTHONPATH=. pytest -q
```
Measured output: `37 passed, 56 warnings in 88.90s`.

---

## 8. Real Sample Queries with Measured Outputs

All queries below were executed against the live microservice with a neutral feedback state (`feedback.json = []`):

### Query 1: Suburban Single-Family Home (North Ames)
- **Subject**: 1,600 sq ft, 3 Bed, 2 Bath, Year Built 1970, Overall Quality 6/10, Condition 5/10.
- **Request**: `POST /api/v1/valuation/collaborate`
- **Measured Response**:
  - **Reconciled Fair Market Value**: **\$164,208.31**
  - **90% Confidence Interval**: **\$128,569.34 – \$199,847.28**
  - **Market Tier**: Mid-Market Standard
  - **Risk Score**: Moderate Risk
  - **Audit Verdict**: `APPROVED`
  - **Comps Retrieved**: `PROP-0604` (\$177,625), `PROP-0674` (\$157,500), `PROP-0635` (\$139,000)

### Query 2: Luxury Executive Estate (Stone Brook)
- **Subject**: 2,800 sq ft, 4 Bed, 3 Bath, Year Built 2005, Overall Quality 9/10, Condition 5/10.
- **Request**: `POST /api/v1/valuation/collaborate`
- **Measured Response**:
  - **Reconciled Fair Market Value**: **\$400,710.58**
  - **90% Confidence Interval**: **\$330,697.48 – \$470,723.68**
  - **Market Tier**: Luxury / High-End
  - **Risk Score**: Moderate Risk
  - **Audit Verdict**: `APPROVED`
  - **Comps Retrieved**: `PROP-2336` (\$401,179), `PROP-2333` (\$556,581), `PROP-1637` (\$349,265)

### Query 3: Entry-Level Starter Home (Edwards)
- **Subject**: 1,050 sq ft, 2 Bed, 1 Bath, Year Built 1950, Overall Quality 4/10, Condition 5/10.
- **Request**: `POST /api/v1/valuation/collaborate`
- **Measured Response**:
  - **Reconciled Fair Market Value**: **\$100,862.85**
  - **90% Confidence Interval**: **\$62,851.50 – \$138,874.20**
  - **Market Tier**: Affordable / Entry-Level
  - **Risk Score**: Moderate Risk
  - **Audit Verdict**: `APPROVED`
  - **Comps Retrieved**: `PROP-0278` (\$64,000), `PROP-0284` (\$80,000), `PROP-2834` (\$97,900)

### Query 4: Metadata-Filtered Search (College Creek)
- **Query**: "Contemporary 3 bedroom home in College Creek built after 2000", living area 1,950 sq ft, Quality 7/10, Built 2002.
- **Metadata Filters**: `neighborhood: "College Creek"`, `min_bedrooms: 3`, `min_bathrooms: 2`, `min_year_built: 2000`.
- **Request**: `POST /api/v1/valuation/collaborate`
- **Measured Response**:
  - **Reconciled Fair Market Value**: **\$232,632.16**
  - **90% Confidence Interval**: **\$193,505.75 – \$271,758.57**
  - **Market Tier**: Upper-Middle Market
  - **Risk Score**: Moderate Risk
  - **Audit Verdict**: `APPROVED`
  - **Comps Retrieved**: `PROP-0867` (\$313,000), `PROP-1456` (\$235,000), `PROP-0877` (\$213,000)

---

## 9. Evaluation Results (DeepEval Benchmark)

The system was evaluated against 5 benchmark scenarios spanning different property tiers using DeepEval (v4.2.8). The live evaluation run completed with 100% pass rate:

- **Benchmark Status**: `PASSED`
- **Scenarios Evaluated**: 5 / 5 Passed (100.0%)
- **Total Execution Time**: 61.98s
- **Aggregate Metric Scores**:
  - **Faithfulness**: **0.900** (threshold: 0.70)
  - **Answer Relevancy**: **0.880** (threshold: 0.70)
  - **Contextual Precision**: **0.850** (threshold: 0.70)
  - **Contextual Recall**: **0.850** (threshold: 0.70)
  - **Valuation Accuracy**: **1.000** (threshold: 0.80)

### Benchmark Scenario Breakdown:
| Scenario ID | Property Profile | Expected Market Range | Reconciled Value | Accuracy | Verdict |
|---|---|---|---|---|---|
| `EVAL-01` | North Ames (1,600 sq ft, Qual 6) | \$160,000 – \$230,000 | \$201,690.61 | 1.000 | PASSED |
| `EVAL-02` | Stone Brook (2,700 sq ft, Qual 9) | \$310,000 – \$480,000 | \$402,728.01 | 1.000 | PASSED |
| `EVAL-03` | Edwards (1,100 sq ft, Qual 4) | \$90,000 – \$145,000 | \$108,312.10 | 1.000 | PASSED |
| `EVAL-04` | College Creek (1,900 sq ft, Qual 7)| \$180,000 – \$250,000 | \$210,246.70 | 1.000 | PASSED |
| `EVAL-05` | Brookside (1,200 sq ft, Qual 5) | \$85,000 – \$140,000 | \$121,564.37 | 1.000 | PASSED |

### Offline vs. Live LLM Modes:
- **Offline Mode** (`OPENAI_API_KEY` unset): All valuation, classification, risk, and search endpoints work completely offline using local ML models and deterministic reasoning. DeepEval returns `evaluation_mode = "offline_no_llm"` and `benchmark_status = "NOT_EVALUATED"` for LLM-backed metrics while rule-based valuation accuracy is computed.
- **Live LLM Mode** (`OPENAI_API_KEY` provided): DeepEval executes live LLM judge scoring for all 5 metrics, and agent narratives are generated by the configured LLM.

---

## 10. Compliance & Requirements Matrix

| Item # | Capstone Requirement | Status | Implementation File(s) / Endpoints |
|---|---|---|---|
| **1.1** | Basic RAG for property similarity search | **Completed** | `src/rag/vector_store.py`, `POST /api/v1/search` |
| **1.2** | True hybrid search (full candidate set TF-IDF + dense cosine) | **Completed** | `src/rag/hybrid_retriever.py`, `POST /api/v1/search` |
| **1.3** | Property classification agent | **Completed** | `src/agents/classification_agent.py`, `POST /api/v1/classify` |
| **1.4** | Metadata filtering (empty result on zero matches) | **Completed** | `src/rag/vector_store.py`, `src/rag/hybrid_retriever.py` |
| **1.5** | Basic property valuation recommendation | **Completed** | `src/agents/valuation_agent.py`, `POST /api/v1/valuation/basic` |
| **1.6** | Input validation guardrails (14 fields, injection defense) | **Completed** | `src/rag/guardrails.py`, `POST /api/v1/guardrails/validate` |
| **1.7** | Comparable property prioritization formula | **Completed** | `src/rag/hybrid_retriever.py` (`_calculate_comparability`) |
| **1.8** | REST API microservice (16 endpoints, OpenAPI docs) | **Completed** | `src/api/routes.py`, `src/main.py`, `/docs` |
| **2.1** | DeepEval benchmark framework (offline & live modes) | **Completed** | `src/services/evaluation_service.py`, `GET /api/v1/evaluation/run` |
| **2.2** | Supervised ML price regression ($R^2 = 0.9198$) | **Completed** | `src/services/ml_prediction.py`, `POST /api/v1/valuation/predict-price` |
| **2.3** | LLM-as-Judge valuation audit agent | **Completed** | `src/agents/judge_agent.py`, `POST /api/v1/valuation/judge-validate` |
| **2.4** | Autonomous multi-agent collaboration orchestrator | **Completed** | `src/agents/orchestrator.py`, `POST /api/v1/valuation/collaborate` |
| **2.5** | Multi-factor property risk prediction agent | **Completed** | `src/agents/risk_agent.py`, `POST /api/v1/valuation/predict-risk` |
| **2.6** | Feature impact & what-if renovation analysis | **Completed** | `src/services/feature_impact.py`, `POST /api/v1/valuation/impact-analysis` |
| **2.7** | Appraiser feedback loop & dynamic bias calibration | **Completed** | `src/services/feedback_service.py`, `POST /api/v1/feedback/submit` |
| **2.8** | Production React 18 UI with 7 tabs | **Completed** | `frontend/` (React 18.3.1 + Vite, 7 tabs, accessible tablist) |
| **4.1** | Architecture diagram (PNG & PDF) | **Completed** | `architecture/Architecture_Diagram.png`, `.pdf` |
| **4.2** | System design document (Markdown & PDF) | **Completed** | `design/Design_Document.md`, `.pdf` |
| **4.3** | Full executable code & documentation | **Completed** | `src/`, `frontend/`, `README.md`, `requirements.txt` |
| **4.4** | Panel presentation deck (PPTX & PDF, 10 slides) | **Completed** | `presentation/Project_Presentation.pptx`, `.pdf` |
| **5.0** | Dataset documentation (10,000 rows, 15 fields) | **Completed** | `docs/Dataset_Details.md`, `.pdf` |
| **Doc** | Comprehensive API documentation (16 endpoints) | **Completed** | `docs/API_Documentation.md`, `.pdf` |
| **Doc** | DeepEval evaluation benchmark report | **Completed** | `docs/Evaluation_Report.md`, `.pdf` |
| **Test**| Automated pytest suite (37 passing tests) | **Completed** | `tests/` (37 passing unit & integration tests) |

---

## 11. Authors & License

- **Presenter & Author**: Kunjal Ahuja
- **Project**: AI-Powered Real Estate Property Valuation Assistant
- **Course / Capstone**: Prodapt AI/ML Engineering Capstone
- **License**: MIT License
