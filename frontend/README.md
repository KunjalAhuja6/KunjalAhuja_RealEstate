# Frontend: Real Estate Valuation Assistant UI

A production Single Page Application (SPA) built with **React 18** and **Vite** serving as the interactive interface for the **AI-Powered Real Estate Property Valuation Assistant** (Author / Presenter: Kunjal Ahuja).

---

## 1. Environment & Prerequisites

- **Node.js**: Node 18.0.0+ required (tested on Node v23.11.0)
- **Package Manager**: npm 9+ (or pnpm / yarn)
- **Backend Service**: FastAPI running on `http://localhost:8000`

---

## 2. Getting Started

### Installation
From the repository root or the `frontend/` directory:
```bash
cd frontend
npm install
```

### Starting Development Server
```bash
npm run dev
```
- The application will be available at **`http://localhost:5173`**.
- Vite automatically proxies API calls matching `/api/*` to `http://localhost:8000`.

### Building for Production
```bash
npm run build
```
Generates static assets in the `dist/` directory. You can preview the production build locally with:
```bash
npm run preview
```

---

## 3. Architecture & API Proxy

- **Vite Proxy (`vite.config.js`)**:
  ```javascript
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  }
  ```
- **Optional API URL Override**:
  You can set `VITE_API_URL` in a `.env` or `.env.local` file inside `frontend/` if connecting to a remote backend host:
  ```env
  VITE_API_URL=http://localhost:8000
  ```

---

## 4. Workflow Tabs (7 Modules)

The interface features a shared 14-field property input sidebar on the left and 7 dedicated workflow tabs:

1. **Valuation Tab** (`ValuationTab.jsx`):
   - Interactive valuation range bar (`RangeBar.jsx`) displaying lower bound, point estimate, and upper bound.
   - Grounded plain-English narrative citing verified comparables, price-per-square-foot metrics, and quality adjustments.
   - Market tier and typology badges.

2. **Comparable Sales Tab** (`ComparableSalesTab.jsx`):
   - Ranked historical comparable transactions retrieved via hybrid search ($\alpha = 0.55$).
   - Physical comparability index breakdown (Location 30%, Living Area 25%, Quality 20%, Age 15%, Room Count 10%).
   - Verified real-only comp indicators (synthetic records excluded).

3. **ML Prediction Tab** (`MLPredictionTab.jsx`):
   - Supervised `GradientBoostingRegressor` point estimate ($R^2 = 0.9198$).
   - 90% empirical prediction interval calculated from real test residuals.
   - Top feature drivers (Living Area, Overall Quality, Basement SF, Garage Area, Year Built).

4. **What-If Simulator Tab** (`WhatIfTab.jsx`):
   - Real-time simulation of property renovations and upgrades (e.g., adding full baths, expanding square footage, improving quality score).
   - Instant calculation of expected dollar value lift.

5. **Risk and Audit Tab** (`RiskAndAuditTab.jsx`):
   - Composite 0–100 physical and market risk gauge across 4 dimensions: physical depreciation, condition discrepancy, neighborhood volatility, and layout liquidity.
   - Independent LLM-as-Judge audit status badge (`APPROVED`, `FLAGGED_OVERVALUED`, `FLAGGED_UNDERVALUED`, `NEEDS_REVIEW`).

6. **Feedback Tab** (`FeedbackTab.jsx`):
   - Appraiser closing price logger and star rating submission.
   - Live calibration factor tracking ($\text{recommended} / \text{applied\_factor}$) to monitor continuous pricing bias adjustments.

7. **Evaluation Tab** (`EvaluationTab.jsx`):
   - Trigger and view DeepEval quality benchmark runs.
   - History retention for the last 3 runs with run timestamp selector.
   - Real metrics: Faithfulness (0.900), Answer Relevancy (0.880), Contextual Precision (0.850), Contextual Recall (0.850), Valuation Accuracy (1.000).

---

## 5. UI Features & Accessibility

- **Theme**: Light / Dark mode toggle with system preference detection and localStorage persistence.
- **Form State Synchronization**: Notice alerts when sidebar parameters are modified after a valuation run.
- **Defensive Error Handling**: Empty states, working/loading indicators, and user-friendly error banners.
