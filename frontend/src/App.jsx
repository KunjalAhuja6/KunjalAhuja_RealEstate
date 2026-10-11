import { useState, useEffect, useRef, useMemo } from "react";
import { DEFAULTS, api } from "./api";
import { useTheme } from "./theme";

import Header from "./components/Header";
import SidebarForm from "./components/SidebarForm";

import ValuationTab from "./tabs/ValuationTab";
import ComparableSalesTab from "./tabs/ComparableSalesTab";
import MLPredictionTab from "./tabs/MLPredictionTab";
import WhatIfTab from "./tabs/WhatIfTab";
import RiskAndAuditTab from "./tabs/RiskAndAuditTab";
import FeedbackTab from "./tabs/FeedbackTab";
import EvaluationTab from "./tabs/EvaluationTab";

const TABS = [
  { id: "valuation", label: "Valuation" },
  { id: "comps", label: "Comparable sales" },
  { id: "ml", label: "ML prediction" },
  { id: "whatif", label: "What-if" },
  { id: "risk", label: "Risk and audit" },
  { id: "feedback", label: "Feedback" },
  { id: "evaluation", label: "Evaluation" },
];

export default function App() {
  const { theme, toggleTheme } = useTheme();

  // Shared property form
  const [form, setForm] = useState(DEFAULTS);
  const [query, setQuery] = useState("");

  // Navigation
  const [activeTabIdx, setActiveTabIdx] = useState(0);
  const tabButtonRefs = useRef([]);

  // Server & dataset status
  const [datasetInfo, setDatasetInfo] = useState(null);

  // Valuation results & snapshots
  const [valuationData, setValuationData] = useState(null);
  const [valuationSnapshot, setValuationSnapshot] = useState(null);
  const [valLoading, setValLoading] = useState(false);
  const [valError, setValError] = useState("");

  // Feedback summary (shared for Tab 6 & Tab 7)
  const [feedbackSummary, setFeedbackSummary] = useState(null);

  // Initial connection check
  useEffect(() => {
    api
      .getDatasetInfo()
      .then(setDatasetInfo)
      .catch(() => setDatasetInfo(false));

    api
      .getFeedbackSummary()
      .then(setFeedbackSummary)
      .catch(() => {});
  }, []);

  // Determine whether current form differs from snapshot
  const isOutOfDate = useMemo(() => {
    if (!valuationSnapshot) return false;
    const currentKeys = Object.keys(form);
    return currentKeys.some((k) => form[k] !== valuationSnapshot[k]);
  }, [form, valuationSnapshot]);

  // Execute collaborative valuation
  const runValuation = async (overrideForm = null) => {
    const targetForm = overrideForm || form;
    setValLoading(true);
    setValError("");
    try {
      const res = await api.collaborate(targetForm, query || null, 5);
      setValuationData(res);
      setValuationSnapshot({ ...targetForm });
    } catch (err) {
      setValError(err.message);
    } finally {
      setValLoading(false);
    }
  };

  // Switch tab helper
  const navigateTabByKey = (key) => {
    const idx = TABS.findIndex((t) => t.id === key);
    if (idx !== -1) {
      setActiveTabIdx(idx);
      tabButtonRefs.current[idx]?.focus();
    }
  };

  // Apply What-if changes into form and re-run valuation
  const handleApplyAndRevalue = (modifiedFeatures) => {
    if (!modifiedFeatures) return;
    const updatedForm = {
      ...form,
      gr_liv_area: Number(modifiedFeatures.gr_liv_area || modifiedFeatures.GrLivArea || form.gr_liv_area),
      overall_qual: Number(modifiedFeatures.overall_qual || modifiedFeatures.OverallQual || form.overall_qual),
      garage_cars: Number(modifiedFeatures.garage_cars || modifiedFeatures.GarageCars || form.garage_cars),
      bathrooms: Number(modifiedFeatures.bathrooms || modifiedFeatures.FullBath || form.bathrooms),
      year_remod_add: Number(modifiedFeatures.year_remod_add || modifiedFeatures.YearRemodAdd || form.year_remod_add || form.year_built),
    };
    setForm(updatedForm);
    setActiveTabIdx(0); // switch to Valuation tab
    runValuation(updatedForm);
  };

  // Keyboard navigation for accessible tablist
  const handleKeyDown = (e, index) => {
    if (e.key === "ArrowRight") {
      e.preventDefault();
      const nextIdx = (index + 1) % TABS.length;
      setActiveTabIdx(nextIdx);
      tabButtonRefs.current[nextIdx]?.focus();
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      const prevIdx = (index - 1 + TABS.length) % TABS.length;
      setActiveTabIdx(prevIdx);
      tabButtonRefs.current[prevIdx]?.focus();
    } else if (e.key === "Home") {
      e.preventDefault();
      setActiveTabIdx(0);
      tabButtonRefs.current[0]?.focus();
    } else if (e.key === "End") {
      e.preventDefault();
      const lastIdx = TABS.length - 1;
      setActiveTabIdx(lastIdx);
      tabButtonRefs.current[lastIdx]?.focus();
    }
  };

  const statusText = useMemo(() => {
    if (datasetInfo === null) return "Connecting…";
    if (datasetInfo === false) return "API offline: start server on port 8000";
    return `${datasetInfo.total_records.toLocaleString()} reference sales · ${datasetInfo.neighborhoods_count} neighborhoods`;
  }, [datasetInfo]);

  return (
    <div className="app-container">
      <Header
        statusText={statusText}
        isOffline={datasetInfo === false}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <main className="workbench-layout">
        <aside className="sidebar-column">
          <SidebarForm
            form={form}
            setForm={setForm}
            query={query}
            setQuery={setQuery}
            onRun={() => runValuation()}
            loading={valLoading}
          />
        </aside>

        <section className="main-content-column">
          {/* Accessible Tab List */}
          <nav
            className="tabs-nav-bar"
            role="tablist"
            aria-label="Valuation workbench modules"
          >
            {TABS.map((tab, idx) => {
              const isSelected = activeTabIdx === idx;
              return (
                <button
                  key={tab.id}
                  id={`tab-${tab.id}`}
                  ref={(el) => (tabButtonRefs.current[idx] = el)}
                  role="tab"
                  type="button"
                  aria-selected={isSelected}
                  aria-controls={`tabpanel-${tab.id}`}
                  tabIndex={isSelected ? 0 : -1}
                  className={`tab-item ${isSelected ? "selected" : ""}`}
                  onClick={() => setActiveTabIdx(idx)}
                  onKeyDown={(e) => handleKeyDown(e, idx)}
                >
                  {tab.label}
                </button>
              );
            })}
          </nav>

          {/* Tab Panels */}
          <div
            id={`tabpanel-${TABS[activeTabIdx].id}`}
            role="tabpanel"
            aria-labelledby={`tab-${TABS[activeTabIdx].id}`}
            className="tab-panel-container"
          >
            {activeTabIdx === 0 && (
              <ValuationTab
                data={valuationData}
                loading={valLoading}
                error={valError}
                isOutOfDate={isOutOfDate}
                onReRun={() => runValuation()}
                onNavigateTab={navigateTabByKey}
              />
            )}

            {activeTabIdx === 1 && (
              <ComparableSalesTab
                form={form}
                query={query}
                valuationData={valuationData}
                isOutOfDate={isOutOfDate}
                onReRun={() => runValuation()}
                onNavigateTab={navigateTabByKey}
              />
            )}

            {activeTabIdx === 2 && (
              <MLPredictionTab
                form={form}
                valuationData={valuationData}
                isOutOfDate={isOutOfDate}
                onReRun={() => runValuation()}
              />
            )}

            {activeTabIdx === 3 && (
              <WhatIfTab
                form={form}
                isOutOfDate={isOutOfDate}
                onReRun={() => runValuation()}
                onApplyAndRevalue={handleApplyAndRevalue}
              />
            )}

            {activeTabIdx === 4 && (
              <RiskAndAuditTab
                form={form}
                valuationData={valuationData}
                isOutOfDate={isOutOfDate}
                onReRun={() => runValuation()}
                onNavigateTab={navigateTabByKey}
              />
            )}

            {activeTabIdx === 5 && (
              <FeedbackTab
                form={form}
                valuationData={valuationData}
              />
            )}

            {activeTabIdx === 6 && (
              <EvaluationTab
                valuationData={valuationData}
                feedbackSummary={feedbackSummary}
              />
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
