export const API_BASE = (import.meta.env.VITE_API_URL || "") + "/api/v1";

export const NEIGHBORHOODS = [
  "Bloomington Heights",
  "Blueste",
  "Briardale",
  "Brookside",
  "Clear Creek",
  "College Creek",
  "Crawford",
  "Edwards",
  "Gilbert",
  "Green Hills",
  "Greens",
  "Iowa DOT and Rail Road",
  "Landmark",
  "Meadow Village",
  "Mitchell",
  "North Ames",
  "Northpark Villa",
  "Northridge",
  "Northridge Heights",
  "Northwest Ames",
  "Old Town",
  "Sawyer",
  "Sawyer West",
  "Somerset",
  "South and West of Iowa State University",
  "Stone Brook",
  "Timberland",
  "Veenker",
];

export const DEFAULTS = {
  neighborhood: "North Ames",
  gr_liv_area: 1800,
  lot_area: 9000,
  bedrooms: 3,
  bathrooms: 2,
  year_built: 1995,
  total_bsmt_sf: 900,
  garage_cars: 2,
  overall_cond: 6,
  overall_qual: 7,
};

export function toPayload(f) {
  const yb = Number(f.year_built) || 1995;
  const gc = Number(f.garage_cars) || 0;
  return {
    neighborhood: f.neighborhood || "North Ames",
    gr_liv_area: Number(f.gr_liv_area) || 1600,
    lot_area: Number(f.lot_area) || 9000,
    bedrooms: Number(f.bedrooms) || 3,
    bathrooms: Number(f.bathrooms) || 2,
    year_built: yb,
    year_remod_add: Number(f.year_remod_add || yb),
    overall_qual: Number(f.overall_qual) || 6,
    overall_cond: Number(f.overall_cond) || 5,
    total_bsmt_sf: Number(f.total_bsmt_sf) || 900,
    garage_cars: gc,
    garage_area: gc * 240,
  };
}

export function usd(n) {
  if (n == null || isNaN(n)) return "–";
  return Number(n).toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

export function pct(n) {
  if (n == null || isNaN(n)) return "–";
  const num = Number(n);
  return `${num >= 0 ? "+" : ""}${num.toFixed(1)}%`;
}

function cleanErrorMessage(detail) {
  if (!detail) return "The request could not be processed.";
  if (typeof detail === "string") {
    // Translate common raw backend error messages into plain agent-friendly sentences
    return detail
      .replace(/gr_liv_area/g, "living area")
      .replace(/lot_area/g, "lot area")
      .replace(/overall_qual/g, "build quality")
      .replace(/overall_cond/g, "condition")
      .replace(/total_bsmt_sf/g, "basement square footage")
      .replace(/year_built/g, "year built");
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const field = item.loc ? item.loc[item.loc.length - 1] : "field";
        const cleanField = String(field).replace(/_/g, " ");
        return `${cleanField}: ${item.msg || "invalid value"}`;
      })
      .join("; ");
  }
  return String(detail);
}

export async function call(path, body, method = "POST") {
  const url = API_BASE + path;
  let res;
  try {
    res = await fetch(url, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    throw new Error("Unable to connect to the valuation service. Please ensure the server is running on port 8000.");
  }

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const message = cleanErrorMessage(data.detail);
    throw new Error(message || `Request failed with status ${res.status}`);
  }
  return data;
}

export const api = {
  getDatasetInfo: () => call("/dataset/info", null, "GET"),
  collaborate: (payload, query = null, topK = 5) =>
    call("/valuation/collaborate", {
      property_features: toPayload(payload),
      query: query || null,
      top_k_comps: topK,
    }),
  basicValuation: (payload, query = null, topK = 5) =>
    call("/valuation/basic", {
      property_features: toPayload(payload),
      query: query || null,
      top_k_comps: topK,
    }),
  searchComparables: (query, topK = 5, filters = {}, targetProperty = null) =>
    call("/search", {
      query,
      top_k: topK,
      filters: filters && Object.keys(filters).length > 0 ? filters : {},
      target_property: targetProperty ? toPayload(targetProperty) : null,
    }),
  predictPrice: (payload) =>
    call("/valuation/predict-price", {
      property_features: toPayload(payload),
    }),
  whatIf: (basePayload, modifications) =>
    call("/valuation/impact-analysis", {
      base_features: toPayload(basePayload),
      modifications,
    }),
  predictRisk: (payload) =>
    call("/valuation/predict-risk", {
      property_features: toPayload(payload),
    }),
  judgeValidate: (validationRequest) =>
    call("/valuation/judge-validate", validationRequest),
  submitFeedback: (feedbackRequest) =>
    call("/feedback/submit", feedbackRequest),
  getFeedbackSummary: () =>
    call("/feedback/summary", null, "GET"),
  runEvaluation: () =>
    call("/evaluation/run", null, "GET"),
};
