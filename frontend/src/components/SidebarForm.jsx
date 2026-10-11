import { NEIGHBORHOODS } from "../api";

export default function SidebarForm({
  form,
  setForm,
  query,
  setQuery,
  onRun,
  loading,
}) {
  const updateField = (key, val) => {
    setForm((prev) => ({ ...prev, [key]: val }));
  };

  const handleNumChange = (key) => (e) => {
    const raw = e.target.value;
    updateField(key, raw === "" ? "" : Number(raw));
  };

  return (
    <form
      className="sidebar-sheet"
      onSubmit={(e) => {
        e.preventDefault();
        onRun();
      }}
    >
      <div className="sheet-header">
        <h2>Subject property</h2>
        <p className="sheet-subtitle">Key attributes for market appraisal</p>
      </div>

      <div className="form-group">
        <label htmlFor="prop-neighborhood">Neighborhood</label>
        <select
          id="prop-neighborhood"
          value={form.neighborhood || "North Ames"}
          onChange={(e) => updateField("neighborhood", e.target.value)}
        >
          {NEIGHBORHOODS.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </div>

      <div className="form-grid-2">
        <div className="form-group">
          <label htmlFor="prop-liv-area">Living area (sq ft)</label>
          <input
            id="prop-liv-area"
            type="number"
            min="300"
            max="6000"
            step="50"
            required
            value={form.gr_liv_area ?? 1800}
            onChange={handleNumChange("gr_liv_area")}
          />
        </div>

        <div className="form-group">
          <label htmlFor="prop-lot-area">Lot area (sq ft)</label>
          <input
            id="prop-lot-area"
            type="number"
            min="1000"
            max="100000"
            step="250"
            value={form.lot_area ?? 9000}
            onChange={handleNumChange("lot_area")}
          />
        </div>

        <div className="form-group">
          <label htmlFor="prop-bedrooms">Bedrooms</label>
          <input
            id="prop-bedrooms"
            type="number"
            min="0"
            max="10"
            value={form.bedrooms ?? 3}
            onChange={handleNumChange("bedrooms")}
          />
        </div>

        <div className="form-group">
          <label htmlFor="prop-baths">Full baths</label>
          <input
            id="prop-baths"
            type="number"
            min="0"
            max="6"
            value={form.bathrooms ?? 2}
            onChange={handleNumChange("bathrooms")}
          />
        </div>

        <div className="form-group">
          <label htmlFor="prop-year-built">Year built</label>
          <input
            id="prop-year-built"
            type="number"
            min="1870"
            max="2026"
            value={form.year_built ?? 1995}
            onChange={handleNumChange("year_built")}
          />
        </div>

        <div className="form-group">
          <label htmlFor="prop-bsmt">Basement (sq ft)</label>
          <input
            id="prop-bsmt"
            type="number"
            min="0"
            max="4000"
            step="50"
            value={form.total_bsmt_sf ?? 900}
            onChange={handleNumChange("total_bsmt_sf")}
          />
        </div>

        <div className="form-group">
          <label htmlFor="prop-garage">Garage spaces</label>
          <input
            id="prop-garage"
            type="number"
            min="0"
            max="5"
            value={form.garage_cars ?? 2}
            onChange={handleNumChange("garage_cars")}
          />
        </div>

        <div className="form-group">
          <label htmlFor="prop-cond">Condition (1–10)</label>
          <input
            id="prop-cond"
            type="number"
            min="1"
            max="10"
            value={form.overall_cond ?? 6}
            onChange={handleNumChange("overall_cond")}
          />
        </div>
      </div>

      <div className="form-group">
        <div className="slider-label-row">
          <label htmlFor="prop-qual">Build quality</label>
          <span className="slider-value-badge">{form.overall_qual ?? 7} of 10</span>
        </div>
        <input
          id="prop-qual"
          type="range"
          min="1"
          max="10"
          value={form.overall_qual ?? 7}
          onChange={handleNumChange("overall_qual")}
        />
      </div>

      <div className="form-group">
        <label htmlFor="prop-note">Appraiser note (optional)</label>
        <textarea
          id="prop-note"
          rows="3"
          placeholder="e.g. Modern kitchen remodel, quiet cul-de-sac, client needs quick close"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      <button
        type="submit"
        className="primary-button run-button"
        disabled={loading}
      >
        {loading ? "Working on your valuation…" : "Run valuation"}
      </button>
    </form>
  );
}
