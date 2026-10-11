#!/usr/bin/env python3
"""
generate_synthetic_data.py
==========================
Grows the Ames Housing dataset to a target size (default 10,000 rows) by adding
SYNTHETIC rows on top of the untouched real rows.

Design (read this before presenting it to a panel):
  * Real rows are never modified. Output goes to a NEW file; the input file is never
    overwritten, and re-running always starts again from the real rows (idempotent).
  * New rows are NOT jittered copies. A Gaussian copula is fitted on the real rows
    (marginals = empirical quantiles, dependence = rank correlation) and sampled
    conditionally on a neighbourhood, so neighbourhood price levels are respected.
  * A novelty filter rejects any candidate that sits closer to a real row (or to an
    already accepted synthetic row) than real houses sit to each other (default: the
    25th percentile of real nearest-neighbour gaps).
  * Dependent fields are rebuilt so rows stay internally consistent
    (floor areas, garage, years, rooms, basement, coordinates, derived columns).
  * Every synthetic row has is_synthetic=True and a "SYN-xxxxx" Property_ID.
  * A validation report is produced, including a "train on real vs real+synthetic,
    test on HELD-OUT REAL rows" accuracy check.

Usage (from the repo root):
    python generate_synthetic_data.py                       # 10,000 total rows
    python generate_synthetic_data.py --target-total 8000 --seed 7
    python generate_synthetic_data.py --skip-ml-check       # faster, no accuracy check
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp, norm, rankdata

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

DEFAULT_IN = REPO / "data" / "sample_data" / "ames_housing_cleaned.csv"
DEFAULT_OUT = REPO / "data" / "sample_data" / "ames_housing_augmented.csv"
DEFAULT_REPORT = REPO / "docs" / "synthetic_data_report.json"

CORE = [
    "GrLivArea", "LotArea", "TotalBsmtSF", "OverallQual", "OverallCond",
    "YearBuilt", "YearRemodAdd", "GarageCars", "GarageArea", "FullBath",
    "BedroomAbvGr", "TotRmsAbvGrd", "SalePrice",
]
TIER = "_nbhd_tier"          # neighbourhood median price/sqft (used for conditioning)
CURRENT_YEAR = 2026          # same convention as src/rag/data_pipeline.py


# --------------------------------------------------------------------------- #
# 1. Copula fit + conditional sampling
# --------------------------------------------------------------------------- #
class NeighbourhoodCopula:
    def __init__(self, real: pd.DataFrame):
        self.real = real
        n = len(real)
        self.tier_map = real.groupby("Neighborhood")["PricePerSqFt"].median()
        M = real[CORE].astype(float).copy()
        M[TIER] = real["Neighborhood"].map(self.tier_map).values
        self.cols = CORE + [TIER]
        self.sorted_vals = {c: np.sort(M[c].values) for c in self.cols}
        U = np.column_stack([(rankdata(M[c], method="average") - 0.5) / n for c in self.cols])
        G = norm.ppf(U)
        C = np.corrcoef(G.T)
        self.C_rr = C[:-1, :-1]
        self.C_rt = C[:-1, -1]
        # conditional covariance of the 13 core variables given the tier variable
        cov = self.C_rr - np.outer(self.C_rt, self.C_rt)
        cov = (cov + cov.T) / 2 + 1e-6 * np.eye(len(cov))
        self.L = np.linalg.cholesky(cov)
        # z-score of each neighbourhood's tier on the Gaussian scale
        tier_u = pd.Series(U[:, -1], index=real.index)
        self.z_tier = tier_u.groupby(real["Neighborhood"]).first().pipe(lambda s: pd.Series(norm.ppf(s.values), index=s.index))
        freq = real["Neighborhood"].value_counts(normalize=True)
        self.nbhds = freq.index.to_numpy()
        self.freq = freq.values

    def sample(self, m: int, rng: np.random.Generator) -> pd.DataFrame:
        nb = rng.choice(self.nbhds, size=m, p=self.freq)
        z_t = self.z_tier.loc[nb].values
        mean = np.outer(z_t, self.C_rt)
        g = mean + rng.standard_normal((m, len(CORE))) @ self.L.T
        u = norm.cdf(g)
        out = {c: np.quantile(self.sorted_vals[c], u[:, j]) for j, c in enumerate(CORE)}
        df = pd.DataFrame(out)
        df["Neighborhood"] = nb
        df[TIER] = self.tier_map.loc[nb].values
        return df


# --------------------------------------------------------------------------- #
# 2. Core constraints (integers, garage logic, rooms, price rounding)
# --------------------------------------------------------------------------- #
def enforce_core(df: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    lo, hi = real["GrLivArea"].min(), real["GrLivArea"].max()
    d["GrLivArea"] = d["GrLivArea"].round().clip(lo, hi)
    d["LotArea"] = d["LotArea"].round().clip(real["LotArea"].min(), real["LotArea"].max())
    d["OverallQual"] = d["OverallQual"].round().clip(1, 10)
    d["OverallCond"] = d["OverallCond"].round().clip(1, 10)
    d["YearBuilt"] = d["YearBuilt"].round().clip(real["YearBuilt"].min(), real["YearBuilt"].max())
    d["YearRemodAdd"] = d["YearRemodAdd"].round().clip(real["YearRemodAdd"].min(), real["YearRemodAdd"].max())
    d["YearRemodAdd"] = np.maximum(d["YearRemodAdd"], d["YearBuilt"])
    d["FullBath"] = d["FullBath"].round().clip(0, real["FullBath"].max())
    d["BedroomAbvGr"] = d["BedroomAbvGr"].round().clip(0, real["BedroomAbvGr"].max())
    d["TotRmsAbvGrd"] = np.maximum(d["TotRmsAbvGrd"].round(), d["BedroomAbvGr"] + 1)
    d["TotRmsAbvGrd"] = d["TotRmsAbvGrd"].clip(upper=real["TotRmsAbvGrd"].max())
    # garage: cars is an integer, area is plausible per car, no cars -> no area
    d["GarageCars"] = d["GarageCars"].round().clip(0, real["GarageCars"].max())
    ga = d["GarageArea"].round()
    per_car = np.clip(ga / d["GarageCars"].replace(0, np.nan), 150, 480)
    d["GarageArea"] = np.where(d["GarageCars"] == 0, 0, (per_car * d["GarageCars"]).round())
    # basement: tiny values -> no basement
    b = d["TotalBsmtSF"].round()
    d["TotalBsmtSF"] = np.where(b < 50, 0, b).clip(0, real["TotalBsmtSF"].max())
    d["SalePrice"] = ((d["SalePrice"] / 100).round() * 100).clip(real["SalePrice"].min(), real["SalePrice"].max())
    return d


# --------------------------------------------------------------------------- #
# 3. Novelty filter
# --------------------------------------------------------------------------- #
class Novelty:
    def __init__(self, real: pd.DataFrame, tier_map: pd.Series, percentile: float):
        self.cols = CORE + [TIER]
        R = real[CORE].astype(float).copy()
        R[TIER] = real["Neighborhood"].map(tier_map).values
        self.mu = R.mean().values
        self.sd = R.std().replace(0, 1).values
        self.Zr = (R.values - self.mu) / self.sd
        self.tree = cKDTree(self.Zr)
        d, _ = self.tree.query(self.Zr, k=2)
        self.real_gaps = d[:, 1]
        self.threshold = float(np.percentile(self.real_gaps, percentile))

    def z(self, df: pd.DataFrame) -> np.ndarray:
        return (df[self.cols].astype(float).values - self.mu) / self.sd

    def accept(self, cand: pd.DataFrame, accepted_z: list) -> np.ndarray:
        """Boolean mask of candidates that are novel vs real rows and vs accepted ones."""
        Z = self.z(cand)
        d_real, _ = self.tree.query(Z, k=1)
        keep = d_real >= self.threshold
        A = np.array(accepted_z) if accepted_z else np.empty((0, Z.shape[1]))
        buf = np.empty((len(A) + int(keep.sum()), Z.shape[1]))
        buf[: len(A)] = A
        n = len(A)
        for i in np.flatnonzero(keep):
            if n and np.min(np.linalg.norm(buf[:n] - Z[i], axis=1)) < self.threshold:
                keep[i] = False
            else:
                buf[n] = Z[i]
                n += 1
        return keep


# --------------------------------------------------------------------------- #
# 4. Build full rows: donor attributes + rebuilt dependent fields
# --------------------------------------------------------------------------- #
def build_rows(core: pd.DataFrame, real: pd.DataFrame, rng: np.random.Generator, start_id: int) -> pd.DataFrame:
    real = real.reset_index(drop=True)
    sd = {"OverallQual": real["OverallQual"].std(), "YearBuilt": real["YearBuilt"].std(), "GrLivArea": real["GrLivArea"].std()}
    has_b = real["TotalBsmtSF"] > 0
    has_g = real["GarageCars"] > 0
    groups = {}
    for key, idx in real.groupby(["Neighborhood", has_b, has_g]).groups.items():
        groups[key] = np.array(idx)
    by_nb = {k: np.array(v) for k, v in real.groupby("Neighborhood").groups.items()}

    donor_idx = []
    for r in core.itertuples(index=False):
        key = (r.Neighborhood, r.TotalBsmtSF > 0, r.GarageCars > 0)
        pool = groups.get(key)
        if pool is None or len(pool) < 2:
            pool = by_nb[r.Neighborhood]
        sub = real.loc[pool]
        dist = (
            ((sub["OverallQual"] - r.OverallQual) / sd["OverallQual"]) ** 2
            + ((sub["YearBuilt"] - r.YearBuilt) / sd["YearBuilt"]) ** 2
            + ((sub["GrLivArea"] - r.GrLivArea) / sd["GrLivArea"]) ** 2
        ).values
        k = min(5, len(pool))
        donor_idx.append(rng.choice(pool[np.argsort(dist)[:k]]))
    donors = real.iloc[donor_idx].reset_index(drop=True)
    out = donors.copy()
    c = core.reset_index(drop=True)

    for col in CORE:
        out[col] = c[col].values
    out["Neighborhood"] = c["Neighborhood"].values

    # years must be consistent with the (donor) sale year
    ys = out["Year_Sold"].values
    out["YearBuilt"] = np.minimum(out["YearBuilt"].values, ys)
    out["YearRemodAdd"] = np.clip(out["YearRemodAdd"].values, out["YearBuilt"].values, ys)
    # garage year: same convention as the real data (built with the house when no garage)
    gy_off = np.maximum(0, donors["GarageYrBlt"].values - donors["YearBuilt"].values)
    out["GarageYrBlt"] = np.where(out["GarageCars"] > 0, np.minimum(out["YearBuilt"] + gy_off, ys), out["YearBuilt"])

    # floors: keep the real identity 1st + 2nd + low-qual = GrLivArea
    ratio2 = np.where(donors["GrLivArea"] > 0, donors["Second_Flr_SF"] / donors["GrLivArea"], 0)
    second = np.round(out["GrLivArea"].values * ratio2)
    out["Second_Flr_SF"] = second.astype(int)
    out["Low_Qual_Fin_SF"] = 0
    out["First_Flr_SF"] = (out["GrLivArea"].values - second).astype(int)

    # basement: cap vs first floor, rescale the donor's sub-areas proportionally
    out["TotalBsmtSF"] = np.minimum(out["TotalBsmtSF"], np.where(out["TotalBsmtSF"] > 0, np.round(out["First_Flr_SF"] * 1.3), 0))
    r_b = np.where(donors["TotalBsmtSF"] > 0, out["TotalBsmtSF"] / donors["TotalBsmtSF"].replace(0, 1), 0)
    for col in ("BsmtFin_SF_1", "BsmtFin_SF_2", "Bsmt_Unf_SF"):
        if col in out and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = np.round(donors[col].values * r_b).astype(int)

    if "Lot_Frontage" in out and pd.api.types.is_numeric_dtype(out["Lot_Frontage"]):
        scale = np.sqrt(out["LotArea"].values / donors["LotArea"].replace(0, 1).values)
        out["Lot_Frontage"] = np.round(donors["Lot_Frontage"].values * scale).astype(int)

    # coordinates: stay inside the neighbourhood's real footprint
    lat, lon = out["Latitude"].values.copy(), out["Longitude"].values.copy()
    for nb, idx in out.groupby("Neighborhood").groups.items():
        pts = real[real["Neighborhood"] == nb]
        for col, arr in (("Latitude", lat), ("Longitude", lon)):
            s = max(0.3 * pts[col].std(ddof=0), 0.0003)
            pos = out.index.get_indexer(idx)
            base = rng.choice(pts[col].values, size=len(pos))
            arr[pos] = np.clip(base + rng.normal(0, s, len(pos)), pts[col].min() - s, pts[col].max() + s)
    out["Latitude"], out["Longitude"] = np.round(lat, 6), np.round(lon, 6)

    # derived columns (same formulas as data_pipeline.clean_and_preprocess_dataset)
    out["PropertyAge"] = CURRENT_YEAR - out["YearBuilt"]
    out["EffectiveAge"] = CURRENT_YEAR - out["YearRemodAdd"]
    out["PricePerSqFt"] = (out["SalePrice"] / out["GrLivArea"].replace(0, 1)).round(2)
    out["Property_ID"] = [f"SYN-{start_id + i:05d}" for i in range(len(out))]
    out["is_synthetic"] = True

    for col in CORE:                                   # integers like the real file
        out[col] = out[col].astype(int)
    for col in ("YearBuilt", "YearRemodAdd", "GarageYrBlt", "PropertyAge", "EffectiveAge"):
        out[col] = out[col].astype(int)
    return out[real.columns]


# --------------------------------------------------------------------------- #
# 5. Validation
# --------------------------------------------------------------------------- #
def consistency_checks(d: pd.DataFrame) -> dict:
    ok = lambda s: round(float(np.mean(s)), 4)
    return {
        "floors_sum_to_living_area": ok((d.First_Flr_SF + d.Second_Flr_SF + d.Low_Qual_Fin_SF - d.GrLivArea).abs() < 2),
        "rooms_ge_bedrooms_plus_1": ok(d.TotRmsAbvGrd >= d.BedroomAbvGr + 1),
        "no_garage_means_zero_area": ok(np.where(d.GarageCars == 0, d.GarageArea == 0, d.GarageArea > 0)),
        "remodel_not_before_built": ok(d.YearRemodAdd >= d.YearBuilt),
        "built_not_after_sale": ok(d.Year_Sold >= d.YearBuilt),
        "remodel_not_after_sale": ok(d.Year_Sold >= d.YearRemodAdd),
        "garage_year_not_before_built": ok(d.GarageYrBlt >= d.YearBuilt),
        "price_per_sqft_matches": ok((d.PricePerSqFt - d.SalePrice / d.GrLivArea).abs() < 0.01),
        "quality_in_1_to_10": ok(d.OverallQual.between(1, 10) & d.OverallCond.between(1, 10)),
        "unique_property_ids": bool(d.Property_ID.is_unique),
    }


def fidelity(real: pd.DataFrame, syn: pd.DataFrame) -> dict:
    ks = {c: round(float(ks_2samp(real[c], syn[c]).statistic), 4) for c in CORE}
    cr = real[CORE].corr(method="spearman").values
    cs = syn[CORE].corr(method="spearman").values
    nb_ratio = (syn.groupby("Neighborhood")["PricePerSqFt"].median() / real.groupby("Neighborhood")["PricePerSqFt"].median()).dropna()
    return {
        "ks_statistic_per_column(0=identical)": ks,
        "mean_abs_diff_of_spearman_correlations": round(float(np.abs(cr - cs).mean()), 4),
        "neighbourhood_median_ppsf_ratio(syn/real)": {"min": round(float(nb_ratio.min()), 3), "max": round(float(nb_ratio.max()), 3),
                                                       "median": round(float(nb_ratio.median()), 3)},
    }


def ml_utility(real: pd.DataFrame, syn: pd.DataFrame, seed: int) -> dict:
    """Train on real-only vs real+synthetic; ALWAYS test on held-out REAL rows."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_percentage_error as mape, r2_score
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    try:
        from src.services.ml_prediction import NUMERIC_FEATURES
    except Exception:
        NUMERIC_FEATURES = [c for c in CORE if c != "SalePrice"] + ["GarageYrBlt"]

    def model():
        prep = ColumnTransformer([("n", "passthrough", NUMERIC_FEATURES),
                                  ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Neighborhood"])])
        gb = GradientBoostingRegressor(n_estimators=180, learning_rate=0.08, max_depth=5, random_state=42)
        return Pipeline([("p", prep), ("m", gb)])

    feats = NUMERIC_FEATURES + ["Neighborhood"]
    res = {"real_only": {"r2": [], "mape": []}, "real_plus_synthetic": {"r2": [], "mape": []}}
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(real):
        test = real.iloc[te]
        for name, train in (("real_only", real.iloc[tr]), ("real_plus_synthetic", pd.concat([real.iloc[tr], syn]))):
            m = model().fit(train[feats], train["SalePrice"])
            p = m.predict(test[feats])
            res[name]["r2"].append(r2_score(test["SalePrice"], p))
            res[name]["mape"].append(mape(test["SalePrice"], p))
    return {k: {"r2": round(float(np.mean(v["r2"])), 4), "mape_pct": round(float(np.mean(v["mape"])) * 100, 2)} for k, v in res.items()}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--target-total", type=int, default=10000, help="real + synthetic rows in the output")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-gap-percentile", type=float, default=25.0,
                    help="novelty threshold = this percentile of real nearest-neighbour gaps")
    ap.add_argument("--skip-ml-check", action="store_true")
    a = ap.parse_args()

    if a.input.resolve() == a.output.resolve():
        sys.exit("Refusing to overwrite the input file. Choose a different --output.")
    t0 = time.time()
    real = pd.read_csv(a.input)
    if "is_synthetic" in real.columns:          # idempotent: always rebuild from real rows only
        real = real[~real["is_synthetic"].astype(bool)].drop(columns=["is_synthetic"])
    real = real.reset_index(drop=True)
    n_real, need = len(real), a.target_total - len(real)
    if need <= 0:
        sys.exit(f"Target {a.target_total} <= real rows ({n_real}); nothing to generate.")
    print(f"Real rows: {n_real} | synthetic needed: {need} | target total: {a.target_total}")

    rng = np.random.default_rng(a.seed)
    copula = NeighbourhoodCopula(real)
    nov = Novelty(real, copula.tier_map, a.min_gap_percentile)
    print(f"Novelty threshold (scaled distance): {nov.threshold:.3f}  "
          f"[{a.min_gap_percentile:.0f}th pct of real-real gaps; real median gap {np.median(nov.real_gaps):.3f}]")

    accepted_core, accepted_z, tried = [], [], 0
    n_acc = 0
    for _ in range(40):
        remaining = need - n_acc
        if remaining <= 0:
            break
        cand = enforce_core(copula.sample(int(remaining * 1.4) + 200, rng), real)
        keep = nov.accept(cand, accepted_z)
        tried += len(cand)
        take = cand[keep].head(remaining)
        accepted_core.append(take)
        accepted_z.extend(nov.z(take).tolist())
        n_acc += len(take)
        print(f"  batch: sampled {len(cand)}, accepted {len(take)} (total {n_acc}/{need})")
    core = pd.concat(accepted_core, ignore_index=True)
    if len(core) < need:
        sys.exit(f"Could only produce {len(core)} novel rows; lower --min-gap-percentile or --target-total.")

    syn = build_rows(core, real, rng, start_id=1)
    # searchable text uses the project's own function so retrieval is unchanged
    try:
        from src.rag.data_pipeline import generate_property_document
        syn["searchable_text"] = syn.apply(generate_property_document, axis=1)
    except Exception as e:
        sys.exit(f"Run this script from the repo root so 'src' is importable ({e}).")

    syn["is_synthetic"] = True
    real_out = real.copy()
    real_out["is_synthetic"] = False
    full = pd.concat([real_out, syn], ignore_index=True)
    full["is_synthetic"] = full["is_synthetic"].astype(bool)
    full = full[list(real.columns) + ["is_synthetic"]]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(a.output, index=False)

    # ---- validation ----
    syn_v = syn.copy()
    syn_v[TIER] = syn_v["Neighborhood"].map(copula.tier_map).values
    Zs = nov.z(syn_v)      # novelty measured on the FINAL rows, after all consistency fixes
    d_real, _ = nov.tree.query(Zs, k=1)
    d_syn, _ = cKDTree(Zs).query(Zs, k=2)
    report = {
        "rows": {"real": n_real, "synthetic": len(syn), "total": len(full),
                 "synthetic_share_pct": round(100 * len(syn) / len(full), 1)},
        "settings": {"seed": a.seed, "min_gap_percentile": a.min_gap_percentile, "novelty_threshold": round(nov.threshold, 4),
                     "candidates_sampled": tried, "acceptance_rate_pct": round(100 * len(syn) / tried, 1)},
        "novelty": {
            "min_distance_to_any_real_row": round(float(d_real.min()), 4),
            "median_distance_to_nearest_real_row": round(float(np.median(d_real)), 4),
            "median_real_to_real_gap": round(float(np.median(nov.real_gaps)), 4),
            "min_distance_between_synthetic_rows": round(float(d_syn[:, 1].min()), 4),
            "exact_duplicates_of_real_rows(core fields)": int(syn[CORE + ["Neighborhood"]].merge(real[CORE + ["Neighborhood"]]).shape[0]),
        },
        "consistency_real_rows(for reference)": consistency_checks(real),
        "consistency_synthetic_rows": consistency_checks(syn),
        "fidelity": fidelity(real, syn),
    }
    if not a.skip_ml_check:
        print("Running accuracy check (train on real vs real+synthetic, test on held-out real) ...")
        report["ml_utility_5fold_heldout_real"] = ml_utility(real, syn, a.seed)
    report["runtime_seconds"] = round(time.time() - t0, 1)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {a.output}  ({len(full)} rows)\nWrote {a.report}")


if __name__ == "__main__":
    main()
