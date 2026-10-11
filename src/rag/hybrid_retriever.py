import re
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.config import settings
from src.utils.logger import get_logger
from src.rag.vector_store import vector_store
from src.rag.data_pipeline import load_cleaned_data, KEY_FIELDS

logger = get_logger("HybridRetriever")

ALLOWED_DATA_FIELDS = KEY_FIELDS + ["Property_ID", "PricePerSqFt", "PropertyAge", "EffectiveAge", "is_synthetic"]

def sanitize_comp_data(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitizes row dict to only include key fields, casting NaNs to safe JSON types."""
    clean = {}
    for k, v in row_dict.items():
        if k in ALLOWED_DATA_FIELDS:
            if pd.isna(v) or v is None:
                clean[k] = 0.0 if isinstance(v, (float, int)) else ""
            elif isinstance(v, (np.floating, float)):
                clean[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                clean[k] = int(v)
            else:
                clean[k] = str(v)
    return clean

class HybridRetriever:
    """
    Hybrid retriever combining TF-IDF lexical keyword search with dense semantic embeddings,
    metadata filtering, and multi-factor comparable-property prioritization.
    """
    def __init__(self, alpha: float = settings.HYBRID_ALPHA):
        self.alpha = alpha  # Weight on semantic (alpha) vs keyword (1 - alpha)
        self.vector_store = vector_store
        self.df = vector_store.df
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z0-9_-]+\b",
            ngram_range=(1, 2),
            max_features=5000,
        )
        logger.info("Fitting TF-IDF index on property text documents...")
        self.tfidf_matrix = self.tfidf.fit_transform(self.df["searchable_text"])
        logger.info("TF-IDF matrix built successfully.")

    def keyword_search(
        self,
        query: str,
        top_k: int = 50,
        candidate_indices: Optional[np.ndarray] = None,
    ) -> Dict[int, float]:
        """Calculates normalized TF-IDF keyword scores."""
        q_vec = self.tfidf.transform([query])
        scores = (self.tfidf_matrix * q_vec.T).toarray().flatten()

        # If candidate indices are provided (after metadata filtering)
        if candidate_indices is not None and len(candidate_indices) > 0:
            mask = np.zeros(len(self.df), dtype=bool)
            mask[candidate_indices] = True
            scores[~mask] = 0.0

        max_score = np.max(scores) if len(scores) > 0 else 0
        if max_score > 0:
            scores = scores / max_score

        top_indices = np.argsort(scores)[::-1][:top_k]
        return {int(idx): float(scores[idx]) for idx in top_indices if scores[idx] > 0}

    def retrieve(
        self,
        query: str,
        top_k: int = settings.DEFAULT_TOP_K,
        filters: Optional[Dict[str, Any]] = None,
        target_property: Optional[Dict[str, Any]] = None,
        include_synthetic: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Executes true hybrid retrieval:
        1. Resolve the full metadata-filtered candidate set (real sales only by default).
        2. Run semantic vector cosine similarity over that candidate set.
        3. Run TF-IDF keyword scoring over the *same* full candidate set
           (not just the semantic top-k*4 subset), so keyword-only matches
           are not silently dropped before fusion.
        4. Fuse semantic and keyword scores over their union.
        5. Apply comparable-property prioritization as before.
        """
        inc_synth = include_synthetic or (filters is not None and filters.get("include_synthetic", False))

        # 1. Determine full metadata-filtered candidate indices
        if filters:
            # Re-use vector_store's mask logic by calling search with a large top_k.
            # We request ALL filtered candidates by passing len(df) as top_k.
            # vector_store.search returns [] when filters match nothing — preserve that.
            all_filtered = self.vector_store.search(
                query=query,
                top_k=len(self.df),
                filters=filters,
                include_synthetic=inc_synth,
            )
            if filters and len(all_filtered) == 0:
                logger.info("Filters matched 0 candidates; returning empty comp list.")
                return []
            filtered_indices = np.array(
                [
                    self.df[self.df["Property_ID"] == c["property_id"]].index[0]
                    for c in all_filtered
                ],
                dtype=int,
            )
        else:
            if not inc_synth and "is_synthetic" in self.df.columns:
                filtered_indices = np.where(~self.df["is_synthetic"].astype(bool))[0]
            else:
                filtered_indices = np.arange(len(self.df), dtype=int)

        if len(filtered_indices) == 0:
            return []

        # 2. Semantic scores over the filtered candidate set (top-k*4 subset of filtered)
        semantic_top_k = min(top_k * 4, len(filtered_indices))
        semantic_candidates = self.vector_store.search(
            query=query,
            top_k=semantic_top_k,
            filters=filters,
            include_synthetic=inc_synth,
        )
        semantic_scores = {
            self.df[self.df["Property_ID"] == c["property_id"]].index[0]: c["score"]
            for c in semantic_candidates
        }

        # 3. Keyword (TF-IDF) scores over the FULL filtered candidate set
        keyword_scores = self.keyword_search(
            query, top_k=len(filtered_indices), candidate_indices=filtered_indices
        )

        # 4. Fuse scores over the union of semantic and keyword matches
        candidate_pool = set(semantic_scores.keys()).union(set(keyword_scores.keys()))
        hybrid_results = []

        for idx in candidate_pool:
            sem_s = semantic_scores.get(idx, 0.0)
            sem_norm = max(0.0, min(1.0, (sem_s + 1.0) / 2.0))
            kw_s = keyword_scores.get(idx, 0.0)

            hybrid_s = (self.alpha * sem_norm) + ((1.0 - self.alpha) * kw_s)
            raw_dict = self.df.iloc[idx].to_dict()
            clean_dict = sanitize_comp_data(raw_dict)

            # 5. Multi-factor Comparable-Property Prioritization
            comparability_info = self._calculate_comparability(clean_dict, target_property or filters)

            # Combined prioritized ranking score (50% hybrid relevance + 50% physical comparability)
            prioritized_score = (0.50 * hybrid_s) + (0.50 * comparability_info["comparability_score"])

            hybrid_results.append({
                "property_id": clean_dict.get("Property_ID", f"PROP-{idx}"),
                "hybrid_score": round(float(hybrid_s), 4),
                "semantic_score": round(float(sem_norm), 4),
                "keyword_score": round(float(kw_s), 4),
                "comparability_score": round(float(comparability_info["comparability_score"]), 4),
                "prioritized_score": round(float(prioritized_score), 4),
                "comparability_details": comparability_info["details"],
                "data": clean_dict,
                "text": str(raw_dict.get("searchable_text", "")),
            })

        # Sort by prioritized score
        hybrid_results.sort(key=lambda x: x["prioritized_score"], reverse=True)
        return hybrid_results[:top_k]

    def _calculate_comparability(
        self,
        comp_row: Dict[str, Any],
        target_ref: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculates multi-dimensional comparability score."""
        if not target_ref:
            return {
                "comparability_score": 0.75,
                "details": {"note": "No target property features specified; scored purely by query relevance."},
            }

        weights = {
            "location": 0.30,
            "sqft": 0.25,
            "quality": 0.20,
            "year_built": 0.15,
            "rooms": 0.10,
        }

        # 1. Location match
        target_neigh = str(target_ref.get("neighborhood") or target_ref.get("location") or "").lower().strip()
        comp_neigh = str(comp_row.get("Neighborhood", "")).lower().strip()
        if target_neigh and target_neigh in comp_neigh:
            loc_score = 1.0
        elif not target_neigh:
            loc_score = 0.8
        else:
            loc_score = 0.35

        # 2. Living area proximity
        target_sqft = float(target_ref.get("gr_liv_area") or target_ref.get("GrLivArea") or comp_row.get("GrLivArea", 1500))
        comp_sqft = float(comp_row.get("GrLivArea", 1500))
        sqft_pct_diff = abs(comp_sqft - target_sqft) / max(target_sqft, 500.0)
        sqft_score = max(0.0, 1.0 - sqft_pct_diff)

        # 3. Overall Quality proximity
        target_qual = float(target_ref.get("overall_qual") or target_ref.get("OverallQual") or comp_row.get("OverallQual", 6))
        comp_qual = float(comp_row.get("OverallQual", 6))
        qual_diff = abs(comp_qual - target_qual)
        qual_score = max(0.0, 1.0 - (qual_diff / 9.0))

        # 4. Year Built / Age proximity
        target_year = float(target_ref.get("year_built") or target_ref.get("YearBuilt") or comp_row.get("YearBuilt", 1990))
        comp_year = float(comp_row.get("YearBuilt", 1990))
        year_diff = abs(comp_year - target_year)
        year_score = max(0.0, 1.0 - (year_diff / 50.0))

        # 5. Room / Bed / Bath proximity
        target_beds = float(target_ref.get("bedrooms") or target_ref.get("BedroomAbvGr") or comp_row.get("BedroomAbvGr", 3))
        comp_beds = float(comp_row.get("BedroomAbvGr", 3))
        bed_score = max(0.0, 1.0 - (abs(comp_beds - target_beds) / 4.0))

        target_baths = float(target_ref.get("bathrooms") or target_ref.get("FullBath") or comp_row.get("FullBath", 2))
        comp_baths = float(comp_row.get("FullBath", 2))
        bath_score = max(0.0, 1.0 - (abs(comp_baths - target_baths) / 3.0))

        room_score = (bed_score + bath_score) / 2.0

        composite_comp_score = (
            weights["location"] * loc_score
            + weights["sqft"] * sqft_score
            + weights["quality"] * qual_score
            + weights["year_built"] * year_score
            + weights["rooms"] * room_score
        )

        return {
            "comparability_score": round(float(composite_comp_score), 4),
            "details": {
                "location_similarity": round(float(loc_score), 3),
                "sqft_similarity": round(float(sqft_score), 3),
                "quality_similarity": round(float(qual_score), 3),
                "age_similarity": round(float(year_score), 3),
                "room_similarity": round(float(room_score), 3),
                "sqft_delta": round(float(comp_sqft - target_sqft), 1),
                "price_per_sqft": round(float(comp_row.get("PricePerSqFt", 0)), 2),
            },
        }

# Singleton instance
hybrid_retriever = HybridRetriever()
