import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from src.utils.config import settings
from src.utils.logger import get_logger
from src.rag.data_pipeline import load_cleaned_data

logger = get_logger("VectorStore")

class PropertyVectorStore:
    """
    In-memory vector store with precomputed normalized embeddings and metadata filtering.
    Supports SentenceTransformers (local cache) and fast TFIDF-LSA fallback.
    """
    def __init__(
        self,
        cleaned_data_path: Path = settings.CLEANED_DATA_PATH,
        embedding_cache_path: Path = settings.SAMPLE_DATA_DIR / "property_embeddings.npy",
        model_name: str = settings.EMBEDDING_MODEL_NAME,
    ):
        self.cleaned_data_path = cleaned_data_path
        self.embedding_cache_path = embedding_cache_path
        self.model_name = model_name
        self.st_model = None
        self.tfidf_fallback = None
        self.df: pd.DataFrame = pd.DataFrame()
        self.embeddings: Optional[np.ndarray] = None
        self._initialize()

    def _initialize(self):
        """Loads dataset and initializes or loads cached embeddings."""
        self.df = load_cleaned_data(self.cleaned_data_path)
        logger.info(f"Loaded {len(self.df)} properties for vector indexing.")

        # Try to initialize SentenceTransformer with local_files_only
        try:
            from sentence_transformers import SentenceTransformer
            try:
                self.st_model = SentenceTransformer(self.model_name, local_files_only=True)
                logger.info(f"Loaded {self.model_name} from local HuggingFace cache.")
            except Exception:
                self.st_model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded {self.model_name}.")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Initializing TF-IDF vectorizer fallback.")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_fallback = TfidfVectorizer(max_features=384, stop_words="english")
            self.tfidf_fallback.fit(self.df["searchable_text"])

        # Check if precomputed embeddings exist
        if os.path.exists(self.embedding_cache_path):
            try:
                logger.info(f"Loading cached embeddings from {self.embedding_cache_path}...")
                self.embeddings = np.load(self.embedding_cache_path)
                if len(self.embeddings) == len(self.df):
                    logger.info("Embeddings loaded successfully from cache.")
                    return
                else:
                    logger.warning("Cache size mismatch with dataset. Recomputing embeddings...")
            except Exception as e:
                logger.warning(f"Error loading cache ({e}). Recomputing embeddings...")

        # Compute embeddings
        self._build_and_cache_embeddings()

    def _build_and_cache_embeddings(self):
        """Encodes all searchable documents and caches the matrix."""
        logger.info(f"Encoding {len(self.df)} property documents into dense vectors...")
        texts = self.df["searchable_text"].tolist()
        if self.st_model is not None:
            emb = self.st_model.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
            self.embeddings = np.array(emb, dtype=np.float32)
        else:
            sparse_mat = self.tfidf_fallback.transform(texts)
            norms = np.linalg.norm(sparse_mat.toarray(), axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings = (sparse_mat.toarray() / norms).astype(np.float32)

        self.embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.embedding_cache_path, self.embeddings)
        logger.info(f"Saved {self.embeddings.shape} embeddings to {self.embedding_cache_path}.")

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes query string into normalized dense vector."""
        if self.st_model is not None:
            q_emb = self.st_model.encode([query], normalize_embeddings=True)
            return q_emb[0]
        else:
            sparse_vec = self.tfidf_fallback.transform([query]).toarray()[0]
            norm = np.linalg.norm(sparse_vec)
            if norm > 0:
                sparse_vec = sparse_vec / norm
            return sparse_vec.astype(np.float32)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_synthetic: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic vector search with optional metadata filtering.
        Excludes synthetic rows by default unless include_synthetic is True.
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # 1. Apply metadata filtering to candidate indices
        mask = np.ones(len(self.df), dtype=bool)

        # Real sales only by default
        inc_synth = include_synthetic or (filters is not None and filters.get("include_synthetic", False))
        if not inc_synth and "is_synthetic" in self.df.columns:
            mask = mask & (~self.df["is_synthetic"].astype(bool))

        if filters:
            if "neighborhood" in filters and filters["neighborhood"]:
                neigh_target = str(filters["neighborhood"]).lower().strip()
                neigh_match = self.df["Neighborhood"].str.lower().str.contains(neigh_target, na=False)
                mask = mask & neigh_match

            if "location" in filters and filters["location"]:
                loc_target = str(filters["location"]).lower().strip()
                loc_match = self.df["Neighborhood"].str.lower().str.contains(loc_target, na=False)
                mask = mask & loc_match

            if "min_year_built" in filters and filters["min_year_built"] is not None:
                mask = mask & (self.df["YearBuilt"] >= float(filters["min_year_built"]))

            if "max_year_built" in filters and filters["max_year_built"] is not None:
                mask = mask & (self.df["YearBuilt"] <= float(filters["max_year_built"]))

            if "min_age" in filters and filters["min_age"] is not None:
                mask = mask & (self.df["PropertyAge"] >= float(filters["min_age"]))

            if "max_age" in filters and filters["max_age"] is not None:
                mask = mask & (self.df["PropertyAge"] <= float(filters["max_age"]))

            if "min_bedrooms" in filters and filters["min_bedrooms"] is not None:
                mask = mask & (self.df["BedroomAbvGr"] >= float(filters["min_bedrooms"]))

            if "max_bedrooms" in filters and filters["max_bedrooms"] is not None:
                mask = mask & (self.df["BedroomAbvGr"] <= float(filters["max_bedrooms"]))

            if "min_bathrooms" in filters and filters["min_bathrooms"] is not None:
                mask = mask & (self.df["FullBath"] >= float(filters["min_bathrooms"]))

            if "min_sqft" in filters and filters["min_sqft"] is not None:
                mask = mask & (self.df["GrLivArea"] >= float(filters["min_sqft"]))

            if "max_sqft" in filters and filters["max_sqft"] is not None:
                mask = mask & (self.df["GrLivArea"] <= float(filters["max_sqft"]))

            if "min_total_rooms" in filters and filters["min_total_rooms"] is not None:
                mask = mask & (self.df["TotRmsAbvGrd"] >= float(filters["min_total_rooms"]))

            if "max_total_rooms" in filters and filters["max_total_rooms"] is not None:
                mask = mask & (self.df["TotRmsAbvGrd"] <= float(filters["max_total_rooms"]))

            # Quality filters (OverallQual)
            min_q = filters.get("min_quality") if "min_quality" in filters else filters.get("min_overall_qual")
            if min_q is not None:
                mask = mask & (self.df["OverallQual"] >= float(min_q))

            max_q = filters.get("max_quality") if "max_quality" in filters else filters.get("max_overall_qual")
            if max_q is not None:
                mask = mask & (self.df["OverallQual"] <= float(max_q))

        candidate_indices = np.where(mask)[0]
        if len(candidate_indices) == 0:
            logger.info("Metadata filters matched 0 properties; respecting empty candidate set.")
            return []

        # 2. Compute cosine similarity against candidate subset
        q_vec = self.encode_query(query)
        candidate_embeddings = self.embeddings[candidate_indices]
        sims = np.dot(candidate_embeddings, q_vec)

        # 3. Top K selection
        top_k_local = min(top_k, len(candidate_indices))
        best_local_indices = np.argpartition(sims, -top_k_local)[-top_k_local:]
        sorted_order = np.argsort(sims[best_local_indices])[::-1]
        best_indices = candidate_indices[best_local_indices[sorted_order]]
        best_scores = sims[best_local_indices[sorted_order]]

        # 4. Format results
        results = []
        for idx, score in zip(best_indices, best_scores):
            row = self.df.iloc[idx].to_dict()
            results.append({
                "property_id": row["Property_ID"],
                "score": float(score),
                "data": row,
                "text": row["searchable_text"],
            })
        return results

# Singleton instance
vector_store = PropertyVectorStore()
