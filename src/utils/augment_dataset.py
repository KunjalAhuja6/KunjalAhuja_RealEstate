import os
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.config import settings
from src.utils.logger import get_logger
from src.rag.data_pipeline import generate_property_document

logger = get_logger("DatasetAugmenter")

def augment_dataset(target_total_records: int = 11720) -> pd.DataFrame:
    """
    Statistically augments the Ames Housing dataset to target_total_records (approx 4x scale: ~11,720 rows).
    Maintains empirical multivariate correlation:
    - Preserves neighborhood geographic boundaries and base distributions
    - Realistic physical proportional perturbations for living area, lot area, basement, garage
    - Preserves price-per-square-foot elasticity and physical constraints
    """
    cleaned_path = settings.CLEANED_DATA_PATH
    df_orig = pd.read_csv(cleaned_path)
    orig_count = len(df_orig)
    logger.info(f"Loaded original dataset with {orig_count} records.")

    np.random.seed(42)
    # Target 3 synthetic variants per original row -> 1 orig + 3 aug = 4x = 11,720 records
    augmented_dfs = [df_orig.copy()]

    for variant in range(3):
        df_aug = df_orig.copy()

        # 1. Living area: +/- 4% realistic variation (bounded [400, 6000])
        sqft_jitter = np.random.normal(1.0, 0.04, len(df_aug))
        df_aug["GrLivArea"] = (df_aug["GrLivArea"] * sqft_jitter).round().clip(400, 6000)

        # 2. Lot area: +/- 5% variation
        lot_jitter = np.random.normal(1.0, 0.05, len(df_aug))
        df_aug["LotArea"] = (df_aug["LotArea"] * lot_jitter).round().clip(1500, 70000)

        # 3. Total Basement: proportional to living area change
        df_aug["TotalBsmtSF"] = (df_aug["TotalBsmtSF"] * sqft_jitter).round().clip(0, 5000)

        # 4. Garage Area: minor realistic variation
        df_aug["GarageArea"] = (df_aug["GarageArea"] * np.random.normal(1.0, 0.03, len(df_aug))).round().clip(0, 1600)

        # 5. Sale Price: maintains neighborhood PPSF with small variance (+/- 2.5%)
        price_noise = np.random.normal(1.0, 0.025, len(df_aug))
        df_aug["SalePrice"] = (df_aug["PricePerSqFt"] * df_aug["GrLivArea"] * price_noise).round(-2).clip(30000, 800000)
        df_aug["PricePerSqFt"] = (df_aug["SalePrice"] / df_aug["GrLivArea"].replace(0, 1)).round(2)

        # Assign unique sequential Property_ID
        start_id = orig_count * (variant + 1) + 1
        df_aug["Property_ID"] = [f"PROP-{start_id + i:05d}" for i in range(len(df_aug))]

        # Re-generate rich searchable text for RAG indexing
        df_aug["searchable_text"] = df_aug.apply(generate_property_document, axis=1)

        augmented_dfs.append(df_aug)

    df_total = pd.concat(augmented_dfs, ignore_index=True)
    logger.info(f"Successfully generated expanded dataset: {len(df_total)} records across {df_total['Neighborhood'].nunique()} neighborhoods.")

    # Save to cleaned data path
    df_total.to_csv(cleaned_path, index=False)
    logger.info(f"Saved expanded dataset to {cleaned_path}.")
    return df_total

if __name__ == "__main__":
    df = augment_dataset()
    print("New dataset row count:", len(df))
