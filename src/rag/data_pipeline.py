import os
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger("DataPipeline")

KEY_FIELDS = [
    "SalePrice",
    "OverallQual",
    "OverallCond",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "GarageArea",
    "YearBuilt",
    "YearRemodAdd",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "Neighborhood",
    "LotArea",
    "GarageYrBlt",
]

COLUMN_MAPPING = {
    "Sale_Price": "SalePrice",
    "Overall_Qual": "OverallQual",
    "Overall_Cond": "OverallCond",
    "Gr_Liv_Area": "GrLivArea",
    "Total_Bsmt_SF": "TotalBsmtSF",
    "Garage_Cars": "GarageCars",
    "Garage_Area": "GarageArea",
    "Year_Built": "YearBuilt",
    "Year_Remod_Add": "YearRemodAdd",
    "Full_Bath": "FullBath",
    "Bedroom_AbvGr": "BedroomAbvGr",
    "TotRms_AbvGrd": "TotRmsAbvGrd",
    "Neighborhood": "Neighborhood",
    "Lot_Area": "LotArea",
    "Garage_Yr_Blt": "GarageYrBlt",
}

QUALITY_MAP = {
    "very_poor": 1,
    "poor": 2,
    "fair": 3,
    "below_average": 4,
    "average": 5,
    "above_average": 6,
    "good": 7,
    "very_good": 8,
    "excellent": 9,
    "very_excellent": 10,
}

def clean_and_preprocess_dataset(
    raw_path: Path = settings.RAW_DATA_PATH,
    output_path: Path = settings.CLEANED_DATA_PATH,
) -> pd.DataFrame:
    """
    Cleans raw Ames Housing dataset, standardizes columns to key fields,
    converts ordinal quality scores, computes derived metrics, and formats searchable text representations.
    """
    logger.info(f"Loading raw dataset from {raw_path}...")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset file not found at {raw_path}")

    df = pd.read_csv(raw_path)

    # Standardize column names
    rename_dict = {}
    for raw_col in df.columns:
        if raw_col in COLUMN_MAPPING:
            rename_dict[raw_col] = COLUMN_MAPPING[raw_col]
        elif raw_col.replace("_", "").lower() in [k.lower() for k in KEY_FIELDS]:
            for k in KEY_FIELDS:
                if raw_col.replace("_", "").lower() == k.lower():
                    rename_dict[raw_col] = k
                    break

    df = df.rename(columns=rename_dict)

    # Handle GarageYrBlt if missing
    if "GarageYrBlt" not in df.columns:
        if "YearBuilt" in df.columns:
            df["GarageYrBlt"] = df["YearBuilt"]
        else:
            df["GarageYrBlt"] = 1980

    # Ensure all key fields are present
    missing_keys = [k for k in KEY_FIELDS if k not in df.columns]
    if missing_keys:
        logger.warning(f"Fields missing from dataset: {missing_keys}. Imputing defaults.")
        for k in missing_keys:
            df[k] = 0

    # Map OverallQual and OverallCond if strings
    for qual_col in ["OverallQual", "OverallCond"]:
        if qual_col in df.columns:
            # If already numeric
            if pd.api.types.is_numeric_dtype(df[qual_col]):
                df[qual_col] = df[qual_col].fillna(5).astype(int)
            else:
                df[qual_col] = df[qual_col].astype(str).str.lower().map(QUALITY_MAP).fillna(5).astype(int)

    # Impute missing values in numeric fields
    numeric_fields = [
        "SalePrice", "OverallQual", "OverallCond", "GrLivArea",
        "TotalBsmtSF", "GarageCars", "GarageArea", "YearBuilt",
        "YearRemodAdd", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd",
        "LotArea", "GarageYrBlt"
    ]
    for col in numeric_fields:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Standardize categorical neighborhood names (replace underscores with spaces)
    df["Neighborhood"] = df["Neighborhood"].astype(str).str.replace("_", " ").str.strip()

    # Property ID
    if "Property_ID" not in df.columns:
        df.insert(0, "Property_ID", [f"PROP-{i+1:04d}" for i in range(len(df))])

    # Compute derived valuation features
    current_year = 2026
    df["PropertyAge"] = current_year - df["YearBuilt"]
    df["EffectiveAge"] = current_year - df["YearRemodAdd"]
    df["PricePerSqFt"] = (df["SalePrice"] / df["GrLivArea"].replace(0, 1)).round(2)

    # Construct searchable textual representations (chunks)
    df["searchable_text"] = df.apply(generate_property_document, axis=1)

    # Save cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned dataset saved to {output_path} with {len(df)} records.")
    return df

def generate_property_document(row: pd.Series) -> str:
    """Creates a comprehensive semantic text document for vector & keyword indexing."""
    text = (
        f"Property {row['Property_ID']} located in {row['Neighborhood']} neighborhood. "
        f"Living area: {int(row['GrLivArea'])} sq ft, Lot area: {int(row['LotArea'])} sq ft. "
        f"Accommodations: {int(row['BedroomAbvGr'])} bedrooms, {int(row['FullBath'])} full bathrooms, "
        f"{int(row['TotRmsAbvGrd'])} total rooms above ground. "
        f"Basement: {int(row['TotalBsmtSF'])} sq ft. "
        f"Garage: {int(row['GarageCars'])} cars ({int(row['GarageArea'])} sq ft). "
        f"Built in {int(row['YearBuilt'])}, remodeled in {int(row['YearRemodAdd'])}. "
        f"Overall quality rating: {int(row['OverallQual'])}/10, overall condition rating: {int(row['OverallCond'])}/10. "
        f"Historical sale price: ${int(row['SalePrice']):,} (${row['PricePerSqFt']:.2f}/sq ft)."
    )
    return text

def load_cleaned_data(cleaned_path: Path = settings.CLEANED_DATA_PATH) -> pd.DataFrame:
    """Loads cleaned data or runs clean_and_preprocess_dataset if not available."""
    if not os.path.exists(cleaned_path):
        return clean_and_preprocess_dataset()
    return pd.read_csv(cleaned_path)

if __name__ == "__main__":
    df = clean_and_preprocess_dataset()
    print("Preprocessed rows:", len(df))
    print("Sample searchable text:\n", df["searchable_text"].iloc[0])
