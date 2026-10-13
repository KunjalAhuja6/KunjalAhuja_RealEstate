# Dataset Details & Documentation: Ames Housing Dataset & Synthetic Augmentation

**Author / Presenter**: Kunjal Ahuja  
**Domain**: Real Estate & PropTech Valuation Intelligence  
**Capstone**: Prodapt AI/ML Engineering Capstone  

---

## 1. Dataset Overview

- **Dataset Name**: Ames Housing Dataset with Gaussian Copula Synthetic Augmentation
- **Primary Source**: OpenML Data Repository (Dataset ID: 43926)
- **Primary Link**: [OpenML Ames Housing Dataset](https://www.openml.org/search?type=data&sort=runs&id=43926)
- **Original Author**: Dean De Cock (Truman State University, 2011)
- **Total Record Count**: **10,000 records**
  - **Real Historical Transactions**: **2,930 records** (IDs: `PROP-0001` through `PROP-2930`)
  - **Synthetic Augmented Transactions**: **7,070 records** (IDs: `SYN-00001` through `SYN-07070`)
- **Total Neighborhoods**: **28 distinct Ames neighborhoods**
- **Construction Year Range**: **1872 to 2010**
- **Standardized Clean File**: `data/sample_data/ames_housing_cleaned.csv`
- **Dense Embedding Matrix**: `data/sample_data/property_embeddings.npy` (10,000 x 384 matrix)

---

## 2. Dataset Purpose & Role in System

In this valuation intelligence system, the dataset serves as the foundational data source for:
1. **Semantic & Hybrid Property Retrieval**: All 10,000 properties are serialized into natural language documents for dense vector indexing and TF-IDF keyword searching.
2. **Comparable-Property Ground Truth**: Only genuine historical sales (`is_synthetic == False`) are retrieved as comparable properties for client reports, ensuring grounded market evidence.
3. **Supervised ML Price Modeling**: Training a `GradientBoostingRegressor` on the augmented training set ($n = 9,414$), validated against a held-out test set of real transactions ($n = 586$) achieving $R^2 = 0.9198$.
4. **Physical & Market Risk Scoring**: Computing neighborhood price volatility, age depreciation, and layout liquidity benchmarks.
5. **Continuous Calibration Baseline**: Providing historical price distribution anchors for the appraiser feedback loop.

---

## 3. The 15 Standardized Key Fields

The system extracts, validates, and models the following 15 key features across all real and synthetic records:

| Key Field | Data Type | Physical Description | Sample Range / Values |
|---|---|---|---|
| **SalePrice** | Numeric (USD) | Historical property transaction closing price (Target variable). | \$12,789 – \$755,000 (Median: \$160,000) |
| **OverallQual** | Integer (1–10) | Rates the overall material quality and architectural finishes. | 1 (Very Poor) to 10 (Very Excellent) |
| **OverallCond** | Integer (1–10) | Rates the current physical maintenance and condition. | 1 (Very Poor) to 10 (Very Excellent) |
| **GrLivArea** | Numeric (sq ft) | Above-grade (ground level) finished living area. | 334 – 5,642 sq ft |
| **TotalBsmtSF** | Numeric (sq ft) | Total square footage of basement area. | 0 – 6,110 sq ft |
| **GarageCars** | Integer | Size of garage in terms of car capacity. | 0 – 5 cars |
| **GarageArea** | Numeric (sq ft) | Size of garage in square feet. | 0 – 1,488 sq ft |
| **YearBuilt** | Integer (Year) | Original construction year. | 1872 – 2010 |
| **YearRemodAdd** | Integer (Year) | Remodel date (same as construction year if no remodel). | 1950 – 2010 |
| **FullBath** | Integer | Full bathrooms above grade. | 0 – 4 bathrooms |
| **BedroomAbvGr** | Integer | Number of bedrooms above basement level. | 0 – 8 bedrooms |
| **TotRmsAbvGrd** | Integer | Total rooms above ground (excluding bathrooms). | 2 – 15 rooms |
| **Neighborhood** | Categorical | Location / subdivision within Ames (28 distinct neighborhoods). | North Ames, College Creek, Stone Brook, Somerset, Edwards, etc. |
| **LotArea** | Numeric (sq ft) | Lot size in square feet. | 1,300 – 215,245 sq ft |
| **GarageYrBlt** | Integer (Year) | Year garage was constructed (imputed from YearBuilt if null). | 1895 – 2010 |

---

## 4. Synthetic Data Generation Methodology

To enrich training diversity while preserving statistical validity, 7,070 synthetic records were generated using a neighborhood-conditioned Gaussian Copula framework (`generate_synthetic_data.py`):
1. **Neighborhood Conditioning**: Separate Gaussian Copula models were fitted for each of the 28 neighborhoods, conditioning distributions on local price levels and architectural styles.
2. **Empirical Marginals**: Continuous features (square footage, lot size, prices) were mapped to empirical cumulative distribution functions (ECDF) to preserve skewness and multi-modality.
3. **Novelty Distance Filter**: Synthetic candidates were evaluated against their nearest real neighbor using Euclidean distance across standardized features. Candidates falling closer than the 25th percentile of real nearest-neighbor gaps were rejected to prevent memorization and duplicate points.
4. **Physical Integrity Constraints**: Synthetic records enforce physical relationships:
   - `YearRemodAdd >= YearBuilt`
   - `TotalBsmtSF >= 0` and `GarageArea >= GarageCars * 150`
   - Standard 1–10 integer constraints on `OverallQual` and `OverallCond`
5. **Metadata Tagging**: All synthetic records are flagged with `is_synthetic = True` and assigned IDs starting with `SYN-`.

---

## 5. Honest Train / Test Split & Comparable Protection

To uphold rigorous scientific standards and avoid synthetic data leakage:
1. **Real-Only Test Evaluation**:
   - The test set ($n = 586$) consists strictly of a 20% random sample (`random_state=42`) of the **2,930 real historical records**.
   - **Synthetic records never appear in the test set**.
2. **Training Set Composition**:
   - The training set contains 2,344 real records (80% of real) plus all 7,070 synthetic records, totaling **9,414 training rows**.
3. **Comparable Sales Isolation**:
   - In all comparable property search workflows, synthetic records are filtered out by default (`include_synthetic = False`). Real estate clients and appraisers are presented only with genuine historical sales transactions.

### Model Evaluation on Real Test Set ($n = 586$):
- **Trained Model (Real + Synthetic)**: $R^2 = 0.9198$, $\text{RMSE} = \$25,361.84$, $\text{MAE} = \$16,161.76$, $\text{MAPE} = 8.87\%$
- **Baseline Model (Real Only)**: $R^2 = 0.9031$, $\text{RMSE} = \$27,870.39$, $\text{MAE} = \$15,889.09$, $\text{MAPE} = 8.46\%$

Synthetic augmentation provided a **+0.0167 gain in $R^2$** and reduced RMSE by **\$2,508.55** on real unseen test transactions.

---

## 6. Derived Features & Searchable Representation

During data loading and preprocessing, the pipeline creates:
- **PropertyAge**: `2026 - YearBuilt`
- **EffectiveAge**: `2026 - YearRemodAdd`
- **PricePerSqFt**: `SalePrice / GrLivArea`
- **searchable_text**: Textual document summarizing location, layout, quality, and historical sales metrics (1 row = 1 document, no text chunking required).
