# Dataset Details & Documentation: Ames Housing Dataset

## 1. Dataset Overview
- **Dataset Name**: Ames Housing Dataset
- **Primary Source**: OpenML Data Repository (Dataset ID: 43926)
- **Primary Link**: [OpenML Ames Housing Dataset](https://www.openml.org/search?type=data&sort=runs&id=43926)
- **Original Author**: Dean De Cock (Truman State University, 2011)
- **Data Format**: CSV (Comma-Separated Values)
- **Total Record Count**: 2,930 historical property sales
- **Original Feature Count**: 81 explanatory variables
- **Standardized Cleaned File**: `data/sample_data/ames_housing_cleaned.csv`

---

## 2. Dataset Purpose
The Ames Housing Dataset provides detailed physical, geographical, and sales records of residential real estate transactions in Ames, Iowa from 2006 to 2010. 

In this system, the dataset serves as the empirical ground truth for:
1. **Semantic Property-Case Retrieval**: Converting records into textual representations for dense vector indexing.
2. **Hybrid Search**: Allowing lexical and semantic queries across property attributes.
3. **Comparable-Property Identification**: Calculating multi-dimensional distance metrics between subject properties and historical comps.
4. **Supervised Price Prediction**: Training Gradient Boosting and Random Forest regressors ($R^2 = 0.9031$).
5. **Property Value Risk Analysis**: Computing neighborhood volatility and effective physical age depreciation.
6. **Explainable Valuation Recommendations**: Deriving price-per-square-foot benchmarks and quality adjustment factors.

---

## 3. The 15 Key Dataset Fields

As specified in the project requirements, the system standardizes and utilizes the following 15 key features:

| Key Field | Data Type | Physical Description | Sample Range / Values |
|---|---|---|---|
| **SalePrice** | Numeric (USD) | Historical property transaction closing price (Target variable). | \$12,789 – \$755,000 (Median: \$160,000) |
| **OverallQual** | Integer (1–10) | Rates the overall material quality and architectural finishes of the house. | 1 (Very Poor) to 10 (Very Excellent) |
| **OverallCond** | Integer (1–10) | Rates the current physical maintenance and overall condition of the property. | 1 (Very Poor) to 10 (Very Excellent) |
| **GrLivArea** | Numeric (sq ft) | Above-grade (ground level) finished living area. | 334 – 5,642 sq ft |
| **TotalBsmtSF** | Numeric (sq ft) | Total square footage of basement area. | 0 – 6,110 sq ft |
| **GarageCars** | Integer | Size of garage in terms of car capacity. | 0 – 5 cars |
| **GarageArea** | Numeric (sq ft) | Size of garage in square feet. | 0 – 1,488 sq ft |
| **YearBuilt** | Integer (Year) | Original construction year. | 1872 – 2010 |
| **YearRemodAdd** | Integer (Year) | Remodel date (same as construction year if no remodeling). | 1950 – 2010 |
| **FullBath** | Integer | Full bathrooms above ground. | 0 – 4 bathrooms |
| **BedroomAbvGr** | Integer | Number of bedrooms above basement level. | 0 – 8 bedrooms |
| **TotRmsAbvGrd** | Integer | Total rooms above ground (excluding bathrooms). | 2 – 15 rooms |
| **Neighborhood** | Categorical | Physical location / subdivision within Ames (28 distinct neighborhoods). | North Ames, College Creek, Stone Brook, Somerset, Edwards, etc. |
| **LotArea** | Numeric (sq ft) | Lot size in square feet. | 1,300 – 215,245 sq ft |
| **GarageYrBlt** | Integer (Year) | Year garage was constructed (imputed from YearBuilt if missing). | 1895 – 2010 |

---

## 4. Derived Valuation Features
During preprocessing, the system computes the following derived analytical metrics:
- **PropertyAge**: `2026 - YearBuilt` (Physical structural age)
- **EffectiveAge**: `2026 - YearRemodAdd` (Effective mechanical age since modernization)
- **PricePerSqFt**: `SalePrice / GrLivArea` (Normalized unit cost benchmark)
- **searchable_text**: Formatted natural language document summarizing location, layout, quality, and price.

---

## 5. Data Cleaning & Preprocessing Performed
1. **Schema Standardization**: Mapped raw underscore column names (`Sale_Price`, `Overall_Qual`, `Gr_Liv_Area`) to canonical camelCase/snake_case aliases.
2. **Quality & Condition Decoding**: Converted OpenML ordinal categorical strings (`Very_Poor`, `Poor`, `Fair`, `Below_Average`, `Average`, `Above_Average`, `Good`, `Very_Good`, `Excellent`, `Very_Excellent`) to standard numerical ratings (1 through 10).
3. **Missing Value Imputation**: Imputed missing `GarageYrBlt` from `YearBuilt`, replaced null basement and garage square footage with 0.
4. **Text Cleaning**: Replaced underscore characters in neighborhood names with standard spaces (`North_Ames` $\rightarrow$ `North Ames`).
5. **Deduplication & Indexing**: Created synthetic unique property keys (`PROP-0001` through `PROP-2930`).
6. **Artifact Persistence**: Exported clean dataset to `data/sample_data/ames_housing_cleaned.csv` and dense embeddings to `data/sample_data/property_embeddings.npy`.
