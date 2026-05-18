# Melbourne-Housing-Price-Prediction-
## Overview
An end-to-end machine learning project predicting Melbourne residential 
property prices using the Melbourne Housing Snapshot dataset. The project 
covers exploratory data analysis, feature engineering, model comparison, 
and hyperparameter tuning achieving an R² of 0.888.

## Dataset
- **Source:** Melbourne Housing Snapshot (Kaggle)
- **Size:** 13,580 properties
- **Target:** Property Price (AUD)
- **Features:** Location, property type, rooms, building area, land size, 
  year built, distance from CBD, selling method

  ## Project Structure
Data Loading
Exploratory Data Analysis
Data Quality Checks
Data Cleaning & Imputation
Feature Engineering
EDA Visualizations
Train/Test Split & Preprocessing
Model Training & Comparison
Hyperparameter Tuning (2 rounds)
Model Analysis & Visualization

---

## Key Findings
- **Southern Metropolitan** is Melbourne's most expensive region (~$1.25M median)
- **Distance from CBD** is the strongest continuous predictor — every km further 
  reduces expected price significantly
- **Property type** creates a structural price gap — houses command ~3x more 
  than units on average
- **Heritage properties** (pre-1920) command premiums despite their age due to 
  character and location

---

## Feature Engineering
- `build_ratio` — land size relative to building area
- `Distance_zone` — categorical CBD proximity zones
- `is_close_cbd` — binary flag for properties within 10km
- `era` — construction era categories (heritage → new)
- `Age` — property age in years
- `Year` / `Month` — extracted from sale date

---

## Models Compared

| Model | R² |
|---|---|
| CatBoostRegressor | 0.884 |
| RandomForestRegressor | 0.858 |
| GradientBoostingRegressor | 0.830 |
| LinearRegression | 0.791 |
| Ridge | 0.791 |
| Lasso | ~0.00 |
| KNeighborsRegressor | 0.564 |

---

## Results After Tuning

| | R² | RMSE |
|---|---|---|
| Base CatBoost | 0.884 | 0.182 |
| After Tune 1 | 0.886 | 0.180 |
| **After Tune 2** | **0.888** | **0.178** |

**Best params:** depth=7, iterations=1200, learning_rate=0.12, l2_leaf_reg=2

---

## Top Features by Importance
1. Property Type (unit)
2. Southern Metropolitan region
3. Distance from CBD
4. Latitude / Longitude
5. Number of Rooms
6. is_close_cbd ← engineered feature
7. build_ratio ← engineered feature

---

## Technical Stack
- Python, Pandas, NumPy
- Scikit-learn (Pipeline, ColumnTransformer, GridSearchCV)
- CatBoost, XGBoost, LightGBM
- Matplotlib, Seaborn

---

## Notebook
[View on Kaggle](https://www.kaggle.com/code/abdallahhashad0/melbourne-housing-price-prediction-eda-ml-mode)

---

## Author
**Abdallah Hashad**  
Self-taught ML practitioner building toward a junior data scientist role.
