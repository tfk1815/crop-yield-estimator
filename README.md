# 🌾 Crop Yield Estimator (Nigeria)

## Project Overview
This project predicts crop yield (**tonnes per hectare**) for Nigerian farms based on season and farm-level inputs: state, crop type, farm size, rainfall, temperature, soil pH/type, fertilizer application, pesticide use, irrigation, and seed variety.

To prevent data leakage and bias toward naturally heavier crops (e.g., Cassava vs. Rice), the core machine learning model does not predict raw weight. Instead, it predicts a **Yield Performance Index** measuring how well a farm is expected to perform relative to the historical baseline average for that specific crop. The interactive web application then seamlessly converts this index back into actionable tonnes/hectare for the end user.

Intended users include extension officers, agribusinesses, cooperatives, and farmers planning input purchases or harvest logistics ahead of a growing season.

The dataset is synthetic (15,000 rows), generated to match this project's exact schema, with agronomic rules baked in (e.g., diminishing/plateauing returns from rainfall and fertilizer, yield suppression at poor soil pH) plus random noise.

## Model Summary
- **Algorithm:** Gradient Boosting Regressor
- **Target Variable:** `yield_performance_index` (Actual yield divided by baseline expected yield)
- **Preprocessing:** `StandardScaler` on numeric features and `OneHotEncoder` on categorical features, combined via a single `ColumnTransformer` + `Pipeline`. This exact pipeline is serialized and reused at inference time in the Streamlit app.
- **Engineered Features:** 
  - `rainfall_per_ha`: Rainfall normalized by farm size.
  - `fertilizer_efficiency_proxy`: Fertilizer application rate mathematically adjusted for soil acidity (pH).
  - `baseline_expected_yield`: The historical average yield for the specified crop type.

### Test-Set Performance (Predicting Performance Index)
| Metric | Value |
|---|---|
| **MAE**  | 0.113 |
| **RMSE** | 0.160 |
| **R²**   | 0.473 |

*The model successfully captures the environmental and operational variance (R² ≈ 0.47) without over-fitting to the naturally heavy baselines of certain crop types.*

## Project Structure
```text
crop-yield-estimator/
├── README.md
├── app.py
├── cleaned_crop_data.csv
├── crop_yield_estimator.ipynb
├── crop_yield_model.pkl
├── model_features.pkl
├── raw_crop_data.csv
└── requirements.txt
