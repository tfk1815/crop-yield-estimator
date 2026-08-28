# 🌾 Crop Yield Estimator (Nigeria)

## Project Overview
This project predicts crop yield (**tonnes per hectare**) for Nigerian farms based on
season and farm-level inputs: state, crop type, farm size, rainfall, temperature, soil
pH/type, fertilizer application, pesticide use, irrigation, and seed variety.

It's a **regression** problem — the model outputs a continuous number, not a class label.
Intended users: extension officers, agribusinesses, cooperatives, and farmers planning
input purchases or harvest logistics ahead of a growing season.

The dataset is synthetic (15,000 rows), generated to match this project's exact schema,
with agronomic rules baked in (e.g. diminishing/plateauing returns from rainfall and
fertilizer, yield suppression at poor soil pH) plus random noise.

## Model Summary
- **Algorithm:** Random Forest Regressor, tuned via `GridSearchCV` (best params:
  `n_estimators=200`, `max_depth=15`)
- **Preprocessing:** `StandardScaler` on numeric features, `OneHotEncoder` on categorical
  features, combined via a single `ColumnTransformer` + `Pipeline` (so the exact same
  preprocessing is reused at inference time in the Streamlit app)
- **Engineered features:** `rainfall_per_ha` (rainfall normalized by farm size),
  `fertilizer_efficiency_proxy` (fertilizer rate adjusted for soil pH)

### Test-set performance
| Metric | Value |
|---|---|
| MAE  | 0.570 tonnes/ha |
| RMSE | 0.946 tonnes/ha |
| R²   | 0.928 |

A Gradient Boosting model performed marginally better in the model comparison
(R² ≈ 0.933) — see the notebook's Step 10 comparison table — but the tuned Random Forest
was carried forward as the final model.

## Project Structure
```
crop_yield_estimator/
├── data/
│   ├── raw_crop_data.csv
│   └── cleaned_crop_data.csv
├── notebooks/
│   └── crop_yield_estimator.ipynb
├── models/
│   ├── crop_yield_model.pkl
│   └── model_features.pkl
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```

## How to Run the Notebook
```bash
conda create -n crop-yield-env python=3.11
conda activate crop-yield-env
pip install -r requirements.txt jupyter
cd notebooks
jupyter notebook crop_yield_estimator.ipynb
```
Run all cells top to bottom. The notebook covers problem definition, EDA, cleaning,
feature engineering, model training/evaluation/tuning, and saves the final pipeline to
`models/crop_yield_model.pkl`.

## How to Run the App
```bash
pip install -r requirements.txt
cd app
streamlit run app.py
```
Fill in the farm/season details and click **Estimate Yield** to get a prediction in
tonnes/hectare, plus a total-tonnage estimate for the entered farm size.

## Notes / Caveats
- The model was trained on synthetic data with realistic but simplified agronomic rules —
  treat predictions as directional estimates, not agronomic guarantees.
- Inputs far outside the training data's typical ranges (e.g. extremely small farm sizes)
  can push the model into extrapolation, where predictions become less reliable.
