import streamlit as st
import pandas as pd
import joblib
import time

# --- PAGE CONFIGURATION & WIDE LAYOUT ---
st.set_page_config(page_title="Crop Yield Estimator", page_icon="🌾", layout="wide")

# --- CUSTOM CSS FOR BACKGROUND, STYLING & POPUP ANIMATION ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f4f9f4; /* Light earthy green background */
    }
    h1 {
        color: #2e7d32; /* Dark green title */
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    @keyframes popIn {
        0%   { transform: scale(0.4); opacity: 0; }
        60%  { transform: scale(1.08); opacity: 1; }
        100% { transform: scale(1); opacity: 1; }
    }
    .yield-popup {
        animation: popIn 0.55s ease-out;
        border-radius: 16px;
        padding: 1.5rem 1.2rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        margin-bottom: 0.75rem;
    }
    .yield-popup h2 {
        margin: 0 0 0.3rem 0;
        font-size: 1.9rem;
    }
    .yield-popup p {
        margin: 0;
        font-size: 1.05rem;
    }
    .pop-great   { background: linear-gradient(135deg, #d4f8d4, #a8e6a8); border: 2px solid #2e7d32; }
    .pop-solid   { background: linear-gradient(135deg, #eaf4ff, #cfe8ff); border: 2px solid #1565c0; }
    .pop-low     { background: linear-gradient(135deg, #fff3e0, #ffe0b2); border: 2px solid #e65100; }
    </style>
    """, unsafe_allow_html=True)

# Cache the model load so it doesn't slow down the app on every click
@st.cache_resource
def load_models():
    model = joblib.load("crop_yield_model.pkl")
    feature_names = joblib.load("model_features.pkl")
    return model, feature_names

model, feature_names = load_models()

AVG_YIELD_MAP = {
    "Cassava": 9.48, "Yam": 8.20, "Rice": 3.13, "Maize": 2.26, "Sorghum": 1.34
}

# --- CROP IMAGES ---
# Put one image per crop in an "assets" folder next to this script, e.g.:
#   assets/maize.jpg, assets/cassava.jpg, assets/rice.jpg, assets/sorghum.jpg, assets/yam.jpg
# Any format st.image supports (jpg/png/webp) works. Emoji fallback below covers
# the case where a file is missing so the app never crashes.
CROP_IMAGES = {
    "Maize": "assets/maize.jpg",
    "Cassava": "assets/cassava.jpg",
    "Rice": "assets/rice.jpg",
    "Sorghum": "assets/sorghum.jpg",
    "Yam": "assets/yam.jpg",
}
CROP_EMOJI = {
    "Maize": "🌽", "Cassava": "🥔", "Rice": "🍚", "Sorghum": "🌾", "Yam": "🍠"
}

def show_crop_image(crop_type: str):
    """Displays the crop's image if the file exists, otherwise a large emoji fallback."""
    import os
    path = CROP_IMAGES.get(crop_type)
    if path and os.path.exists(path):
        st.image(path, caption=crop_type, use_container_width=True)
    else:
        st.markdown(
            f"<div style='text-align:center; font-size:5rem;'>{CROP_EMOJI.get(crop_type, '🌱')}"
            f"<p style='font-size:1rem; color:#2e7d32;'>{crop_type}</p></div>",
            unsafe_allow_html=True,
        )

st.markdown("<h1>🌾 Crop Yield Estimator</h1>", unsafe_allow_html=True)
st.write("### Enter farm and season details below to estimate expected crop yield.")
st.divider()

# --- HORIZONTAL DATA ENTRY (grouped rows, side-by-side fields, no vertical scrolling) ---
st.subheader("📍 Location & Crop")
r1c1, r1c2, r1c3, r1c4 = st.columns(4)
with r1c1:
    state = st.selectbox("State", ["Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara", "Fct"])
with r1c2:
    crop_type = st.selectbox("Crop Type", ["Maize", "Cassava", "Rice", "Sorghum", "Yam"])
with r1c3:
    seed_variety = st.selectbox("Seed Variety", ["Local", "Improved"])
with r1c4:
    farm_size_ha = st.number_input("Farm Size (hectares)", min_value=0.1, value=2.0)

st.subheader("🌦️ Environment")
r2c1, r2c2, r2c3, r2c4 = st.columns(4)
with r2c1:
    avg_rainfall_mm = st.number_input("Avg Rainfall (mm)", min_value=0.0, value=150.0)
with r2c2:
    avg_temperature_c = st.number_input("Avg Temperature (°C)", min_value=0.0, max_value=50.0, value=28.0)
with r2c3:
    soil_type = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy"])
with r2c4:
    soil_ph = st.slider("Soil pH", 3.0, 9.0, 6.5)

st.subheader("🚜 Farm Practices")
r3c1, r3c2, r3c3 = st.columns(3)
with r3c1:
    fertilizer_kg_per_ha = st.number_input("Fertilizer (kg/ha)", min_value=0.0, value=100.0)
with r3c2:
    pesticide_used = st.selectbox("Pesticide Used?", ["Yes", "No"])
with r3c3:
    irrigation_used = st.selectbox("Irrigation Used?", ["Yes", "No"])

st.divider()

# --- Helper: rank inputs by the model's feature importance ---
def get_ranked_inputs(input_row: dict, model, feature_names):
    """Returns list of (feature, value, importance) sorted by importance desc."""
    regressor = model.named_steps['regressor']
    preprocessor = model.named_steps['preprocessor']
    raw_importances = regressor.feature_importances_
    transformed_names = preprocessor.get_feature_names_out()
    imp_map_raw = dict(zip(transformed_names, raw_importances))

    importance_map = {}
    for feat in feature_names:
        matched = [v for k, v in imp_map_raw.items() if k.split('__')[-1].startswith(feat)]
        importance_map[feat] = sum(matched)

    ranked = []
    for feat in feature_names:
        if feat in input_row:
            ranked.append((feat, input_row[feat], importance_map.get(feat, 0.0)))
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked

FEATURE_LABELS = {
    "state": "State",
    "crop_type": "Crop Type",
    "farm_size_ha": "Farm Size (ha)",
    "avg_rainfall_mm": "Avg Rainfall (mm)",
    "avg_temperature_c": "Avg Temperature (°C)",
    "soil_ph": "Soil pH",
    "soil_type": "Soil Type",
    "fertilizer_kg_per_ha": "Fertilizer (kg/ha)",
    "pesticide_used": "Pesticide Used",
    "irrigation_used": "Irrigation Used",
    "seed_variety": "Seed Variety",
    "rainfall_per_ha": "Rainfall per Hectare",
    "fertilizer_efficiency_proxy": "Fertilizer Efficiency Proxy",
}

# --- PREDICTION & ANIMATION LOGIC ---
if st.button("🚀 Estimate Yield", use_container_width=True):
    rainfall_per_ha = avg_rainfall_mm / farm_size_ha
    fertilizer_efficiency_proxy = fertilizer_kg_per_ha / (soil_ph + 1)

    input_row = {
        "state": state, "crop_type": crop_type, "farm_size_ha": farm_size_ha,
        "avg_rainfall_mm": avg_rainfall_mm, "avg_temperature_c": avg_temperature_c,
        "soil_ph": soil_ph, "soil_type": soil_type, "fertilizer_kg_per_ha": fertilizer_kg_per_ha,
        "pesticide_used": pesticide_used, "irrigation_used": irrigation_used,
        "seed_variety": seed_variety, "rainfall_per_ha": rainfall_per_ha,
        "fertilizer_efficiency_proxy": fertilizer_efficiency_proxy
    }

    input_df = pd.DataFrame([input_row])
    # Ensure column order matches training
    input_df = input_df[feature_names]

    with st.spinner("Crunching the numbers..."):
        time.sleep(0.4)  # tiny pause so the popup feels intentional, not instant
        predicted_index = model.predict(input_df)[0]

    baseline_yield = AVG_YIELD_MAP[crop_type]
    final_yield_per_ha = predicted_index * baseline_yield
    total_yield = final_yield_per_ha * farm_size_ha

    # --- 1. Pop-up animation, styled by performance tier ---
    if predicted_index >= 1.05:
        st.balloons()
        tier_class, tier_label = "pop-great", "🔥 Fantastic performance!"
    elif predicted_index <= 0.95:
        st.snow()
        tier_class, tier_label = "pop-low", "📉 Needs improvement"
    else:
        tier_class, tier_label = "pop-solid", "💡 Solid, on-track performance"

    st.markdown(f"""
        <div class="yield-popup {tier_class}">
            <h2>🎉 Estimated Yield: {final_yield_per_ha:.2f} t/ha</h2>
            <p>{tier_label} — performing at {predicted_index * 100:.1f}% of the average {crop_type} farm</p>
            <p>For a {farm_size_ha:.1f} ha farm, that's roughly <b>{total_yield:.2f} tonnes total</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- 3. Full summary as captions, most important features first ---
    st.subheader("📋 Prediction Summary")
    st.caption(f"**Predicted yield:** {final_yield_per_ha:.2f} t/ha  |  **Total for farm:** {total_yield:.2f} t  |  **Performance index:** {predicted_index:.3f}")

    ranked_inputs = get_ranked_inputs(input_row, model, feature_names)
    for feat, value, importance in ranked_inputs:
        label = FEATURE_LABELS.get(feat, feat)
        if isinstance(value, float):
            value_str = f"{value:.2f}"
        else:
            value_str = str(value)
        st.caption(f"**{label}** (importance {importance:.3f}%): {value_str}")

st.divider()
st.caption("**Model:** Gradient Boosting Regressor | **Test-set performance:** MAE ≈ 0.115, RMSE ≈ 0.159, R² ≈ 0.47")
