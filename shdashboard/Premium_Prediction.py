import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
import os

os.environ["TRANSFORMERS_NO_TF"] = "1"


# Load data and encoders
@st.cache_data
def load_data():
    df = pd.read_csv('data/Assure_PremiumChart.csv', encoding='utf-8')
    model = joblib.load('data/xgb_model.pkl')
    le_zone = joblib.load('data/zone_encoder.pkl')
    le_plan = joblib.load('data/plan_encoder.pkl')
    return df, model, le_zone, le_plan

data, model, le_zone, le_plan = load_data()

features = ['Zone Encoded', 'Term', 'Plan Type Encoded', 'Sum Insured', 'Age Lower', 'Age Upper']
best_mae = 12732.13
baseline_mae = 22102.16

# Age Band Utilities
def build_age_band_lookup(df):
    age_map = []
    for band in df['Age Band'].unique():
        band_clean = band.strip().lower()
        try:
            if 'above' in band_clean:
                age_map.append((band, 80.0, 100.0))
            elif 'days' in band_clean:
                parts = band_clean.replace('days', '').split('-')
                low = float(parts[0]) / 365
                high = float(parts[1].replace('yrs', ''))
                age_map.append((band, low, high))
            elif '-' in band_clean:
                parts = band_clean.replace('yrs', '').split('-')
                low = float(parts[0])
                high = float(parts[1])
                age_map.append((band, low, high))
            else:
                val = float(band_clean.replace('yrs', ''))
                age_map.append((band, val, val))
        except:
            continue
    return age_map

def match_age_to_band(user_age, known_bands):
    for band, low, high in known_bands:
        if low <= user_age <= high:
            return band.title(), low, high
    return None, None, None

def prepare_new_sample(sample_df, le_zone, le_plan):
    def split_age_band(age_band):
        if 'days' in age_band.lower():
            parts = age_band.lower().replace('days', '').replace('yrs', '').split('-')
            low = float(parts[0]) / 365
            high = float(parts[1])
            return low, high
        elif '-' in age_band:
            parts = age_band.replace('yrs', '').split('-')
            return float(parts[0]), float(parts[1])
        elif 'above' in age_band.lower():
            return 80.0, 100.0
        else:
            val = float(age_band.replace('yrs', ''))
            return val, val

    sample_df['Zone Encoded'] = le_zone.transform(sample_df['Zone'])
    sample_df['Plan Type Encoded'] = le_plan.transform(sample_df['Plan Type'])
    sample_df['Term'] = sample_df['Term'].astype(int)
    sample_df['Sum Insured'] = sample_df['Sum Insured'].astype(int)
    sample_df[['Age Lower', 'Age Upper']] = sample_df['Age Band'].apply(lambda x: pd.Series(split_age_band(x)))
    return sample_df[['Zone Encoded', 'Term', 'Plan Type Encoded', 'Sum Insured', 'Age Lower', 'Age Upper']]

known_bands = build_age_band_lookup(data)

@st.cache_resource
def load_model():
    return joblib.load("models/xgb_model.pkl")

@st.cache_resource
def load_explainer(model):
    return shap.TreeExplainer(model)

model = load_model()
explainer = load_explainer(model)

@st.cache_data
def get_sample_shap(explainer, sample_df):
    X = sample_df[features]
    return explainer.shap_values(X)

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Premium Prediction",
        options=["🏠 Home", "📊 Visuals", "💡 Predict Premium"],
        icons=["house", "bar-chart", "activity"],
        default_index=0
    )

if selected == "🏠 Home":
    st.title("💸 Health Insurance Premium Predictor")

    st.markdown("""
    This module predicts health insurance premiums based on:
    - ✅ **Zone**
    - ✅ **Plan Type**
    - ✅ **Term Duration**
    - ✅ **Sum Insured**
    - ✅ **Age Band**

    The model is trained using real brochure-based premium charts and outputs SHAP-based explanations for transparency.
    """)

    st.markdown("---")
    st.subheader("📊 Model Performance Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Best MAE (XGBoost)", f"₹ {best_mae:,.2f}")
    col2.metric("Baseline MAE (Random Forest)", f"₹ {baseline_mae:,.2f}")
    col3.metric("Selected Model", "✅ XGBoost (Tuned)")

    st.markdown("---")
    st.subheader("📝 Module Overview")
    st.markdown("""
    - Trained on pre-defined **age bands** from insurer brochures
    - Handles zone, plan, and sum insured as categorical/numeric inputs
    - Uses **SHAP** to explain each prediction's drivers
    - Run live inside this internship dashboard
    """)


elif selected == "📊 Visuals":
    st.title("📊 Feature Insights and SHAP Analysis")

    st.subheader("📌 XGBoost Feature Importance")
    st.markdown("This bar chart shows which features contribute most to premium prediction.")

    # --- Feature Importance Bar Plot ---
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance')
    fig1, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='Blues_d', ax=ax)
    ax.set_title("XGBoost Feature Importances")
    st.pyplot(fig1)

    st.markdown("---")
    st.subheader("🔍 SHAP Summary Plot")
    st.markdown("This SHAP plot shows how each feature influences predictions across 500 samples.")

    # Sample and preprocess
    sample_df = data.sample(n=500, random_state=42).copy()

    sample_df['Zone Encoded'] = le_zone.transform(sample_df['Zone'])
    sample_df['Plan Type Encoded'] = le_plan.transform(sample_df['Plan Type'])
    sample_df['Term'] = sample_df['Term'].astype(str).str.extract(r'(\d+)').astype(int)
    sample_df['Sum Insured'] = sample_df['Sum Insured'].astype(int)

    def split_age_band(band):
        band = band.lower().replace('yrs', '').replace('year', '')
        if 'above' in band:
            return 80.0, 100.0
        elif 'days' in band:
            parts = band.replace('days', '').split('-')
            return float(parts[0])/365, float(parts[1])
        elif '-' in band:
            parts = band.split('-')
            return float(parts[0]), float(parts[1])
        else:
            val = float(band)
            return val, val

    sample_df[['Age Lower', 'Age Upper']] = sample_df['Age Band'].apply(lambda x: pd.Series(split_age_band(x)))

    # SHAP Explanation
    X_sample = sample_df[features]
    shap_values = get_sample_shap(explainer, sample_df)

    # Clear plot before rendering
    plt.clf()
    
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    fig2 = plt.gcf()
    st.pyplot(fig2)
    
    # Clear after render
    plt.clf()


elif selected == "💡 Predict Premium":
    st.title("💡 Predict Your Premium Instantly")
    st.markdown("""Fill in the inputs below to estimate your premium.
    The system uses your age to map to the closest **brochure-based age band**.
    """)
    col1, col2 = st.columns(2)
    zone = col1.selectbox("Zone", le_zone.classes_)
    plan = col2.selectbox("Plan Type", le_plan.classes_)
    term = col1.selectbox("Term (Years)", ['1', '2', '3'])
    sum_insured = col2.number_input("Sum Insured", value=500000, step=50000)
    user_age = st.number_input("Enter your age", min_value=0, max_value=120, value=36)

    if st.button("🔮 Predict"):
        matched_band, age_low, age_high = match_age_to_band(user_age, known_bands)

        if matched_band:
            st.markdown(f"✅ **Matched Brochure Band:** `{matched_band}`")
            st.markdown(f"📌 **Using Age Range:** `{age_low} – {age_high}`")

            input_df = pd.DataFrame({
                'Zone Encoded': le_zone.transform([zone]),
                'Term': [int(term)],
                'Plan Type Encoded': le_plan.transform([plan]),
                'Sum Insured': [int(sum_insured)],
                'Age Lower': [age_low],
                'Age Upper': [age_high]
            })

            prediction = model.predict(input_df)[0]
            st.success(f"💰 **Predicted Premium:** ₹{prediction:,.2f}")

            # 🔍 SHAP Explanation
            st.subheader("📌 SHAP Explanation")

            shap_values = explainer.shap_values(input_df)
            base_value = shap_values.base_values[0]
            predicted_value = shap_values[0].values.sum() + base_value

            st.markdown(f"""
            <b>🧠 Base Premium (Expected Value):</b> ₹{base_value:,.0f}<br>
            <b>📈 Final Predicted Premium (after feature effects):</b> ₹{predicted_value:,.0f}
            """, unsafe_allow_html=True)

            fig, ax = plt.subplots()
            shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=input_df.iloc[0]), show=False)
            
            # Remove stray labels like "= xxx"
            for txt in fig.axes[0].texts:
                if "=" in txt.get_text():
                    txt.set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()  # Important cleanup
            st.caption("🔎 The SHAP plot shows how each input feature nudges the premium away from the average.")

        else:
            st.error("❌ This age is not covered in the brochure. Please try a different age.")


st.caption("📁 This premium estimator retrains and runs live inside the internship dashboard.")
