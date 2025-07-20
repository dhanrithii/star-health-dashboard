import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Main Navigation",
        options=["🏠 Home", "📊 Product Analysis", "💸 Premium Prediction", "📚 RAG Q&A", "🕵 Fraud Detection"],
        default_index=0
    )

# Pages
if selected == "🏠 Home":
    st.markdown("""
        <style>
        .home-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.5rem;
        }
        .home-subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #888;
            margin-bottom: 2rem;
        }
        .card {
            background-color: #ffffff;
            border: 1px solid #e6e6e6;
            padding: 1.2rem 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.2rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        }
        .card h4 {
            margin-top: 0;
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
            color: #333;
        }
        .card p {
            font-size: 0.95rem;
            color: #555;
            margin: 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='home-title'>🌐 Internship Project Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='home-subtitle'>Explore all internship modules developed at <b>Star Health </b></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""<div class='card'>
            <h4>📊 Product Analysis</h4>
            <p>Compare health plans using visualizations like radar charts, pie charts, and stacked bars. Understand plan differences and family-friendliness.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='card'>
            <h4>💸 Premium Prediction</h4>
            <p>Predict premium costs based on features like age, sum insured, and city tier using a trained regression model.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='card'>
            <h4>🕵 Fraud Detection</h4>
            <p>Unsupervised anomaly detection on FHIR-based claims data using Longformer embeddings and GNNs. Visualizations include UMAP and graph structures.</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class='card'>
            <h4>📚 RAG Q&A</h4>
            <p>Upload any brochure PDF and ask questions about its contents. Uses embedding similarity + Gemini Pro for answers.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class='card'>
            <h4>🧰 Technologies Used</h4>
            <p>Python · Streamlit · Sklearn · Transformers · DGL · Google Gemini · Matplotlib · Pandas</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='margin-top:2rem;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.9rem; color: #888;'>Made by Dhanrithii D · Data Science Intern 2025 · Star Health and Allied Insurance Co Ltd</p>", unsafe_allow_html=True)


elif selected == "📊 Product Analysis":
    exec(open("shdashboard/Product_Analysis.py", encoding="utf-8").read())

elif selected == "💸 Premium Prediction":
    exec(open("shdashboard/Premium_Prediction.py", encoding="utf-8").read())

elif selected == "📚 RAG Q&A":
    exec(open("shdashboard/Brochure_QA.py", encoding="utf-8").read())

elif selected == "🕵 Fraud Detection":
    exec(open("shdashboard/Fraud_Detection.py", encoding="utf-8").read())
