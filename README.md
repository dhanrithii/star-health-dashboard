# 🏥 Star Health Insurance Analytics Dashboard

A centralized Streamlit-based dashboard developed during my internship at **Star Health and Allied Insurance**.  
This dashboard includes **Product Analysis**, **Premium Prediction**, **Brochure Q&A (RAG)**, and **Fraud Detection** modules using machine learning and graph-based techniques.

---

## 📁 Project Structure

```

dashboard/
├── data/                         # All models, datasets, encoders, graphs, and images
│   ├── Assure\_PremiumChart.csv
│   ├── xgb\_model.pkl             # Trained XGBoost model
│   ├── plan\_encoder.pkl, zone\_encoder.pkl
│   ├── claim\_anomalies.csv, provider\_risk\_scores.csv
│   ├── health\_insurance.db      # SQLite database for product data
│   ├── fhir.zip                 # Synthetic FHIR data (Synthea)
│   └── \*.png                    # Visualizations (UMAP, architecture, etc.)
│
├── shdashboard/                 # Streamlit modules
│   ├── Product\_Analysis.py
│   ├── Premium\_Prediction.py
│   ├── Brochure\_QA.py
│   └── Fraud\_Detection.py
│
├── streamlit\_app.py             # Master Streamlit launcher
└── requirements.txt             # Dependencies

````

---

## 🚀 Features

- **Product Analysis**: Visual insights into age groups, pricing, and medical benefits.
- **Premium Prediction**: ML-based premium estimator using XGBoost with SHAP explanations.
- **Brochure Q&A**: Retrieval-Augmented Generation using Gemini API for brochure understanding.
- **Fraud Detection**: Longformer + Heterogeneous GNN-based unsupervised anomaly detection from FHIR data.

---

## ⚙️ Setup Instructions

1. **Clone the Repo**
```bash
git clone https://github.com/your-username/star-health-dashboard.git
cd star-health-dashboard/dashboard
````

2. **Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate     # or venv\Scripts\activate on Windows
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Add Gemini API Key (for Brochure\_QA)**
   Use Streamlit secrets (recommended):

```bash
mkdir -p ~/.streamlit
echo -e "[general]\n" > ~/.streamlit/secrets.toml
echo "GEMINI_API_KEY = 'your-api-key-here'" >> ~/.streamlit/secrets.toml
```

5. **Run the Dashboard**

```bash
streamlit run streamlit_app.py
```

---

## 📸 Sample Visuals

* Premium prediction SHAP waterfall plots
* UMAP embedding visualization of fraud nodes
* Risk-based provider scores
* Radar charts comparing health insurance products

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 👩‍💼 Author

**Dhanrithii D.**
Data Science Intern @ Star Health Insurance

---

## 🙏 Acknowledgements

* Synthetic health data generated using [Synthea](https://synthetichealth.github.io/synthea/)
* Longformer Embeddings by [AllenAI](https://arxiv.org/abs/2004.05150)
* Heterogeneous Graphs using [DGL](https://www.dgl.ai/)
* Gemini API by Google

---

```
Let me know if you want to:
- Customize license (e.g., Apache-2.0, GPLv3, etc.)
- Add contributor guidelines
- Include a deploy-to-Streamlit Cloud badge  
- Add citation for a paper/presentation

Ready to assist with `.gitignore`, `LICENSE`, and Streamlit Cloud deployment next if needed.
```
