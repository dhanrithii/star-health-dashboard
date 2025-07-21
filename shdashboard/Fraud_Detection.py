import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from streamlit_option_menu import option_menu
import zipfile
import os
import json
import base64



# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Fraud Detection",
        options=["🗂 Overview", "📎 FHIR Insights", "📊 Claim Anomalies", "📈 Provider Risk" ],
        icons=["clipboard-data", "exclamation-circle", "bar-chart-line", "file-earmark-text"],
        default_index=0
    )

zip_path = 'data/fhir.zip'
extract_path = 'data/fhir_data'

# Unzip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
fhir_dir = 'data/fhir_data/fhir'
all_fhir_resources = []
for filename in os.listdir(fhir_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(fhir_dir, filename)
        with open(filepath, 'r') as f:
            try:
                bundle = json.load(f)
                if bundle.get('resourceType') == 'Bundle' and 'entry' in bundle:
                    for entry in bundle['entry']:
                        if 'resource' in entry:
                            all_fhir_resources.append(entry['resource'])
                elif 'resourceType' in bundle:
                    all_fhir_resources.append(bundle)  # Handle non-bundle resources if any
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {filename}: {e}")

claim_data = []
for resource in all_fhir_resources:
    if resource.get('resourceType') == 'Claim':
        claim_id = resource.get('id')
        patient_id = resource.get('patient', {}).get('reference')
        provider_id = resource.get('provider', {}).get('reference')
        created_date = resource.get('created')
        billable_start = resource.get('billablePeriod', {}).get('start')
        billable_end = resource.get('billablePeriod', {}).get('end')
        claim_type = resource.get('type', {}).get('coding', [{}])[0].get('code')
        claim_status = resource.get('status')
        priority = resource.get('priority', {}).get('coding', [{}])[0].get('code')
        facility_id = resource.get('facility', {}).get('reference')
        insurance_coverage = resource.get('insurance', [{}])[0].get('coverage', {}).get('display')
        total_amount = resource.get('total', {}).get('value')
        currency = resource.get('total', {}).get('currency')

        diagnosis_codes = [d.get('diagnosisReference', {}).get('reference') for d in resource.get('diagnosis', [])]
        procedure_codes = [item.get('productOrService', {}).get('coding', [{}])[0].get('code') for item in resource.get('item', [])]
        procedure_texts = [item.get('productOrService', {}).get('text') for item in resource.get('item', [])]

        claim_data.append({
            'claim_id': claim_id,
            'patient_id': patient_id,
            'provider_id': provider_id,
            'created_date': created_date,
            'billable_start': billable_start,
            'billable_end': billable_end,
            'claim_type': claim_type,
            'claim_status': claim_status,
            'priority': priority,
            'facility_id': facility_id,
            'insurance_coverage': insurance_coverage,
            'total_amount': total_amount,
            'currency': currency,
            'diagnosis_codes': diagnosis_codes,
            'procedure_codes': procedure_codes,
            'procedure_texts': procedure_texts
        })

claim_df = pd.DataFrame(claim_data)

def generate_claim_document(row):
    return f"""
    Claim created on {row['created_date']} for patient {row['patient_id']}.
    Procedures performed: {', '.join(row['procedure_texts'])}.
    Diagnoses recorded: {', '.join(row['diagnosis_codes'])}.
    Priority level: {row['priority']}.
    Covered by: {row['insurance_coverage']}.
    Total billed: {row['total_amount']} {row['currency']}.
    Provider: {row['provider_id']} at facility {row['facility_id']}.
    """.strip()

# Create the new claim document column
claim_df['claim_document'] = claim_df.apply(generate_claim_document, axis=1)

# Load CSVs and images
claims_df = pd.read_csv("data/claim_anomalies.csv", encoding='utf-8')
provider_df = pd.read_csv("data/provider_risk_scores.csv", encoding='utf-8')

def get_base64_image(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def display_image_centered(path, caption="", width=600):
    img_base64 = get_base64_image(path)
    st.markdown(
        f'''
        <div style='text-align: center;'>
            <img src="data:image/png;base64,{img_base64}" width="{width}">
            <p><i>{caption}</i></p>
        </div>
        ''',
        unsafe_allow_html=True
    )

if selected == "🗂 Overview":
    st.title("🩺 Healthcare Fraud Detection Dashboard")
    st.markdown("""
    Welcome to the **Fraud Detection** module. This dashboard is built using synthetic FHIR data to detect potential anomalies in healthcare claims.

    **Components:**
    - Claims overview and suspicious flags
    - Risk scoring of providers
    - Graph-based visualizations and UMAP embeddings
    - FHIR-derived summaries and example documents
    """)

    st.subheader("📊 Capability Comparison Radar Chart")
    st.markdown("""
    Our proposed ClaimNet framework exhibits superior performance across most axes,
    highlighting its balanced design and adaptability.
    """)
    display_image_centered("data/compare.png", width=750)

    st.subheader("🧠 End-to-End System Architecture")
    st.markdown("""
    This diagram illustrates the architecture of the fraud detection pipeline:
    - **Preprocessing** synthetic FHIR data to extract claims, patients, and providers
    - **Longformer** embeddings for textual context
    - **Heterogeneous Graph Construction** to link entities
    - **Unsupervised Anomaly Detection** using node embeddings and dual-model voting (Isolation Forest & LOF)

    Each step is designed to preserve interpretability while ensuring scalable, real-time fraud detection.
    """)
    display_image_centered("data/architecture.png", width=750)


elif selected == "📎 FHIR Insights":
    st.header("📎 FHIR-Derived Data Snapshots")
    st.markdown("""
    This section provides a peek into the structured data extracted from synthetic FHIR (Fast Healthcare Interoperability Resources) bundles.

    You'll find:
    - 📝 **Claims** overview with linked entities
    - 💉 **Procedure descriptions** extracted from FHIR Procedure resources
    - 📄 A **sample claim document** in JSON-like format
    - 🕸️ A summary of the **heterogeneous graph** used for downstream fraud detection
    """)

    # --- Claims DataFrame Preview ---
    st.subheader("🔍 Claims DataFrame Preview")
    st.markdown("This table displays the first 25 rows of claim-level data, including key FHIR attributes like `procedure_texts`, `diagnosis`, and `provider_id` that are used for graph construction and anomaly detection.")
    st.dataframe(claim_df.head(25))

    # --- Procedure Texts ---
    st.subheader("💉 Unique Procedure Descriptions")
    st.markdown("Below are sample procedure names extracted from FHIR bundles, such as surgeries or tests. These are embedded using Longformer to capture semantic similarity.")
    all_procedures = pd.Series(sum(claim_df['procedure_texts'].dropna().tolist(), []))
    st.write(all_procedures.unique().tolist()[:15])

    # --- Sample Claim Document ---
    st.subheader("📝 Sample Claim Document")
    st.markdown("A raw claim object as extracted from the FHIR JSON. This includes metadata like billing codes, diagnosis details, and patient-provider associations.")
    st.markdown(f"""
    ```
    {claim_df['claim_document'].iloc[0]}
    ```
    """)

    # --- Enriched Graph Summary ---
    st.subheader("🧠 Enriched Graph Summary")
    st.markdown("""
    After processing the FHIR data, we generate a heterogeneous graph connecting claims, patients, and providers.

    - **🧑 Patient Nodes**
    - **🏥 Provider Nodes**
    - **📄 Claim Nodes**
    - **🔗 Edges** based on relationships like 'processed_by', 'submitted_by', and shared codes

    This graph is then embedded and analyzed for structural anomalies.
    """)
    st.write("**Nodes:** 521   **Edges:** 954")

    # --- Graph Visualizations ---
    display_image_centered("data/full graph.png", "📌 Full Heterogeneous Graph", width=1000)
    st.markdown("""
    The full graph shows all claims, patients, and providers interconnected based on FHIR-derived relationships. Colors represent different node types.
    """)

    display_image_centered("data/subgraph.png", "🔍 Subgraph of Suspicious Claims", width=1000)
    st.markdown("""
    This subgraph zooms into anomalous clusters identified by Isolation Forest and Local Outlier Factor, highlighting potential fraudulent behavior.
    """)


elif selected == "📊 Claim Anomalies":
    st.header("📊 Anomalous Claims Overview")
    st.markdown("""
    This section displays suspicious or anomalous claims identified by two unsupervised learning algorithms:

    - **🧪 Isolation Forest (IF)**: Detects anomalies by randomly isolating data points. Outliers are easier to isolate.
    - **🔍 Local Outlier Factor (LOF)**: Measures the local density of data points. Claims that deviate significantly from their neighbors are flagged.

    These methods operate on graph-based embeddings of claims derived from FHIR data.
    """)

    # Data preview
    st.subheader("📄 Top 50 Claims with Anomaly Flags")
    st.markdown("The table below lists the first 50 claim records along with anomaly labels from both algorithms.")
    st.dataframe(claims_df.head(50))

    # Outlier counts
    iso_count = claims_df['is_iso_outlier'].sum()
    lof_count = claims_df['is_lof_outlier'].sum()
    both_count = claims_df[(claims_df['is_iso_outlier'] == 1) & (claims_df['is_lof_outlier'] == 1)].shape[0]


    st.markdown("### 🚨 Outlier Counts Detected")
    col1, col2, col3 = st.columns(3)
    col1.metric("Isolation Forest Outliers", iso_count)
    col2.metric("LOF Outliers", lof_count)
    col3.metric("Common in Both", both_count)

    # UMAP visualization
    st.subheader("🔻 UMAP Visualization of Embeddings")
    st.markdown("""
    To understand how claim data clusters in high-dimensional space, we used **UMAP** (Uniform Manifold Approximation and Projection) to reduce dimensionality for visualization.

    Each dot in the plot represents a node (claim, patient, or provider). Outliers often appear far from dense clusters.
    """)
    display_image_centered("data/umap.png", "📉 Low-Dimensional Projection of Node Embeddings", width=700)


elif selected == "📈 Provider Risk":
    st.header("📈 Provider-Level Risk Analysis")

    st.markdown("""
    Providers are scored based on the **ratio of anomalous claims** they have submitted.

    A **higher anomaly ratio** suggests potentially fraudulent behavior or inconsistent reporting.

    The table below shows the top 10 high-risk providers.
    """)

    # Rename provider IDs to Provider A, B, C...
    provider_df_sorted = provider_df.sort_values(by='anomaly_ratio', ascending=False).copy()
    provider_df_sorted = provider_df_sorted.reset_index(drop=True)
    provider_df_sorted['provider_label'] = ['Provider ' + chr(65 + i) for i in range(len(provider_df_sorted))]

    st.dataframe(
        provider_df_sorted[['provider_label', 'total_claims', 'anomaly_ratio']].head(10).rename(columns={
            'provider_label': 'Provider',
            'total_claims': 'Total Claims',
            'anomaly_ratio': 'Anomaly Ratio'
        }),
        use_container_width=True
    )


    st.subheader("🔵 Bubble Chart: Provider Risk Landscape")
    st.markdown("""
    This chart maps providers based on:
    - **📌 X-axis**: Total number of claims
    - **📌 Y-axis**: Proportion flagged as anomalies
    - **🔴 Bubble size**: Total claim volume
    - **Color**: Risk intensity (based on anomaly ratio)
    """)

    fig_bubble, ax_bubble = plt.subplots(figsize=(10, 6))
    scatter = ax_bubble.scatter(
        provider_df_sorted['total_claims'],
        provider_df_sorted['anomaly_ratio'],
        s=provider_df_sorted['anomaly_ratio'] * 500,
        c=provider_df_sorted['anomaly_ratio'],
        cmap='coolwarm',
        alpha=0.7,
        edgecolors='black'
    )

    # Group by unique anomaly_ratio and pick one provider per ratio
    unique_providers = provider_df_sorted.groupby('anomaly_ratio').first().reset_index()

    # Limit to avoid clutter if there are too many unique ratios
    unique_providers = unique_providers.head(10)  # Show first 10 unique ratios (adjust as needed)

    # Apply offsets to prevent overlap
    offsets = [(-20, 20), (20, -20), (-15, -15), (15, 15), (0, 25), (30, 0), (-30, 10), (10, -30), (-10, 30), (20, 10)]

    for i, row in unique_providers.iterrows():
        dx, dy = offsets[i % len(offsets)]
        ax_bubble.annotate(
            row['provider_label'],
            (row['total_claims'], row['anomaly_ratio']),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=9,
            weight='bold',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1)
        )
    ax_bubble.set_xlabel("Total Claims Submitted")
    ax_bubble.set_ylabel("Anomaly Ratio")
    ax_bubble.set_title("Provider Risk Bubble Chart")
    ax_bubble.grid(True, linestyle='--', alpha=0.5)
    cbar = plt.colorbar(scatter, ax=ax_bubble)
    cbar.set_label("Anomaly Ratio")

    st.pyplot(fig_bubble)

    st.caption("📉 Larger and redder bubbles represent providers submitting more and riskier claims.")


st.caption("📁 Fraud module powered by embeddings + unsupervised anomaly detection.")
