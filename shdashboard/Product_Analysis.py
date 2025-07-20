import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
from streamlit_option_menu import option_menu
from adjustText import adjust_text
import matplotlib.cm as cm
import matplotlib.colors as mcolors

st.set_page_config(page_title="Product Analysis", layout="wide")

# --- Local navigation using sidebar ---
with st.sidebar:
    section = option_menu(
        menu_title="Product Analysis",
        options=["Overview", "Visual Comparison", "Rankings"],
        icons=["house", "bar-chart", "award"],
        default_index=0
    )

# --- Load data ---
@st.cache_data
def load_data():
    conn = sqlite3.connect("/content/dashboard/data/health_insurance.db")
    df_products = pd.read_sql_query("SELECT * FROM Products", conn)
    df_coverage = pd.read_sql_query("SELECT * FROM coverage", conn)
    conn.close()
    return df_products, df_coverage

df_products, df_coverage = load_data()
merged_df = pd.merge(df_products, df_coverage, on='product_code')

# --- Feature Engineering ---
df_products['short_name'] = df_products['product_name'].str.replace(' Insurance', '', regex=False)
merged_df['short_name'] = merged_df['product_name'].str.replace(' Insurance', '', regex=False)

df_products['min_entry_age_years'] = df_products['min_entry_age'] / 12
df_products['max_entry_age_years'] = df_products['max_entry_age'] / 12
df_products['age_range'] = df_products['max_entry_age_years'] - df_products['min_entry_age_years']

feature_columns = ['hospitalization', 'pre_post_hospitalization', 'maternity', 'newborn_coverage',
                  'daycare_procedures', 'domiciliary', 'ayush', 'automatic_restoration',
                  'health_checkup', 'second_opinion']

merged_df['total_features'] = merged_df[feature_columns].sum(axis=1)
merged_df['family_score'] = merged_df[['maternity', 'newborn_coverage', 'daycare_procedures']].sum(axis=1) / 3 * 10
merged_df['value_ratio'] = merged_df['total_features'] / (merged_df['max_sum_insured'] / 100000)

# ---------------- Overview ----------------
if section == "Overview":
    st.title("📊 Star Health Product Comparison Dashboard")

    st.markdown("""
    Welcome to the **Product Analysis** module of the dashboard!
    This tool enables a detailed comparison of **six health insurance plans** offered by **Star Health Insurance**:

    - 🟢 **Star Comprehensive**
    - 🔵 **Family Health Optima**
    - 🟠 **Medi Classic**
    - 🟣 **Senior Citizens Red Carpet**
    - 🟡 **Star Health Assure**
    - 🔴 **Young Star**

    This module brings together two key data sources:
    - 📁 **Products Table** – Contains basic policy metadata like plan name, type, zone, and entry age limits.
    - 🧾 **Coverage Table** – Details specific benefits, such as maternity cover, daycare procedures, room rent, and critical illness protection.

    These are joined using the `product_code` to create a unified, enriched dataset for comprehensive policy evaluation.
    """)

    with st.expander("📄 View Products Table"):
        st.dataframe(df_products)

    with st.expander("📄 View Coverage Table"):
        st.dataframe(df_coverage)

    with st.expander("📄 View Merged Table (Products + Coverage)"):
        st.dataframe(merged_df)


# ---------------- Visual Comparison ----------------
elif section == "Visual Comparison":
    st.header("💰 Sum Insured Range by Product")
    st.markdown("Each product offers a range of coverage (in ₹ Lakhs). This chart shows the lowest and highest amount you can claim under each plan.")

    # This part for the first chart remains unchanged
    fig0, ax0 = plt.subplots(figsize=(10, 6))
    products = df_products['short_name']
    min_sum = df_products['min_sum_insured'] / 100000
    max_sum = df_products['max_sum_insured'] / 100000
    bar_width = 0.5
    ax0.barh(products, max_sum - min_sum, left=min_sum, color='teal', height=bar_width)

    for i, (min_val, max_val) in enumerate(zip(min_sum, max_sum)):
        ax0.text(min_val - 1, i, f'{min_val:.1f}L', va='center', ha='right', fontsize=9)
        ax0.text(max_val + 0.5, i, f'{max_val:.1f}L', va='center', ha='left', fontsize=9)

    ax0.set_xlabel("Sum Insured (₹ Lakhs)")
    ax0.set_title("Sum Insured Range")
    ax0.grid(True, axis='x', linestyle='--', alpha=0.6)
    st.pyplot(fig0)

    # --- FORCEFULLY-ADJUSTED CODE FOR THE SCATTER PLOT ---
    st.header("🗺️ Product Positioning Map")
    st.markdown("Helps compare plans based on total benefits (y-axis) and maximum claim amount (x-axis). Color shows whether a health checkup is needed.")

    # 1. Increase the figure size to give labels more room to spread out
    fig2, ax2 = plt.subplots(figsize=(14, 9))

    # Original scatter plot code
    ax2.scatter(
        merged_df['max_sum_insured'] / 100000,
        merged_df['total_features'], # Ensure jitter is removed for consistent anchoring
        s=250,
        alpha=0.7,
        c=merged_df['medical_screening_required'],
        cmap='coolwarm_r',
        edgecolor='black',
        linewidth=0.5
    )

    # Create a list of text objects for each point
    texts = []
    for i, row in merged_df.iterrows():
        texts.append(
            ax2.text(
                row['max_sum_insured'] / 100000,
                row['total_features'],
                row['short_name'],
                fontsize=9 # Specify fontsize
            )
        )

    # 2. Use more aggressive parameters to force labels apart
    adjust_text(
        texts,
        ax=ax2, # Pass the axes to adjust_text
        # Increase the repulsion forces
        force_points=(0.5, 0.5),
        force_text=(0.5, 0.5),
        # Increase the padding around points and text
        expand_points=(2, 2),
        expand_text=(1.5, 1.5),
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
    )

    # Set labels and title
    ax2.set_xlabel("Max Sum Insured (₹ Lakhs)")
    ax2.set_ylabel("Number of Features")
    ax2.set_title("Positioning of Products by Coverage & Features")
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Display the plot in Streamlit
    st.pyplot(fig2)


    st.header("📋 Feature Availability")
    st.markdown("What percentage of plans offer each benefit like AYUSH, newborn cover, or health checkups?")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    feature_coverage = merged_df[feature_columns].sum() / len(merged_df) * 100
    feature_coverage = feature_coverage.sort_values(ascending=False)
    sns.barplot(x=feature_coverage.values, y=feature_coverage.index, palette="Blues_d", ax=ax1)
    ax1.set_xlabel('Coverage (%)')
    ax1.set_title('Feature Availability Across Products')
    ax1.set_xlim(0, 100)
    st.pyplot(fig1)

    st.header("🎯 Target Audience")
    st.markdown("Shows who each product is designed for—like families, young adults, or seniors.")
    fig_aud, ax_aud = plt.subplots()
    target_counts = df_products['target_audience'].value_counts()
    ax_aud.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=90,
               colors=sns.color_palette("Set2", len(target_counts)), textprops={'fontsize': 14})
    ax_aud.set_title("Target Audience Distribution")
    st.pyplot(fig_aud)

    st.header("🩺 Medical Screening Requirement")
    st.markdown("Whether or not a person must undergo a medical test to be eligible for each plan.")
    fig_scr, ax_scr = plt.subplots()
    screening_data = df_products.groupby('medical_screening_required').size()
    labels = ['Not Required', 'Required']
    colors = ['lightgreen', 'coral']
    ax_scr.pie(screening_data, labels=labels, autopct='%1.1f%%', startangle=90,
               colors=colors, textprops={'fontsize': 14})
    ax_scr.set_title("Medical Screening Requirement")
    st.pyplot(fig_scr)

# ---------------- Rankings ----------------
elif section == "Rankings":
    st.header("🧾 Entry Eligibility Window by Audience")
    st.markdown("""
    This chart shows the **average entry eligibility range** for different target audiences.

    - It reflects how many years wide the **entry window** is — i.e., the difference between the **maximum and minimum entry age** allowed.
    - For example, **Senior Citizens** may only enter a policy between ages 60 and 75 (15-year window), but they are covered **for life once enrolled**.
    """)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    grouped = df_products.groupby('target_audience')['age_range'].mean().reset_index()
    sns.barplot(x='target_audience', y='age_range', data=grouped, palette="Set3", ax=ax3)

    ax3.set_ylabel("Entry Eligibility Range (Years)", fontsize=12)
    ax3.set_xlabel("Target Audience", fontsize=12)
    ax3.set_title("Average Entry Eligibility Range by Target Group", fontsize=14)

    for p in ax3.patches:
        height = p.get_height()
        ax3.text(p.get_x() + p.get_width()/2, height + 0.5, f'{height:.1f} yrs', ha='center', fontsize=10)

    st.pyplot(fig3)


    st.header("⏱️ Waiting Periods by Product")
    st.markdown("Waiting periods are the number of months after buying a plan before coverage starts for maternity or pre-existing conditions.")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    waiting_df = merged_df[['short_name', 'maternity_waiting', 'peds_waiting']].copy().fillna(0)
    melted = pd.melt(waiting_df, id_vars='short_name', var_name='Waiting Type', value_name='Months')
    melted['Waiting Type'] = melted['Waiting Type'].replace({
        'maternity_waiting': 'Maternity',
        'peds_waiting': 'Pre-existing Conditions'
    })
    sns.barplot(x='short_name', y='Months', hue='Waiting Type', data=melted, palette="Set2", ax=ax4)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    ax4.set_title("Waiting Periods for Benefits")
    ax4.set_xlabel("Product")
    st.pyplot(fig4)

    st.header("👨‍👩‍👧‍👦 Family-Friendliness of Insurance Plans")
    st.markdown("""
    This chart ranks insurance plans based on a **Family-Friendliness Score** derived from three features:

    - **Maternity Coverage**
    - **Newborn Coverage**
    - **Daycare Procedures**

    The score is scaled from **0 to 10**, and helps identify plans most suitable for family needs.
    """)
    # Sort by score
    sorted_by_family = merged_df.sort_values('family_score', ascending=False)

    # Custom colormap
    cmap = cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=10)

    # Plotting
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    bars = ax5.barh(
        sorted_by_family['short_name'],
        sorted_by_family['family_score'],
        color=[cmap(norm(score)) for score in sorted_by_family['family_score']]
    )

    ax5.set_xlim(0, 10)
    ax5.set_title('Family-Friendliness Score (0–10)', fontsize=16)
    ax5.set_xlabel('Score', fontsize=12)
    ax5.grid(True, axis='x', alpha=0.3)

    # Add text labels to bars
    for i, score in enumerate(sorted_by_family['family_score']):
        ax5.text(score + 0.1, i, f'{score:.1f}/10', va='center', fontsize=10)

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax5, orientation='horizontal', pad=0.2)
    cbar.set_label('Family-Friendliness Scale')

    plt.tight_layout()
    st.pyplot(fig5)


    st.header("🌟 Overall Value Score")
    st.markdown("This smart score (0-10) is based on benefits, waiting time, sum insured, and if health screening is required.")
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    merged_df['value_score'] = (
        merged_df['total_features'] / merged_df['total_features'].max() * 0.4 +
        (1 - merged_df['medical_screening_required']) * 0.2 +
        (1 - merged_df['peds_waiting'] / merged_df['peds_waiting'].max()) * 0.2 +
        (merged_df['max_sum_insured'] / merged_df['max_sum_insured'].max()) * 0.2
    ) * 10
    ranked = merged_df.sort_values('value_score', ascending=False)[['short_name', 'value_score']]
    sns.barplot(x='value_score', y='short_name', data=ranked, palette="viridis", ax=ax6)
    ax6.set_xlim(0, 10)
    ax6.set_title("Overall Value Score (0–10)")
    for p in ax5.patches:
        ax6.text(p.get_width() + 0.2, p.get_y() + p.get_height()/2, f'{p.get_width():.1f}', va='center')
    st.pyplot(fig6)

st.caption("📍 All charts are based on a synthetic Star Health product database.")

