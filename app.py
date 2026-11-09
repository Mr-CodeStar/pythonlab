"""
Bank Customer Churn Analytics Dashboard
Interactive Streamlit Application for Exploratory Data Analysis & Churn Prediction Insights
Dataset: Bank_Churn.csv (10,000 retail banking customer observations, 13 attributes)
"""

import os
from datetime import datetime

# Configure Matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

import streamlit as st
import pandas as pd
import numpy as np

# Set writable cache directory for Matplotlib
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(__file__), '.mpl_cache')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="FinMetrics | Bank Customer Churn Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS Styling (Responsive Cards, Tables, Badges)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Page Structure */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Executive Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 12px;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
    }
    .metric-title {
        color: #64748b;
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-delta {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 5px;
    }
    .delta-positive { color: #dc2626; } /* High churn is critical alert */
    .delta-neutral { color: #2563eb; }
    .delta-good { color: #16a34a; }
    
    /* Category Summary Box */
    .category-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    
    /* Section Headings */
    .section-title {
        color: #1e293b;
        font-size: 1.25rem;
        font-weight: 700;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
    /* Responsive HTML Table Fallback */
    .custom-table-container {
        overflow-x: auto;
        max-height: 480px;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    table.styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        text-align: left;
    }
    table.styled-table thead tr {
        background-color: #0f172a;
        color: #ffffff;
        text-align: left;
        font-weight: 600;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    table.styled-table th, table.styled-table td {
        padding: 9px 13px;
        border-bottom: 1px solid #e2e8f0;
    }
    table.styled-table tbody tr:nth-of-type(even) {
        background-color: #f8fafc;
    }
    table.styled-table tbody tr:hover {
        background-color: #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper Functions for Cross-Platform Resilience
# ---------------------------------------------------------
def display_dataframe(df_to_show, max_rows=50):
    """
    Renders dataframe safely. Tries st.dataframe (pyarrow) first, 
    and gracefully falls back to a responsive HTML table if pyarrow DLL is restricted.
    """
    try:
        import pyarrow
        st.dataframe(df_to_show, use_container_width=True)
    except Exception:
        # Fallback to responsive, styled HTML table
        html_code = (
            f'<div class="custom-table-container">'
            f'{df_to_show.head(max_rows).to_html(classes="styled-table", index=False, escape=False)}'
            f'</div>'
        )
        st.markdown(html_code, unsafe_allow_html=True)


# ---------------------------------------------------------
# Data Ingestion & Caching
# ---------------------------------------------------------
@st.cache_data
def load_data():
    """Loads Bank Customer Churn dataset and applies feature engineering."""
    possible_paths = [
        "Bank_Churn.csv",
        "lab/Bank_Churn.csv",
        os.path.join(os.path.dirname(__file__), "Bank_Churn.csv")
    ]
    df = None
    for p in possible_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
            
    if df is None:
        st.error("Error: 'Bank_Churn.csv' not found. Please ensure the dataset is placed in the project root directory.")
        st.stop()
        
    # Standardize and add human-readable categorical labels
    df['Exited_Label'] = df['Exited'].map({0: 'Retained', 1: 'Churned'})
    df['Activity_Label'] = df['IsActiveMember'].map({1: 'Active Member', 0: 'Inactive Member'})
    df['Card_Label'] = df['HasCrCard'].map({1: 'Has Credit Card', 0: 'No Credit Card'})
    
    # Age Group Bins
    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[17, 30, 40, 50, 60, 100],
        labels=['18-30 Years', '31-40 Years', '41-50 Years', '51-60 Years', '60+ Years']
    )
    
    # Credit Score Tier
    df['CreditScoreBracket'] = pd.cut(
        df['CreditScore'],
        bins=[300, 579, 669, 739, 799, 900],
        labels=['Poor (<580)', 'Fair (580-669)', 'Good (670-739)', 'Very Good (740-799)', 'Exceptional (800+)']
    )
    
    # Balance Tier
    df['BalanceBracket'] = pd.cut(
        df['Balance'],
        bins=[-1, 1, 50000, 100000, 150000, 300000],
        labels=['Zero Balance', '$1 - $50k', '$50k - $100k', '$100k - $150k', '$150k+']
    )
    
    return df

df_raw = load_data()


# ---------------------------------------------------------
# Sidebar Navigation and Filter Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=60)
    st.title("🏦 FinMetrics")
    st.markdown("**Bank Customer Churn Portal**\n*Interactive Analytics & KPI Dashboard*")
    st.divider()

    st.subheader("⚙️ Filter Controls")
    
    # Reset Filters Button
    if st.button("🔄 Reset All Filters", use_container_width=True, type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Search Bar
    search_query = st.text_input("🔍 Search Surname or Customer ID", placeholder="e.g. Hargrave or 15634602")

    # Multiselect: Geography
    geography_options = sorted(df_raw['Geography'].unique())
    selected_geography = st.multiselect(
        "📍 Geography / Country",
        options=geography_options,
        default=geography_options,
        help="Filter customers across countries"
    )

    # Multiselect: Gender
    gender_options = sorted(df_raw['Gender'].unique())
    selected_gender = st.multiselect(
        "👤 Customer Gender",
        options=gender_options,
        default=gender_options
    )

    # Multiselect: Number of Products
    product_options = sorted(df_raw['NumOfProducts'].unique())
    selected_products = st.multiselect(
        "📦 Number of Products",
        options=product_options,
        default=product_options,
        help="Number of bank products held (1-4)"
    )

    # Selectbox: Membership Status
    activity_filter = st.selectbox(
        "⚡ Member Activity Status",
        options=["All Members", "Active Only", "Inactive Only"]
    )

    # Selectbox: Credit Card Status
    card_filter = st.selectbox(
        "💳 Credit Card Ownership",
        options=["All Customers", "Has Card", "No Card"]
    )

    # Selectbox: Churn Status
    churn_filter = st.selectbox(
        "🎯 Customer Status Filter",
        options=["All Statuses", "Retained Only", "Churned Only"]
    )

    # Sliders for continuous numerical attributes
    st.markdown("---")
    st.markdown("**Numerical Range Filters:**")
    
    # Age Slider
    min_age_val, max_age_val = int(df_raw['Age'].min()), int(df_raw['Age'].max())
    age_range = st.slider(
        "🎂 Age Range",
        min_value=min_age_val,
        max_value=max_age_val,
        value=(min_age_val, max_age_val)
    )

    # Credit Score Slider
    min_cs, max_cs = int(df_raw['CreditScore'].min()), int(df_raw['CreditScore'].max())
    credit_score_range = st.slider(
        "📊 Credit Score Range",
        min_value=min_cs,
        max_value=max_cs,
        value=(min_cs, max_cs)
    )

    # Balance Slider
    min_bal, max_bal = float(df_raw['Balance'].min()), float(df_raw['Balance'].max())
    balance_range = st.slider(
        "💰 Account Balance Range ($)",
        min_value=int(min_bal),
        max_value=int(max_bal),
        value=(int(min_bal), int(max_bal)),
        step=5000,
        format="$%d"
    )

    # Tenure Slider
    min_tenure, max_tenure = int(df_raw['Tenure'].min()), int(df_raw['Tenure'].max())
    tenure_range = st.slider(
        "⏳ Tenure Range (Years)",
        min_value=min_tenure,
        max_value=max_tenure,
        value=(min_tenure, max_tenure)
    )

    st.divider()
    st.caption("Data source: Bank Customer Churn Benchmark")
    st.caption("Deployment: Ready for Streamlit Community Cloud")


# ---------------------------------------------------------
# Dynamic Dataset Filtering Logic
# ---------------------------------------------------------
df_filtered = df_raw.copy()

# Apply Categorical Filters
if selected_geography:
    df_filtered = df_filtered[df_filtered['Geography'].isin(selected_geography)]
else:
    df_filtered = df_filtered.iloc[0:0]

if selected_gender:
    df_filtered = df_filtered[df_filtered['Gender'].isin(selected_gender)]
else:
    df_filtered = df_filtered.iloc[0:0]

if selected_products:
    df_filtered = df_filtered[df_filtered['NumOfProducts'].isin(selected_products)]
else:
    df_filtered = df_filtered.iloc[0:0]

# Membership Filter
if activity_filter == "Active Only":
    df_filtered = df_filtered[df_filtered['IsActiveMember'] == 1]
elif activity_filter == "Inactive Only":
    df_filtered = df_filtered[df_filtered['IsActiveMember'] == 0]

# Credit Card Filter
if card_filter == "Has Card":
    df_filtered = df_filtered[df_filtered['HasCrCard'] == 1]
elif card_filter == "No Card":
    df_filtered = df_filtered[df_filtered['HasCrCard'] == 0]

# Churn Status Filter
if churn_filter == "Retained Only":
    df_filtered = df_filtered[df_filtered['Exited'] == 0]
elif churn_filter == "Churned Only":
    df_filtered = df_filtered[df_filtered['Exited'] == 1]

# Apply Numerical Range Filters
df_filtered = df_filtered[
    (df_filtered['Age'] >= age_range[0]) & (df_filtered['Age'] <= age_range[1]) &
    (df_filtered['CreditScore'] >= credit_score_range[0]) & (df_filtered['CreditScore'] <= credit_score_range[1]) &
    (df_filtered['Balance'] >= balance_range[0]) & (df_filtered['Balance'] <= balance_range[1]) &
    (df_filtered['Tenure'] >= tenure_range[0]) & (df_filtered['Tenure'] <= tenure_range[1])
]

# Apply Search Query Filter if entered
if search_query.strip():
    q = search_query.strip().lower()
    df_filtered = df_filtered[
        df_filtered['Surname'].astype(str).str.lower().str.contains(q) |
        df_filtered['CustomerId'].astype(str).str.contains(q)
    ]

# Calculate Filter Metrics
total_raw_count = len(df_raw)
filtered_count = len(df_filtered)
pct_of_total = (filtered_count / total_raw_count * 100) if total_raw_count > 0 else 0


# ---------------------------------------------------------
# Main Application Header & Live Status
# ---------------------------------------------------------
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🏦 Bank Customer Churn Analytics")
    st.markdown(
        "Interactive analytics and decision-support dashboard investigating customer retention patterns, "
        "financial exposures, and churn risk drivers across European retail banking cohorts."
    )
with header_col2:
    st.markdown(
        f"""
        <div style="text-align: right; background-color: #f1f5f9; padding: 12px 16px; border-radius: 8px; border: 1px solid #cbd5e1;">
            <div style="font-size: 0.8rem; color: #64748b; font-weight: 600;">ACTIVE COHORT</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #1e293b;">{filtered_count:,} / {total_raw_count:,}</div>
            <div style="font-size: 0.8rem; color: #0284c7; font-weight: 500;">{pct_of_total:.1f}% of total portfolio</div>
        </div>
        """,
        unsafe_allow_html=True
    )

if filtered_count == 0:
    st.warning("⚠️ No customer records match the selected filter combination. Please broaden your sidebar filters.")
    st.stop()


# ---------------------------------------------------------
# Primary Navigation Tabs
# ---------------------------------------------------------
tab_dashboard, tab_eda, tab_viz, tab_insights, tab_data = st.tabs([
    "📊 Executive Dashboard",
    "🔍 Data Exploration & Group-By",
    "📈 Visual Analytics (Bar, Line & Scatter)",
    "💡 Churn Insights & Strategic Actions",
    "📁 Dataset Explorer & Export"
])


# =========================================================
# TAB 1: EXECUTIVE DASHBOARD
# =========================================================
with tab_dashboard:
    st.markdown("### 📌 Portfolio Key Performance Indicators (KPIs)")
    st.caption("Real-time aggregated metrics computed dynamically from your filtered subset.")
    
    # Calculate Core KPI Metrics
    churned_customers = int(df_filtered['Exited'].sum())
    churn_rate = (churned_customers / filtered_count * 100) if filtered_count > 0 else 0.0
    baseline_churn_rate = 20.37  # Full dataset benchmark
    churn_delta = churn_rate - baseline_churn_rate

    total_balance = float(df_filtered['Balance'].sum())
    avg_balance = float(df_filtered['Balance'].mean()) if filtered_count > 0 else 0.0

    total_salary = float(df_filtered['EstimatedSalary'].sum())
    avg_salary = float(df_filtered['EstimatedSalary'].mean()) if filtered_count > 0 else 0.0

    avg_credit_score = float(df_filtered['CreditScore'].mean()) if filtered_count > 0 else 0.0
    active_member_pct = (df_filtered['IsActiveMember'].mean() * 100) if filtered_count > 0 else 0.0

    # Arrange KPI cards in 4 columns
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">👥 Total Observations</div>
                <div class="metric-value">{filtered_count:,}</div>
                <div class="metric-delta delta-neutral">{pct_of_total:.1f}% of 10,000 Portfolio</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">💳 Average Balance</div>
                <div class="metric-value">${avg_balance:,.2f}</div>
                <div class="metric-delta delta-neutral">Per Customer in Cohort</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_col2:
        delta_class = "delta-positive" if churn_delta > 0 else "delta-good"
        delta_symbol = "+" if churn_delta > 0 else ""
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">📉 Portfolio Churn Rate</div>
                <div class="metric-value">{churn_rate:.2f}%</div>
                <div class="metric-delta {delta_class}">{delta_symbol}{churn_delta:.2f}% vs Base (20.37%)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">💰 Total Balance at Risk</div>
                <div class="metric-value">${total_balance / 1e6:,.2f}M</div>
                <div class="metric-delta delta-neutral">Exact: ${total_balance:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">📊 Average Credit Score</div>
                <div class="metric-value">{avg_credit_score:.1f}</div>
                <div class="metric-delta delta-neutral">Scale: 350 - 850 (FICO)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">💵 Average Estimated Salary</div>
                <div class="metric-value">${avg_salary:,.2f}</div>
                <div class="metric-delta delta-neutral">Total: ${total_salary / 1e6:,.2f}M</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">⚡ Active Member Ratio</div>
                <div class="metric-value">{active_member_pct:.1f}%</div>
                <div class="metric-delta delta-good">{int(df_filtered['IsActiveMember'].sum()):,} Engaged Users</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        has_card_count = int(df_filtered['HasCrCard'].sum())
        has_card_pct = (has_card_count / filtered_count * 100) if filtered_count > 0 else 0.0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🪪 Credit Card Holders</div>
                <div class="metric-value">{has_card_pct:.1f}%</div>
                <div class="metric-delta delta-neutral">{has_card_count:,} Cardholders</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Category-wise Record Counts using Group-By Operations
    st.markdown("### 🏢 Category-Wise Record Counts & Churn Profiles")
    st.caption("Detailed group-by summaries showing customer volume, active counts, and churn breakdown across dimensions.")

    cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)

    # 1. Geography Breakdown
    with cat_col1:
        st.markdown("#### 📍 By Geography")
        geo_group = df_filtered.groupby('Geography').agg(
            Total=('CustomerId', 'count'),
            Churned=('Exited', 'sum')
        ).reset_index()
        geo_group['ChurnRate'] = (geo_group['Churned'] / geo_group['Total'] * 100).round(1)
        for _, row in geo_group.iterrows():
            st.markdown(
                f"""
                <div class="category-box">
                    <strong>{row['Geography']}</strong>: {row['Total']:,} customers<br/>
                    <small>Churn: <strong>{row['ChurnRate']}%</strong> ({row['Churned']:,} exited)</small>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 2. Gender Breakdown
    with cat_col2:
        st.markdown("#### 👤 By Gender")
        gen_group = df_filtered.groupby('Gender').agg(
            Total=('CustomerId', 'count'),
            Churned=('Exited', 'sum')
        ).reset_index()
        gen_group['ChurnRate'] = (gen_group['Churned'] / gen_group['Total'] * 100).round(1)
        for _, row in gen_group.iterrows():
            st.markdown(
                f"""
                <div class="category-box">
                    <strong>{row['Gender']}</strong>: {row['Total']:,} customers<br/>
                    <small>Churn: <strong>{row['ChurnRate']}%</strong> ({row['Churned']:,} exited)</small>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 3. Product Holding Breakdown
    with cat_col3:
        st.markdown("#### 📦 By Product Count")
        prod_group = df_filtered.groupby('NumOfProducts').agg(
            Total=('CustomerId', 'count'),
            Churned=('Exited', 'sum')
        ).reset_index()
        prod_group['ChurnRate'] = (prod_group['Churned'] / prod_group['Total'] * 100).round(1)
        for _, row in prod_group.iterrows():
            st.markdown(
                f"""
                <div class="category-box">
                    <strong>{row['NumOfProducts']} Product(s)</strong>: {row['Total']:,} customers<br/>
                    <small>Churn: <strong>{row['ChurnRate']}%</strong> ({row['Churned']:,} exited)</small>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 4. Activity Breakdown
    with cat_col4:
        st.markdown("#### ⚡ By Member Activity")
        act_group = df_filtered.groupby('Activity_Label').agg(
            Total=('CustomerId', 'count'),
            Churned=('Exited', 'sum')
        ).reset_index()
        act_group['ChurnRate'] = (act_group['Churned'] / act_group['Total'] * 100).round(1)
        for _, row in act_group.iterrows():
            st.markdown(
                f"""
                <div class="category-box">
                    <strong>{row['Activity_Label']}</strong>: {row['Total']:,} customers<br/>
                    <small>Churn: <strong>{row['ChurnRate']}%</strong> ({row['Churned']:,} exited)</small>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Quick Visual Snapshot in Container
    with st.container():
        st.markdown("---")
        st.markdown("#### ⚡ Quick Snapshot: Retained vs Churned Distribution Across Balance Tiers")
        snap_col1, snap_col2 = st.columns([2, 1])
        with snap_col1:
            fig_snap, ax_snap = plt.subplots(figsize=(8, 3.8))
            sns.set_theme(style='whitegrid')
            
            # Prepare data
            balance_group = df_filtered.groupby(['BalanceBracket', 'Exited_Label'], observed=False).size().unstack(fill_value=0)
            if 'Retained' not in balance_group.columns:
                balance_group['Retained'] = 0
            if 'Churned' not in balance_group.columns:
                balance_group['Churned'] = 0
                
            balance_group[['Retained', 'Churned']].plot(
                kind='bar',
                stacked=True,
                color=['#22c55e', '#ef4444'],
                edgecolor='#1e293b',
                alpha=0.9,
                ax=ax_snap
            )
            ax_snap.set_title("Customer Volume by Balance Bracket and Churn Status", fontsize=11, fontweight='bold')
            ax_snap.set_xlabel("Account Balance Bracket", fontsize=9.5, fontweight='bold')
            ax_snap.set_ylabel("Customer Count", fontsize=9.5, fontweight='bold')
            ax_snap.tick_params(axis='x', rotation=15)
            ax_snap.legend(title="Customer Status", loc="upper right")
            plt.tight_layout()
            st.pyplot(fig_snap, use_container_width=True)
            plt.close(fig_snap)

        with snap_col2:
            st.info(
                """
                **💡 Key Snapshot Takeaways:**
                - Notice how **Zero Balance** accounts represent a massive customer block, primarily retained in France & Spain.
                - For customers holding balances above **$100,000**, the churn proportion visibly increases.
                - Use the **Visual Analytics** tab for deep comparative bar charts, trends, and bivariate scatter plots.
                """
            )


# =========================================================
# TAB 2: DATA EXPLORATION & GROUP-BY ANALYSIS
# =========================================================
with tab_eda:
    st.markdown("### 🔍 Exploratory Data Analysis & Dynamic Group-By")
    st.caption("Perform multi-dimensional group-by aggregations and examine the statistical properties of the dataset.")

    # Interactive Group-by Widget Section
    with st.container():
        st.markdown("#### 🎛️ Custom Group-by Aggregator")
        st.write("Group customers by any primary and secondary category, and compute summary aggregates dynamically.")
        
        agg_col1, agg_col2, agg_col3, agg_col4 = st.columns(4)
        with agg_col1:
            group_primary = st.selectbox(
                "Primary Grouping Dimension:",
                options=['Geography', 'Gender', 'NumOfProducts', 'IsActiveMember', 'HasCrCard', 'AgeGroup', 'CreditScoreBracket', 'Tenure'],
                index=0
            )
        with agg_col2:
            group_secondary = st.selectbox(
                "Secondary Grouping Dimension (Optional):",
                options=['None', 'Gender', 'Geography', 'NumOfProducts', 'IsActiveMember', 'HasCrCard', 'AgeGroup'],
                index=1
            )
        with agg_col3:
            target_metric = st.selectbox(
                "Numerical Metric to Aggregate:",
                options=['Balance', 'EstimatedSalary', 'CreditScore', 'Age', 'Tenure'],
                index=0
            )
        with agg_col4:
            agg_funcs = st.multiselect(
                "Aggregation Functions:",
                options=['count', 'mean', 'sum', 'median', 'std', 'min', 'max'],
                default=['count', 'mean', 'sum']
            )

        if not agg_funcs:
            agg_funcs = ['count', 'mean']

        # Perform Groupby
        if group_secondary == 'None' or group_secondary == group_primary:
            group_cols = [group_primary]
        else:
            group_cols = [group_primary, group_secondary]

        agg_dict = {target_metric: agg_funcs, 'Exited': ['mean', 'sum']}
        grouped_df = df_filtered.groupby(group_cols, observed=False).agg(agg_dict)
        
        # Flatten MultiIndex columns
        grouped_df.columns = [f"{col[0]}_{col[1]}" for col in grouped_df.columns]
        grouped_df = grouped_df.reset_index()
        
        # Format churn rate column to %
        if 'Exited_mean' in grouped_df.columns:
            grouped_df['Churn_Rate_%'] = (grouped_df['Exited_mean'] * 100).round(2)
            grouped_df = grouped_df.rename(columns={'Exited_sum': 'Total_Churned'})
            grouped_df = grouped_df.drop(columns=['Exited_mean'])

        display_dataframe(grouped_df)

    st.markdown("---")

    # Expandable Statistical Summary
    with st.expander("📊 Numerical Attributes Descriptive Statistics (5-Number Summary)", expanded=True):
        st.write("Comprehensive statistical profile (Count, Mean, Std, Min, 25%, 50%, 75%, Max) of the currently filtered cohort:")
        num_summary = df_filtered[['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']].describe().T
        num_summary['sum'] = df_filtered[['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']].sum()
        num_summary = num_summary[['count', 'sum', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(2)
        num_summary = num_summary.reset_index().rename(columns={'index': 'Attribute'})
        display_dataframe(num_summary)

    # Correlation Analysis
    with st.expander("🔗 Correlation Matrix (Heatmap Analysis)"):
        st.write("Pearson correlation coefficients between numerical attributes and customer churn (`Exited`):")
        num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']
        corr_matrix = df_filtered[num_cols].corr().round(3)
        
        corr_col1, corr_col2 = st.columns([3, 2])
        with corr_col1:
            fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='vlag', vmin=-0.3, vmax=0.3, ax=ax_corr, cbar_kws={'label': 'Correlation Coefficient'})
            ax_corr.set_title("Correlation Heatmap with Target Variable (Exited)", fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_corr, use_container_width=True)
            plt.close(fig_corr)

        with corr_col2:
            st.markdown(
                """
                **Key Correlation Takeaways:**
                - **`Age` (+0.285)**: Strongest positive linear correlation with churn. Older customers leave at substantially higher rates.
                - **`IsActiveMember` (-0.156)**: Strongest negative correlation with churn. Active banking participation reliably insulates against departure.
                - **`Balance` (+0.119)**: Moderate positive correlation; customers with substantial balances have more choices and churn more easily.
                - **`NumOfProducts` (-0.048)**: Non-linear; moderate products reduce churn, but excess (3-4) skyrockets churn.
                """
            )


# =========================================================
# TAB 3: VISUAL ANALYTICS (BAR, LINE & SCATTER)
# =========================================================
with tab_viz:
    st.markdown("### 📈 Visual Analytics: Distributions, Trends & Relationships")
    st.caption("Interactive charts fulfilling requirements for Categorical Comparison (Bar), Trend/Variation (Line), and Bivariate Association (Scatter).")

    # -----------------------------------------------------
    # Visual 1: Bar Chart (Categorical Comparisons)
    # -----------------------------------------------------
    st.markdown('<div class="section-title">1. Bar Chart: Categorical Comparison & Distribution</div>', unsafe_allow_html=True)
    
    bar_ctrl_col1, bar_ctrl_col2 = st.columns(2)
    with bar_ctrl_col1:
        bar_dimension = st.selectbox(
            "Select Categorical Dimension for Bar Chart:",
            options=['Geography', 'NumOfProducts', 'Gender', 'Activity_Label', 'Card_Label', 'AgeGroup', 'CreditScoreBracket'],
            index=0
        )
    with bar_ctrl_col2:
        bar_metric_type = st.selectbox(
            "Select Bar Chart Representation:",
            options=['Customer Count by Churn Status', 'Churn Rate Percentage (%)', 'Average Balance by Category ($)'],
            index=0
        )

    fig_bar, ax_bar = plt.subplots(figsize=(10, 4.5))
    sns.set_theme(style='whitegrid')

    if bar_metric_type == 'Customer Count by Churn Status':
        grouped_counts = df_filtered.groupby([bar_dimension, 'Exited_Label'], observed=False).size().unstack(fill_value=0)
        if 'Retained' not in grouped_counts.columns:
            grouped_counts['Retained'] = 0
        if 'Churned' not in grouped_counts.columns:
            grouped_counts['Churned'] = 0
        grouped_counts[['Retained', 'Churned']].plot(
            kind='bar',
            color=['#22c55e', '#ef4444'],
            edgecolor='#0f172a',
            alpha=0.9,
            ax=ax_bar
        )
        ax_bar.set_title(f"Customer Retention & Churn Count Across {bar_dimension}", fontsize=12, fontweight='bold', pad=10)
        ax_bar.set_ylabel("Customer Count", fontsize=10, fontweight='bold')
        ax_bar.set_xlabel(bar_dimension, fontsize=10, fontweight='bold')
        ax_bar.legend(title="Customer Status")
    elif bar_metric_type == 'Churn Rate Percentage (%)':
        churn_pct = (df_filtered.groupby(bar_dimension, observed=False)['Exited'].mean() * 100).round(2)
        bars = ax_bar.bar([str(x) for x in churn_pct.index], churn_pct.values, color='#ef4444', edgecolor='#0f172a', alpha=0.9)
        ax_bar.set_title(f"Customer Churn Rate (%) Across {bar_dimension}", fontsize=12, fontweight='bold', pad=10)
        ax_bar.set_ylabel("Churn Rate (%)", fontsize=10, fontweight='bold')
        ax_bar.set_xlabel(bar_dimension, fontsize=10, fontweight='bold')
        for bar in bars:
            yval = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    else:
        avg_bal = df_filtered.groupby(bar_dimension, observed=False)['Balance'].mean().round(2)
        bars = ax_bar.bar([str(x) for x in avg_bal.index], avg_bal.values, color='#3b82f6', edgecolor='#0f172a', alpha=0.9)
        ax_bar.set_title(f"Average Account Balance ($) Across {bar_dimension}", fontsize=12, fontweight='bold', pad=10)
        ax_bar.set_ylabel("Average Balance ($)", fontsize=10, fontweight='bold')
        ax_bar.set_xlabel(bar_dimension, fontsize=10, fontweight='bold')
        ax_bar.yaxis.set_major_formatter('${x:,.0f}')
        for bar in bars:
            yval = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2.0, yval + 1500, f'${yval:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax_bar.tick_params(axis='x', rotation=15 if len(str(df_filtered[bar_dimension].iloc[0])) > 6 else 0)
    plt.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)

    with st.expander("💡 Bar Chart Analytical Interpretation & Insights", expanded=True):
        st.markdown(
            f"""
            - **Category Analyzed**: `{bar_dimension}`
            - **Statistical Observation**:
              - In **Geography**, Germany displays a staggering **32.4%** churn rate—nearly double that of France (16.2%) and Spain (16.7%).
              - In **NumOfProducts**, holding **2 products** represents the optimal retention sweet-spot (only **7.6%** churn), whereas holding **3 products (82.7% churn)** or **4 products (100% churn)** causes catastrophic attrition.
              - In **Gender**, Female churn rate (**25.1%**) significantly outpaces Male churn rate (**16.5%**).
            - **Decision-Making Value**: Enables branch managers to target retention marketing directly at high-loss categories rather than applying generic blanket programs.
            """
        )

    st.markdown("---")

    # -----------------------------------------------------
    # Visual 2: Line Chart (Trends & Variations)
    # -----------------------------------------------------
    st.markdown('<div class="section-title">2. Line Chart: Variations & Trends Across Continuous Dimensions</div>', unsafe_allow_html=True)

    line_ctrl1, line_ctrl2 = st.columns(2)
    with line_ctrl1:
        line_x_axis = st.selectbox(
            "Select Numerical Attribute for X-Axis (Continuous Trend):",
            options=['Age', 'Tenure'],
            index=0
        )
    with line_ctrl2:
        line_y_axis = st.selectbox(
            "Select Response Metric for Y-Axis:",
            options=['Churn Rate (%)', 'Average Account Balance ($)', 'Average Estimated Salary ($)', 'Average Credit Score'],
            index=0
        )

    fig_line, ax_line = plt.subplots(figsize=(10, 4.5))

    # Compute trend line dataframe
    if line_y_axis == 'Churn Rate (%)':
        trend_df = df_filtered.groupby(line_x_axis)['Exited'].agg(['mean', 'count']).reset_index()
        trend_df = trend_df[trend_df['count'] >= 5]
        trend_df['Metric'] = trend_df['mean'] * 100
        y_title = "Churn Rate (%)"
        line_color = "#dc2626"
    elif line_y_axis == 'Average Account Balance ($)':
        trend_df = df_filtered.groupby(line_x_axis)['Balance'].agg(['mean', 'count']).reset_index()
        trend_df = trend_df[trend_df['count'] >= 5]
        trend_df['Metric'] = trend_df['mean']
        y_title = "Average Balance ($)"
        line_color = "#2563eb"
    elif line_y_axis == 'Average Estimated Salary ($)':
        trend_df = df_filtered.groupby(line_x_axis)['EstimatedSalary'].agg(['mean', 'count']).reset_index()
        trend_df = trend_df[trend_df['count'] >= 5]
        trend_df['Metric'] = trend_df['mean']
        y_title = "Average Salary ($)"
        line_color = "#059669"
    else:
        trend_df = df_filtered.groupby(line_x_axis)['CreditScore'].agg(['mean', 'count']).reset_index()
        trend_df = trend_df[trend_df['count'] >= 5]
        trend_df['Metric'] = trend_df['mean']
        y_title = "Average Credit Score"
        line_color = "#7c3aed"

    ax_line.plot(trend_df[line_x_axis], trend_df['Metric'], color=line_color, linewidth=2.5, marker='o', markersize=4, label=y_title)
    if line_x_axis == 'Age' and line_y_axis == 'Churn Rate (%)':
        ax_line.axvspan(45, 60, color='#fee2e2', alpha=0.6, label='Critical Risk Zone (45-60 yrs)')
    elif line_x_axis == 'Tenure' and line_y_axis == 'Churn Rate (%)':
        ax_line.axhline(20.37, color='gray', linestyle='--', linewidth=1.5, label='Portfolio Baseline (20.37%)')

    ax_line.set_title(f"Trend Analysis: {line_y_axis} across Customer {line_x_axis}", fontsize=12, fontweight='bold', pad=10)
    ax_line.set_xlabel(f"Customer {line_x_axis}", fontsize=10, fontweight='bold')
    ax_line.set_ylabel(y_title, fontsize=10, fontweight='bold')
    if '$' in y_title:
        ax_line.yaxis.set_major_formatter('${x:,.0f}')
    ax_line.legend(loc='best')
    plt.tight_layout()
    st.pyplot(fig_line, use_container_width=True)
    plt.close(fig_line)

    with st.expander("💡 Line Chart Analytical Interpretation & Trend Insights", expanded=True):
        st.markdown(
            f"""
            - **Trend Dimension**: `{line_x_axis}` vs `{line_y_axis}`
            - **Critical Pattern**:
              - When viewing **Age vs Churn Rate**, customer churn stays modest (~7.5% - 12%) below age 40, begins climbing sharply past 40, and peaks between **ages 50 and 60 at an alarming 56.2%**.
              - When viewing **Tenure vs Churn Rate**, the churn rate remains remarkably flat (~20%) from Tenure Year 1 to 9, indicating that length of relationship alone does not protect the bank from customer churn.
            - **Decision-Making Value**: Banking executives can formulate age-specific relationship programs for customers turning 45-55 to defend valuable retirement balances.
            """
        )

    st.markdown("---")

    # -----------------------------------------------------
    # Visual 3: Scatter Plot (Bivariate Relationship & Clusters)
    # -----------------------------------------------------
    st.markdown('<div class="section-title">3. Scatter Plot: Bivariate Association, Clusters & Outliers</div>', unsafe_allow_html=True)

    scat_col1, scat_col2, scat_col3, scat_col4 = st.columns(4)
    with scat_col1:
        scat_x = st.selectbox("X-Axis Variable:", options=['Age', 'CreditScore', 'Balance', 'EstimatedSalary', 'Tenure'], index=0)
    with scat_col2:
        scat_y = st.selectbox("Y-Axis Variable:", options=['Balance', 'EstimatedSalary', 'CreditScore', 'Age'], index=0)
    with scat_col3:
        scat_color = st.selectbox("Color By:", options=['Exited_Label', 'Geography', 'Gender', 'Activity_Label'], index=0)
    with scat_col4:
        sample_size = st.slider("Max Points to Plot (Smooth Performance):", min_value=500, max_value=min(10000, filtered_count), value=min(2500, filtered_count), step=250)

    # Sample data if large
    if len(df_filtered) > sample_size:
        scatter_sample = df_filtered.sample(sample_size, random_state=42)
    else:
        scatter_sample = df_filtered

    fig_scat, ax_scat = plt.subplots(figsize=(10, 5))
    if scat_color == 'Exited_Label':
        palette = {'Retained': '#16a34a', 'Churned': '#dc2626'}
    elif scat_color == 'Geography':
        palette = {'France': '#3b82f6', 'Germany': '#ef4444', 'Spain': '#10b981'}
    elif scat_color == 'Gender':
        palette = {'Male': '#2563eb', 'Female': '#ec4899'}
    else:
        palette = {'Active Member': '#059669', 'Inactive Member': '#e11d48'}

    sns.scatterplot(
        x=scat_x,
        y=scat_y,
        hue=scat_color,
        palette=palette,
        data=scatter_sample,
        alpha=0.55,
        s=30,
        edgecolor='none',
        ax=ax_scat
    )

    if scat_y == 'Balance':
        ax_scat.yaxis.set_major_formatter('${x:,.0f}')
        ax_scat.axhline(0, color='#64748b', linestyle=':', alpha=0.7)
    if scat_x == 'Balance':
        ax_scat.xaxis.set_major_formatter('${x:,.0f}')

    ax_scat.set_title(f"Bivariate Scatter Plot: {scat_x} vs {scat_y} (Grouped by {scat_color})", fontsize=12, fontweight='bold', pad=10)
    ax_scat.set_xlabel(scat_x, fontsize=10, fontweight='bold')
    ax_scat.set_ylabel(scat_y, fontsize=10, fontweight='bold')
    ax_scat.legend(title=scat_color, loc='best')
    plt.tight_layout()
    st.pyplot(fig_scat, use_container_width=True)
    plt.close(fig_scat)

    with st.expander("💡 Scatter Plot Pattern Analysis, Clusters & Outliers", expanded=True):
        st.markdown(
            f"""
            - **Axes Explored**: `{scat_x}` (Horizontal) vs `{scat_y}` (Vertical), grouped by `{scat_color}`.
            - **Identified Clusters & Patterns**:
              1. **The Zero-Balance Baseline Cluster**: Looking at Balance on either axis reveals a dense horizontal line at `$0` balance (36.17% of total customers). Interestingly, almost zero customers in Germany have zero balance; German customers are almost exclusively distributed in the `$75,000 - $160,000` balance range.
              2. **The Age-Churn Threshold**: In the `Age vs Balance` plot, red points (churned) are heavily concentrated in the vertical band between **Age 45 and 65**, cutting across all balance ranges.
              3. **Credit Score vs Balance Independence**: Credit score shows minimal clustering with balance or salary, indicating that affluent customers are just as likely to have average credit scores as lower-balance customers.
            - **Decision-Making Value**: Pinpoints distinct customer personas (e.g. affluent middle-aged German churners vs zero-balance young French retainers) for targeted retention intervention.
            """
        )


# =========================================================
# TAB 4: CHURN INSIGHTS & STRATEGIC ACTIONS
# =========================================================
with tab_insights:
    st.markdown("### 💡 Churn Insights & Strategic Business Recommendations")
    st.caption("Synthesizing exploratory findings into executive-level, data-driven decisions.")

    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.markdown(
            """
            #### 🚨 Top Risk Drivers Identified
            
            1. **Geographic Deficit in Germany**:
               - German customers churn at **32.44%**, double the ~16% observed in France and Spain.
               - German accounts carry high average balances (~$119k). This geographic disparity represents the single largest financial leakage.
            
            2. **Product Overloading / Misalignment**:
               - Customers with **1 product** churn at 27.7%.
               - Customers with **2 products** enjoy exceptional retention (**7.6% churn**).
               - Customers holding **3 products (82.7%)** and **4 products (100.0%)** experience catastrophic defection, signaling severe cross-selling friction or forced bundles.
            
            3. **Aging Demographic Cliff (45 - 60 Years)**:
               - Churn reaches **56.2%** for customers aged 51-60.
               - This group often moves retirement funds or wealth to specialized wealth management competitors.
            
            4. **Member Inactivity Risk**:
               - Inactive members churn at **26.85%**, compared to only **14.27%** for active members.
            """
        )

    with rec_col2:
        st.markdown(
            """
            #### 🎯 Recommended Strategic Action Plan
            
            1. **German Branch Quality & Retention Audit**:
               - Conduct localized customer satisfaction reviews in German branches to diagnose competitive fee pressure or service gaps.
               - Assign dedicated relationship managers to German accounts holding balances > $100k.
            
            2. **Cross-Sell Optimization (The '2-Product Rule')**:
               - Promote the secondary product to single-product holders to reduce churn from 27.7% to 7.6%.
               - Immediately halt pushing 3rd and 4th peripheral products until the onboarding journey is audited and streamlined.
            
            3. **Senior Wealth Advisory Transition Program**:
               - Design a personalized "Life-Stage Advisory" package for clients reaching age 45+.
               - Offer competitive term deposits and wealth advisory to prevent assets leaving the bank.
            
            4. **Automated Inactivity Re-Engagement Workflows**:
               - Implement automated 30-day inactivity alert triggers via mobile app notifications and targeted loyalty perks.
            """
        )

    st.markdown("---")
    
    # Financial Impact Calculator Simulation
    st.markdown("#### 🧮 Interactive Churn Reduction ROI Simulator")
    st.write("Simulate the financial capital preserved by reducing churn in the currently filtered portfolio.")
    
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        reduction_target_pct = st.slider("Target Churn Reduction (Percentage Points):", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
    with sim_col2:
        retained_customers_sim = int(filtered_count * (reduction_target_pct / 100.0))
        st.metric("Customers Rescued", f"{retained_customers_sim:,} Customers")
    with sim_col3:
        balance_saved = retained_customers_sim * avg_balance
        st.metric("Estimated Balance Preserved", f"${balance_saved / 1e6:,.2f} Million", delta=f"+${balance_saved:,.0f}")


# =========================================================
# TAB 5: DATASET EXPLORER & EXPORT
# =========================================================
with tab_data:
    st.markdown("### 📁 Filtered Dataset Inspection & Export")
    st.caption("Inspect raw individual records, customize column displays, and download the filtered cohort as CSV.")

    data_ctrl1, data_ctrl2 = st.columns([3, 1])
    with data_ctrl1:
        display_columns = st.multiselect(
            "Select Columns to Display:",
            options=list(df_raw.columns) + ['Exited_Label', 'Activity_Label'],
            default=['CustomerId', 'Surname', 'Geography', 'Gender', 'Age', 'CreditScore', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary', 'Exited_Label']
        )
    with data_ctrl2:
        rows_to_show = st.selectbox("Rows per page view:", options=[25, 50, 100, 250], index=0)

    # Display Table using safe cross-platform helper
    display_dataframe(df_filtered[display_columns], max_rows=rows_to_show)
    st.caption(f"Displaying top {min(rows_to_show, len(df_filtered))} of {len(df_filtered):,} matching records.")

    # CSV Download Button
    csv_bytes = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Dataset (CSV)",
        data=csv_bytes,
        file_name=f"bank_churn_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        type="primary"
    )

# ---------------------------------------------------------
# Application Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 10px;">
        Bank Customer Churn Analytics Platform &bull; Built with Python & Streamlit &bull; 
        Designed for Data Analytics Lab Submission &bull; 10,000 Customer Benchmark
    </div>
    """,
    unsafe_allow_html=True
)
