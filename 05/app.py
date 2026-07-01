import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import zipfile
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import skew, kurtosis, shapiro
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from io import BytesIO
import io

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="RPA Global Intelligence Dashboard 2026",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .stApp { background-color: #0D1117; color: #E6EDF3; }
    .main > div { background-color: #0D1117; }
    .metric-card {
        background: #161B22;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #30363D;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); border-color: #FFB703; }
    .metric-value { font-size: 28px; font-weight: 700; color: #E6EDF3; }
    .metric-label { font-size: 14px; color: #8B949E; font-weight: 500; }
    .section-header {
        color: #FFB703;
        font-size: 20px;
        font-weight: 700;
        border-bottom: 2px solid #30363D;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }
    .sub-header { color: #E6EDF3; font-size: 16px; font-weight: 600; margin-top: 20px; margin-bottom: 12px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #161B22; border-radius: 8px; padding: 6px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; border-radius: 6px; padding: 8px 16px; color: #8B949E; font-weight: 500; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #FFB703; color: #0D1117; }
    .footer {
        color: #8B949E;
        font-size: 12px;
        text-align: center;
        padding: 24px 0 12px 0;
        border-top: 1px solid #30363D;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- DATA LOADING FROM ZIP ----------
@st.cache_data
def load_data_from_zip(zip_path='archive.zip'):
    """Load all files from archive.zip and return dataframes."""
    if not os.path.exists(zip_path):
        st.error(f"❌ {zip_path} not found in current directory.")
        return None, None, None, None
    
    data = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()
        for fname in file_list:
            if fname.endswith('.csv'):
                with z.open(fname) as f:
                    df = pd.read_csv(f, low_memory=False)
                    data[fname] = df
            elif fname.endswith('.json'):
                with z.open(fname) as f:
                    content = f.read().decode('utf-8')
                    json_data = json.loads(content)
                    if isinstance(json_data, list):
                        df = pd.DataFrame(json_data)
                    elif isinstance(json_data, dict):
                        df = pd.json_normalize(json_data)
                    else:
                        df = pd.DataFrame()
                    data[fname] = df
    # Extract specific files
    df_companies = data.get('rpa_companies.csv', pd.DataFrame())
    df_projects = data.get('automation_projects.csv', pd.DataFrame())
    df_bots = data.get('software_bots.csv', pd.DataFrame())
    df_market = data.get('rpa_market_statistics.json', pd.DataFrame())
    return df_companies, df_projects, df_bots, df_market

@st.cache_data
def load_image():
    """Load rpa.png from current directory."""
    for path in ['rpa.png', 'RPA.png', 'rpa.jpg']:
        if os.path.exists(path):
            return Image.open(path)
    return None

# ---------- DATA CLEANING HELPERS ----------
def clean_df(df, name):
    if df.empty:
        return df
    # Strip whitespace from strings
    str_cols = df.select_dtypes(include='object').columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({'nan': np.nan, 'None': np.nan, '': np.nan, 'N/A': np.nan})
    # Drop duplicates
    df = df.drop_duplicates()
    # Drop columns >85% missing
    high_null = df.columns[df.isna().mean() > 0.85].tolist()
    if high_null:
        df = df.drop(columns=high_null)
    # Convert numeric-looking objects
    for c in df.select_dtypes(include='object').columns:
        converted = pd.to_numeric(df[c].astype(str).str.replace(',','').str.replace('%',''), errors='coerce')
        if converted.notna().sum() / max(len(df), 1) > 0.80:
            df[c] = converted
    return df

def safe_col(df, keywords, dtype='any'):
    if df.empty:
        return None
    cols = []
    for c in df.columns:
        if any(k.lower() in c.lower() for k in keywords):
            if dtype == 'num' and pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
            elif dtype == 'cat' and not pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
            elif dtype == 'any':
                cols.append(c)
    return cols[0] if cols else None

# ---------- MAIN APP ----------
def main():
    # Load data
    df_companies, df_projects, df_bots, df_market = load_data_from_zip('archive.zip')
    if df_companies is None:
        st.error("❌ Failed to load data from archive.zip. Please ensure the file is in the current directory.")
        return
    
    # Clean
    df_companies = clean_df(df_companies, 'companies')
    df_projects = clean_df(df_projects, 'projects')
    df_bots = clean_df(df_bots, 'bots')
    df_market = clean_df(df_market, 'market')
    
    st.success(f"✅ Data loaded: Companies={len(df_companies)}, Projects={len(df_projects)}, Bots={len(df_bots)}")
    
    # Load image
    img = load_image()
    
    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.markdown("### 🤖 RPA Global Intelligence")
        st.markdown("---")
        if img is not None:
            st.image(img, use_container_width=True, caption="🤖 RPA 2026")
        else:
            st.info("📷 rpa.png not found.")
        st.markdown("---")
        st.markdown("### 📊 Dashboard Navigation")
        sections = [
            "🏠 Overview",
            "🏢 Company",
            "🏭 Industry",
            "🌍 Geographic",
            "🤝 Vendor",
            "⚙️ Process",
            "📈 Productivity",
         #   "🧠 AI Integration",
            "💰 Economic",
            "📐 Advanced Math",
            "⚛️ Physics-Inspired",
            "📅 Temporal",
            "📊 Statistics",
            "🤖 ML Prediction"
        ]
        selected = st.radio("", sections, index=0, label_visibility="collapsed")
        st.markdown("---")
        st.markdown("### 📋 Dataset Info")
        st.markdown(f"- Companies: {len(df_companies):,}")
        st.markdown(f"- Projects: {len(df_projects):,}")
        st.markdown(f"- Bots: {len(df_bots):,}")
        st.markdown(f"- Market: {len(df_market):,}")
        st.markdown("---")
        st.markdown("💡 **Tip:** Hover over charts for details.")
        st.markdown("---")
        st.markdown("© 2026 RPA Intelligence")
    
    # ---------- COMPUTE KPIs ----------
    KPI = {}
    if not df_companies.empty:
        KPI['total_companies'] = len(df_companies)
        sector_col = safe_col(df_companies, ['sector','industry'], 'cat')
        country_col = safe_col(df_companies, ['country','nation'], 'cat')
        KPI['total_industries'] = df_companies[sector_col].nunique() if sector_col else 'N/A'
        KPI['total_countries'] = df_companies[country_col].nunique() if country_col else 'N/A'
    if not df_projects.empty:
        KPI['total_projects'] = len(df_projects)
        roi_col = safe_col(df_projects, ['roi','return'], 'num')
        save_col = safe_col(df_projects, ['saving','cost'], 'num')
        KPI['avg_roi'] = df_projects[roi_col].mean() if roi_col else None
        KPI['avg_savings'] = df_projects[save_col].mean() if save_col else None
    if not df_bots.empty:
        KPI['total_bots'] = len(df_bots)
        vendor_col = safe_col(df_bots, ['vendor','provider','platform'], 'cat')
        KPI['total_vendors'] = df_bots[vendor_col].nunique() if vendor_col else 'N/A'
    
    # ---------- KPI ROW ----------
    kpi_display = [
        ('Total Companies', KPI.get('total_companies', 'N/A'), '#58a6ff'),
        ('Industries', KPI.get('total_industries', 'N/A'), '#3fb950'),
        ('Countries', KPI.get('total_countries', 'N/A'), '#d29922'),
        ('Vendors', KPI.get('total_vendors', 'N/A'), '#f78166'),
        ('Bots', KPI.get('total_bots', 'N/A'), '#79c0ff'),
        ('Projects', KPI.get('total_projects', 'N/A'), '#56d364'),
    ]
    cols = st.columns(6)
    for i, (label, value, color) in enumerate(kpi_display):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="display:flex;align-items:center;">
                    <span class="metric-value" style="color:{color};">{value}</span>
                </div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("---")
    
    # ---------- SECTION HANDLING ----------
    if selected == "🏠 Overview":
        st.markdown('<div class="section-header">📊 Executive Overview</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            # Companies by sector
            if not df_companies.empty:
                sector_col = safe_col(df_companies, ['sector','industry'], 'cat')
                if sector_col:
                    counts = df_companies[sector_col].value_counts().head(12)
                    fig = px.bar(x=counts.values, y=counts.index, orientation='h',
                                 title='Companies by Sector',
                                 color=counts.values, color_continuous_scale='Blues',
                                 template='plotly_dark')
                    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            # Projects ROI distribution
            if not df_projects.empty:
                roi_col = safe_col(df_projects, ['roi','return'], 'num')
                if roi_col:
                    fig = px.histogram(df_projects, x=roi_col, nbins=40,
                                       title='ROI Distribution (Projects)',
                                       template='plotly_dark', color_discrete_sequence=['#58a6ff'])
                    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        with col2:
            # Bots by vendor
            if not df_bots.empty:
                vendor_col = safe_col(df_bots, ['vendor','provider','platform'], 'cat')
                if vendor_col:
                    counts = df_bots[vendor_col].value_counts().head(10)
                    fig = px.pie(values=counts.values, names=counts.index,
                                 title='Bot Vendor Market Share',
                                 color_discrete_sequence=px.colors.sequential.Blues_r,
                                 template='plotly_dark', hole=0.4)
                    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            # AI integration if available
            # ai_col = safe_col(df_projects, ['ai','ml','intelligence'], 'num') or safe_col(df_companies, ['ai','ml'], 'num')
            # if ai_col:
            #     data = df_projects[ai_col].dropna() if ai_col in df_projects.columns else df_companies[ai_col].dropna()
            #     fig = px.histogram(data, nbins=30, title='AI Integration Score Distribution',
            #                        template='plotly_dark', color_discrete_sequence=['#d29922'])
            #     fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
            #     st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    elif selected == "🏢 Company":
        st.markdown('<div class="section-header">🏢 Company Analytics</div>', unsafe_allow_html=True)
        if df_companies.empty:
            st.warning("Company data not available.")
        else:
            num_cols = df_companies.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                # Distribution of first few numeric columns
                for i in range(0, min(len(num_cols), 4), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(num_cols[i:i+2]):
                        with cols[j]:
                            data = df_companies[col].dropna()
                            fig = px.histogram(data, nbins=30, title=f'{col.replace("_"," ").title()}',
                                               template='plotly_dark', color_discrete_sequence=['#58a6ff'])
                            fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            # Top companies by ROI
            roi_col = safe_col(df_companies, ['roi','return'], 'num')
            name_col = safe_col(df_companies, ['company','name'], 'cat')
            if roi_col and name_col:
                top = df_companies[[name_col, roi_col]].dropna().nlargest(15, roi_col)
                fig = px.bar(top, y=name_col, x=roi_col, orientation='h',
                             title='Top 15 Companies by ROI',
                             color=roi_col, color_continuous_scale='Reds',
                             template='plotly_dark')
                fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    elif selected == "🏭 Industry":
        st.markdown('<div class="section-header">🏭 Industry Sector Analytics</div>', unsafe_allow_html=True)
        # Use companies or projects
        df_use = df_companies if not df_companies.empty else df_projects
        if df_use.empty:
            st.warning("No sector data available.")
        else:
            sector_col = safe_col(df_use, ['sector','industry'], 'cat')
            if sector_col:
                counts = df_use[sector_col].value_counts().head(15)
                fig = px.bar(x=counts.values, y=counts.index, orientation='h',
                             title='Top Sectors by Volume',
                             color=counts.values, color_continuous_scale='Teal',
                             template='plotly_dark')
                fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Average ROI by sector if available
                if 'roi' in df_use.columns or 'return' in df_use.columns:
                    roi_col = safe_col(df_use, ['roi','return'], 'num')
                    if roi_col:
                        sector_roi = df_use.groupby(sector_col)[roi_col].mean().sort_values(ascending=False).head(15)
                        fig2 = px.bar(x=sector_roi.index, y=sector_roi.values,
                                      title='Average ROI by Sector',
                                      color=sector_roi.values, color_continuous_scale='Blues',
                                      template='plotly_dark')
                        fig2.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    
    elif selected == "🌍 Geographic":
        st.markdown('<div class="section-header">🌍 Geographic Analytics</div>', unsafe_allow_html=True)
        df_use = df_companies if not df_companies.empty else df_projects
        if df_use.empty:
            st.warning("No geographic data available.")
        else:
            country_col = safe_col(df_use, ['country','nation'], 'cat')
            if country_col:
                counts = df_use[country_col].value_counts().head(20)
                fig = px.bar(x=counts.values, y=counts.index, orientation='h',
                             title='Top Countries by Volume',
                             color=counts.values, color_continuous_scale='Blues',
                             template='plotly_dark')
                fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Choropleth if we have a numeric metric
                num_col = safe_col(df_use, ['roi','saving','productivity'], 'num')
                if num_col:
                    geo_data = df_use.groupby(country_col)[num_col].mean().reset_index()
                    try:
                        fig_map = px.choropleth(geo_data, locations=country_col, locationmode='country names',
                                                color=num_col, color_continuous_scale='Blues',
                                                title=f'Average {num_col} by Country',
                                                template='plotly_dark')
                        fig_map.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})
                    except Exception as e:
                        st.info(f"Choropleth not available: {e}")
    
    elif selected == "🤝 Vendor":
        st.markdown('<div class="section-header">🤝 Vendor Analytics</div>', unsafe_allow_html=True)
        if df_bots.empty:
            st.warning("Bot data not available.")
        else:
            vendor_col = safe_col(df_bots, ['vendor','provider','platform'], 'cat')
            if vendor_col:
                counts = df_bots[vendor_col].value_counts().head(15)
                fig = px.pie(values=counts.values, names=counts.index,
                             title='Vendor Market Share',
                             color_discrete_sequence=px.colors.sequential.Blues_r,
                             template='plotly_dark', hole=0.4)
                fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Vendor performance on a metric
                metric_col = safe_col(df_bots, ['efficiency','productivity','roi'], 'num')
                if metric_col:
                    perf = df_bots.groupby(vendor_col)[metric_col].mean().sort_values(ascending=False).head(12)
                    fig2 = px.bar(x=perf.index, y=perf.values,
                                  title=f'Average {metric_col} by Vendor',
                                  color=perf.values, color_continuous_scale='Reds',
                                  template='plotly_dark')
                    fig2.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    
    elif selected == "⚙️ Process":
        st.markdown('<div class="section-header">⚙️ Process Automation Analytics</div>', unsafe_allow_html=True)
        df_use = df_projects if not df_projects.empty else df_bots
        if df_use.empty:
            st.warning("Process data not available.")
        else:
            process_col = safe_col(df_use, ['process','workflow','type'], 'cat')
            if process_col:
                counts = df_use[process_col].value_counts().head(15)
                fig = px.bar(x=counts.values, y=counts.index, orientation='h',
                             title='Process Type Distribution',
                             color=counts.values, color_continuous_scale='Purples',
                             template='plotly_dark')
                fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            # Complexity vs volume if available
            complex_col = safe_col(df_use, ['complex','complexity'], 'any')
            volume_col = safe_col(df_use, ['volume','count'], 'num')
            if complex_col and volume_col:
                # Try to convert complexity to numeric
                complex_data = df_use[complex_col]
                if not pd.api.types.is_numeric_dtype(complex_data):
                    complex_data = complex_data.astype('category').cat.codes
                scatter_df = pd.DataFrame({'complexity': complex_data, 'volume': df_use[volume_col]}).dropna()
                fig2 = px.scatter(scatter_df, x='complexity', y='volume',
                                  title='Process Complexity vs Volume',
                                  trendline='ols', template='plotly_dark', opacity=0.6)
                fig2.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    
    elif selected == "📈 Productivity":
        st.markdown('<div class="section-header">📈 Productivity & Performance</div>', unsafe_allow_html=True)
        df_use = df_projects if not df_projects.empty else df_companies
        if df_use.empty:
            st.warning("Productivity data not available.")
        else:
            perf_cols = [c for c in df_use.select_dtypes(include=np.number).columns if 'prod' in c or 'effic' in c or 'throughput' in c or 'error' in c]
            if not perf_cols:
                perf_cols = df_use.select_dtypes(include=np.number).columns[:4].tolist()
            if perf_cols:
                # Show histograms of first 4
                cols = st.columns(2)
                for i, col in enumerate(perf_cols[:4]):
                    with cols[i % 2]:
                        data = df_use[col].dropna()
                        fig = px.histogram(data, nbins=30, title=f'{col.replace("_"," ").title()}',
                                           template='plotly_dark', color_discrete_sequence=['#3fb950'])
                        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                # Boxplot comparison (normalized)
                if len(perf_cols) > 1:
                    box_df = df_use[perf_cols].dropna(how='all')
                    if not box_df.empty:
                        norm_box = (box_df - box_df.mean()) / (box_df.std() + 1e-9)
                        fig_box = go.Figure()
                        for i, col in enumerate(perf_cols[:6]):
                            fig_box.add_trace(go.Box(y=norm_box[col].dropna(), name=col[:15]))
                        fig_box.update_layout(title='Normalized Performance Comparison',
                                              template='plotly_dark',
                                              height=400, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})
    
    # elif selected == "🧠 AI Integration":
    #     st.markdown('<div class="section-header">🧠 AI Integration Analytics</div>', unsafe_allow_html=True)
        # Combine datasets to find AI columns
        all_dfs = [df_companies, df_projects, df_bots]
        ai_cols = []
        for df in all_dfs:
            if not df.empty:
                cols = [c for c in df.columns if any(k in c.lower() for k in ['ai','ml','intelligence','cognitive','nlp'])]
                if cols:
                    ai_cols.extend(cols)
        ai_cols = list(set(ai_cols))
        if not ai_cols:
            st.warning("No AI-related columns found.")
        else:
            # Use the first dataset that has an AI col
            df_ai = None
            for df in all_dfs:
                if any(c in df.columns for c in ai_cols):
                    df_ai = df
                    break
            if df_ai is not None:
                col = ai_cols[0]
                data = df_ai[col].dropna()
                fig = px.histogram(data, nbins=30, title=f'Distribution of {col}',
                                   template='plotly_dark', color_discrete_sequence=['#d29922'])
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                # AI maturity segmentation
                if len(data) > 10:
                    q33, q66 = data.quantile(0.33), data.quantile(0.66)
                    maturity = pd.cut(data, bins=[-np.inf, q33, q66, np.inf],
                                      labels=['Early', 'Developing', 'Advanced'])
                    counts = maturity.value_counts()
                    fig2 = px.pie(values=counts.values, names=counts.index,
                                  title='AI Maturity Segmentation',
                                  color_discrete_sequence=px.colors.sequential.Blues_r,
                                  template='plotly_dark', hole=0.4)
                    fig2.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    
    elif selected == "💰 Economic":
        st.markdown('<div class="section-header">💰 Economic Impact</div>', unsafe_allow_html=True)
        df_use = df_projects if not df_projects.empty else df_companies
        if df_use.empty:
            st.warning("Economic data not available.")
        else:
            econ_cols = [c for c in df_use.select_dtypes(include=np.number).columns if any(k in c.lower() for k in ['roi','saving','cost','revenue','payback'])]
            if not econ_cols:
                econ_cols = df_use.select_dtypes(include=np.number).columns[:4].tolist()
            if econ_cols:
                # Violin plots for first 4
                cols = st.columns(2)
                for i, col in enumerate(econ_cols[:4]):
                    with cols[i % 2]:
                        data = df_use[col].dropna()
                        fig = px.violin(data, box=True, points='all', title=f'{col.replace("_"," ").title()}',
                                        template='plotly_dark', color_discrete_sequence=['#f78166'])
                        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                # ROI by sector if possible
                roi_col = safe_col(df_use, ['roi','return'], 'num')
                sector_col = safe_col(df_use, ['sector','industry'], 'cat')
                if roi_col and sector_col:
                    roi_sector = df_use.groupby(sector_col)[roi_col].mean().sort_values(ascending=False).head(15)
                    fig2 = px.bar(x=roi_sector.index, y=roi_sector.values,
                                  title='Average ROI by Sector',
                                  color=roi_sector.values, color_continuous_scale='Reds',
                                  template='plotly_dark')
                    fig2.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    
    elif selected == "📐 Advanced Math":
        st.markdown('<div class="section-header">📐 Advanced Mathematical Analysis</div>', unsafe_allow_html=True)
        df_use = df_projects if not df_projects.empty else df_companies
        if df_use.empty:
            st.warning("Numeric data not available.")
        else:
            num_df = df_use.select_dtypes(include=np.number)
            if num_df.shape[1] < 2:
                st.warning("Not enough numeric columns.")
            else:
                # Compute CV, skewness, kurtosis
                cv = num_df.std() / (num_df.mean().abs() + 1e-9)
                sk = num_df.apply(skew)
                ku = num_df.apply(kurtosis)
                metrics = pd.DataFrame({'CV': cv, 'Skewness': sk, 'Kurtosis': ku}).head(12)
                # Plot as bar charts
                col1, col2 = st.columns(2)
                with col1:
                    fig1 = px.bar(x=metrics.index[:10], y=metrics['CV'][:10],
                                  title='Coefficient of Variation (Entropy Proxy)',
                                  color=metrics['CV'][:10], color_continuous_scale='Blues',
                                  template='plotly_dark')
                    fig1.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
                with col2:
                    fig2 = px.bar(x=metrics.index[:10], y=metrics['Skewness'][:10],
                                  title='Skewness (Asymmetry)',
                                  color=metrics['Skewness'][:10], color_continuous_scale='Reds',
                                  template='plotly_dark')
                    fig2.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
                # Kurtosis
                fig3 = px.bar(x=metrics.index[:10], y=metrics['Kurtosis'][:10],
                              title='Excess Kurtosis (Tail Heaviness)',
                              color=metrics['Kurtosis'][:10], color_continuous_scale='Purples',
                              template='plotly_dark')
                fig3.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
    
    elif selected == "⚛️ Physics-Inspired":
        st.markdown('<div class="section-header">⚛️ Physics-Inspired System Dynamics</div>', unsafe_allow_html=True)
        df_use = df_projects if not df_projects.empty else df_companies
        if df_use.empty:
            st.warning("Numeric data not available.")
        else:
            num_df = df_use.select_dtypes(include=np.number).fillna(0)
            if num_df.shape[1] < 2:
                st.warning("Not enough numeric columns.")
            else:
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                scaled = pd.DataFrame(scaler.fit_transform(num_df), columns=num_df.columns)
                energy = (scaled ** 2).sum(axis=1)
                momentum = scaled.mean(axis=1)
                stability = 1 / (scaled.std(axis=1) + 1e-9)
                turbulence = scaled.std(axis=1)
                phys_df = pd.DataFrame({
                    'System Energy': energy,
                    'Momentum': momentum,
                    'Stability': stability.clip(upper=stability.quantile(0.99)),
                    'Turbulence': turbulence
                })
                cols = st.columns(2)
                for i, (name, col) in enumerate(zip(phys_df.columns, ['System Energy','Momentum','Stability','Turbulence'])):
                    with cols[i % 2]:
                        data = phys_df[col]
                        fig = px.histogram(data, nbins=40, title=f'{col}',
                                           template='plotly_dark', color_discrete_sequence=['#79c0ff'])
                        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                # Energy vs Turbulence
                fig2 = px.scatter(phys_df, x='Turbulence', y='System Energy', color='Momentum',
                                  size='Stability', title='Energy-Turbulence Landscape',
                                  color_continuous_scale='Blues', template='plotly_dark')
                fig2.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    
    elif selected == "📅 Temporal":
        st.markdown('<div class="section-header">📅 Temporal Analysis</div>', unsafe_allow_html=True)
        df_use = df_projects if not df_projects.empty else df_companies
        if df_use.empty:
            st.warning("Temporal data not available.")
        else:
            year_col = safe_col(df_use, ['year','yr'], 'num')
            if year_col:
                # Group by year
                num_col = safe_col(df_use, ['roi','saving','productivity'], 'num')
                if num_col:
                    trend = df_use.groupby(year_col)[num_col].mean().reset_index()
                    fig = px.line(trend, x=year_col, y=num_col, markers=True,
                                  title=f'Trend of {num_col} by Year',
                                  template='plotly_dark', color_discrete_sequence=['#58a6ff'])
                    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                # Count by year
                counts = df_use[year_col].value_counts().sort_index().reset_index()
                counts.columns = ['Year', 'Count']
                fig2 = px.bar(counts, x='Year', y='Count',
                              title='Record Count by Year',
                              template='plotly_dark', color_discrete_sequence=['#3fb950'])
                fig2.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("No year column found.")
    
    elif selected == "📊 Statistics":
        st.markdown('<div class="section-header">📊 Statistical Analysis</div>', unsafe_allow_html=True)
        df_use = df_projects if not df_projects.empty else df_companies
        if df_use.empty:
            st.warning("No numeric data for statistics.")
        else:
            num_df = df_use.select_dtypes(include=np.number)
            if num_df.shape[1] < 2:
                st.warning("Not enough numeric columns for correlation.")
            else:
                # Correlation heatmap
                corr = num_df.corr(method='pearson')
                fig = px.imshow(corr, text_auto='.2f', title='Pearson Correlation Matrix',
                                color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                                template='plotly_dark')
                fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                # Top correlated pairs
                pairs = corr.unstack().reset_index()
                pairs.columns = ['Var1','Var2','r']
                pairs = pairs[pairs['Var1'] < pairs['Var2']]
                pairs['abs_r'] = pairs['r'].abs()
                top = pairs.sort_values('abs_r', ascending=False).head(10)
                st.dataframe(top[['Var1','Var2','r']].round(4), use_container_width=True)
    
    elif selected == "🤖 ML Prediction":
        st.markdown('<div class="section-header">🤖 Machine Learning Prediction</div>', unsafe_allow_html=True)
        # Select best dataset and target
        target = None
        for df, name in [(df_projects, 'projects'), (df_companies, 'companies'), (df_bots, 'bots')]:
            if not df.empty:
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if num_cols:
                    # Prefer ROI, savings, productivity
                    for kw in ['roi','return','saving','productivity','efficiency']:
                        col = safe_col(df, [kw], 'num')
                        if col:
                            target = col
                            df_ml = df.copy()
                            break
                    if target is None:
                        target = num_cols[0]
                        df_ml = df.copy()
                    break
        if target is None:
            st.warning("No suitable target for ML found.")
        else:
            st.info(f"**Target:** {target}  |  **Dataset:** {len(df_ml):,} rows")
            # Feature engineering: numeric + encoded categorical
            features = df_ml.select_dtypes(include=np.number).columns.tolist()
            features = [c for c in features if c != target and df_ml[c].notna().mean() > 0.4]
            # Add encoded categoricals
            cat_cols = df_ml.select_dtypes(include='object').columns.tolist()[:3]
            for c in cat_cols:
                le = LabelEncoder()
                df_ml[c+'_enc'] = le.fit_transform(df_ml[c].fillna('Unknown').astype(str))
                features.append(c+'_enc')
            X = df_ml[features]
            y = df_ml[target]
            # Clean
            mask = y.notna()
            X = X[mask]
            y = y[mask]
            if len(X) < 30:
                st.warning("Not enough samples for ML.")
            else:
                imputer = SimpleImputer(strategy='median')
                X_imp = pd.DataFrame(imputer.fit_transform(X), columns=features)
                X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                col1, col2, col3 = st.columns(3)
                col1.metric("R²", f"{r2:.4f}")
                col2.metric("MAE", f"{mae:.2f}")
                col3.metric("RMSE", f"{rmse:.2f}")
                # Actual vs Predicted
                fig1 = px.scatter(x=y_test, y=y_pred, title='Actual vs Predicted',
                                  labels={'x':'Actual', 'y':'Predicted'},
                                  template='plotly_dark', opacity=0.5)
                lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
                fig1.add_shape(type='line', x0=lims[0], y0=lims[0], x1=lims[1], y1=lims[1],
                               line=dict(color='#FFB703', dash='dash'))
                fig1.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
                # Feature importance
                importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
                fig2 = px.bar(x=importance.head(10).values, y=importance.head(10).index, orientation='h',
                              title='Top 10 Feature Importances',
                              color=importance.head(10).values, color_continuous_scale='Blues',
                              template='plotly_dark')
                fig2.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    
    # ---------- FOOTER ----------
    st.markdown("""
    <div class="footer">
        🤖 RPA Global Intelligence Dashboard 2026 &nbsp;·&nbsp; 
        Built with Streamlit & Plotly &nbsp;·&nbsp; 
        Data: Worldwide RPA Database 2026
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()