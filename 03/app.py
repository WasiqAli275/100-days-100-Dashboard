import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
import re

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Impact on Jobs 2030 Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background-color: #0D1117;
        color: #E6EDF3;
    }
    .main > div {
        background-color: #0D1117;
    }
    /* Card style */
    .metric-card {
        background: #161B22;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #30363D;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #FFB703;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #E6EDF3;
    }
    .metric-label {
        font-size: 14px;
        color: #8B949E;
        font-weight: 500;
    }
    .metric-icon {
        font-size: 32px;
        margin-right: 12px;
    }
    /* Section headers */
    .section-header {
        color: #FFB703;
        font-size: 20px;
        font-weight: 700;
        border-bottom: 2px solid #30363D;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }
    .sub-header {
        color: #E6EDF3;
        font-size: 16px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 12px;
    }
    /* Sidebar */
    .css-1d391kg {
        background-color: #161B22;
    }
    .css-1aumxhk {
        background-color: #161B22;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161B22;
        border-radius: 8px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        padding: 8px 16px;
        color: #8B949E;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFB703;
        color: #0D1117;
    }
    /* DataFrame */
    .dataframe {
        background-color: #161B22 !important;
        color: #E6EDF3 !important;
        border-radius: 8px;
        border: 1px solid #30363D;
    }
    .dataframe th {
        background-color: #1C2128 !important;
        color: #FFB703 !important;
    }
    /* Plotly container */
    .stPlotlyChart {
        background-color: #161B22;
        border-radius: 12px;
        border: 1px solid #30363D;
        padding: 8px;
    }
    /* Image container */
    .car-image-container {
        border-radius: 12px;
        border: 1px solid #30363D;
        overflow: hidden;
        background: #161B22;
    }
    /* Metric container row */
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin-bottom: 20px;
    }
    .metric-item {
        flex: 1;
        min-width: 160px;
    }
    /* Footer */
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

# ---------- DATA LOADING ----------
@st.cache_data
def load_data():
    """Load the AI job impact dataset from the local directory."""
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not csv_files:
        for root, dirs, files in os.walk('.'):
            for f in files:
                if f.endswith('.csv'):
                    csv_files.append(os.path.join(root, f))
    if not csv_files:
        st.error("❌ No CSV files found in the current directory. Please ensure 'AI_Impact_on_Jobs_2030.csv' is present.")
        return None
    
    # Prefer the exact filename if exists
    target = 'AI_Impact_on_Jobs_2030.csv'
    if target in csv_files:
        file_path = target
    else:
        # Take the largest CSV
        csv_files.sort(key=lambda x: os.path.getsize(x) if os.path.exists(x) else 0, reverse=True)
        file_path = csv_files[0]
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
        return df, file_path
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return None

@st.cache_data
def load_image():
    """Load the AI image from the local directory."""
    image_paths = ['Ai.png', 'AI.png', 'ai.png', 'AI_impact.png']
    for path in image_paths:
        if os.path.exists(path):
            return Image.open(path)
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.lower() in ['ai.png', 'ai_impact.png', 'ai_impact_on_jobs.png']:
                return Image.open(os.path.join(root, f))
    return None

# ---------- DATA PROCESSING ----------
def process_data(df):
    """Clean and prepare data."""
    # Rename columns to snake_case for convenience
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Ensure numeric columns
    num_cols = ['years_experience', 'ai_replacement_risk', 'future_demand_score', 
                'average_salary_usd', 'job_growth_2030', 'work_hours_per_week', 
                'performance_score', 'job_satisfaction']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing (if any) - simple median for numeric, mode for categorical
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        if df[col].isna().any():
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    return df

# ---------- HELPER FUNCTIONS ----------
def get_skills_freq(df, col='required_skills'):
    """Extract skills frequency from comma-separated string."""
    if col not in df.columns:
        return pd.Series()
    skills_series = df[col].dropna().astype(str)
    all_skills = []
    for s in skills_series:
        # Split by comma, strip, and filter out empty
        skills = [x.strip() for x in s.split(',') if x.strip()]
        all_skills.extend(skills)
    freq = pd.Series(Counter(all_skills)).sort_values(ascending=False)
    return freq

# ---------- MAIN APP ----------
def main():
    # Load data
    data = load_data()
    if data is None:
        st.error("❌ Failed to load data. Please ensure 'AI_Impact_on_Jobs_2030.csv' is in the current directory.")
        return
    df_raw, file_path = data
    st.success(f"✅ Data loaded successfully: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    
    # Process
    df = process_data(df_raw)
    
    # Load image
    ai_img = load_image()
    
    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.markdown("### 🤖 AI Impact on Jobs 2030")
        st.markdown("---")
        if ai_img is not None:
            st.image(ai_img, use_container_width=True, caption="🤖 AI & Future of Work")
        else:
            st.info("📷 Ai.png not found in the current directory.")
        st.markdown("---")
        st.markdown("### 📊 Dashboard Navigation")
        
        tabs = [
            "🏠 Overview",
            "🏢 Industry",
            "💼 Job Title",
            "🌍 Country",
            "💰 Salary",
            "🧠 AI Impact",
            "🛠️ Skills",
            "📈 ML Prediction"
        ]
        selected_tab = st.radio("", tabs, index=0, label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### 📋 Dataset Info")
        st.markdown(f"- **Records:** {len(df):,}")
        st.markdown(f"- **Features:** {df.shape[1]}")
        st.markdown(f"- **Job Titles:** {df['job_title'].nunique() if 'job_title' in df.columns else 'N/A'}")
        st.markdown(f"- **Industries:** {df['industry'].nunique() if 'industry' in df.columns else 'N/A'}")
        st.markdown(f"- **Countries:** {df['country'].nunique() if 'country' in df.columns else 'N/A'}")
        
        st.markdown("---")
        st.markdown("💡 **Tip:** Hover over charts for details. Use zoom/pan for deeper exploration.")
        st.markdown("---")
        st.markdown("© 2026 AI Jobs Dashboard")
    
    # ---------- MAIN CONTENT ----------
    # Define KPIs
    kpis = {}
    kpis['Total Records'] = f"{len(df):,}"
    kpis['Unique Job Titles'] = f"{df['job_title'].nunique():,}" if 'job_title' in df.columns else "N/A"
    kpis['Industries'] = f"{df['industry'].nunique():,}" if 'industry' in df.columns else "N/A"
    kpis['Countries'] = f"{df['country'].nunique():,}" if 'country' in df.columns else "N/A"
    if 'average_salary_usd' in df.columns:
        sal = df['average_salary_usd'].dropna()
        kpis['Avg Salary'] = f"${sal.mean():,.0f}"
        kpis['Median Salary'] = f"${sal.median():,.0f}"
    if 'ai_replacement_risk' in df.columns:
        kpis['Avg AI Risk'] = f"{df['ai_replacement_risk'].mean():.2f}"
    if 'future_demand_score' in df.columns:
        kpis['Avg Demand Score'] = f"{df['future_demand_score'].mean():.2f}"
    
    # Display KPIs (first 6)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    kpi_items = list(kpis.items())
    for i, (label, value) in enumerate(kpi_items[:6]):
        with [col1, col2, col3, col4, col5, col6][i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="display:flex;align-items:center;">
                    <span class="metric-value">{value}</span>
                </div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ---------- TAB CONTENT ----------
    if selected_tab == "🏠 Overview":
        st.markdown('<div class="section-header">📊 Executive Dashboard — Overview</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            # Industry distribution
            if 'industry' in df.columns:
                ind_counts = df['industry'].value_counts().reset_index()
                ind_counts.columns = ['Industry', 'Count']
                fig = px.bar(ind_counts.head(15), y='Industry', x='Count', orientation='h',
                             title='Top Industries by Job Count',
                             color='Count', color_continuous_scale='Blues',
                             template='plotly_dark')
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Salary distribution
            if 'average_salary_usd' in df.columns:
                fig = px.histogram(df, x='average_salary_usd', nbins=50,
                                   title='Average Salary Distribution',
                                   labels={'average_salary_usd': 'Salary (USD)'},
                                   template='plotly_dark', color_discrete_sequence=['#023E8A'])
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                fig.add_vline(x=df['average_salary_usd'].mean(), line_dash="dash", line_color="#FFB703",
                              annotation_text=f"Mean: ${df['average_salary_usd'].mean():,.0f}")
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            # AI replacement risk vs future demand scatter
            if 'ai_replacement_risk' in df.columns and 'future_demand_score' in df.columns:
                fig = px.scatter(df, x='ai_replacement_risk', y='future_demand_score',
                                 color='industry', hover_name='job_title',
                                 title='AI Risk vs Future Demand',
                                 labels={'ai_replacement_risk': 'AI Replacement Risk', 
                                         'future_demand_score': 'Future Demand Score'},
                                 template='plotly_dark', opacity=0.6)
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Remote work possibility pie
            if 'remote_work_possibility' in df.columns:
                rw_counts = df['remote_work_possibility'].value_counts().reset_index()
                rw_counts.columns = ['Remote Work', 'Count']
                fig = px.pie(rw_counts, values='Count', names='Remote Work',
                             title='Remote Work Possibility',
                             color_discrete_sequence=px.colors.sequential.Blues_r,
                             template='plotly_dark', hole=0.4)
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Additional: Hiring trend
        if 'hiring_trend_2026' in df.columns:
            trend_counts = df['hiring_trend_2026'].value_counts().reset_index()
            trend_counts.columns = ['Hiring Trend', 'Count']
            fig = px.bar(trend_counts, x='Hiring Trend', y='Count',
                         title='Hiring Trend 2026',
                         color='Hiring Trend', template='plotly_dark')
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    elif selected_tab == "🏢 Industry":
        st.markdown('<div class="section-header">🏢 Industry Intelligence</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            # Industry job count
            ind_counts = df['industry'].value_counts().reset_index()
            ind_counts.columns = ['Industry', 'Count']
            fig = px.bar(ind_counts, y='Industry', x='Count', orientation='h',
                         title='Job Count by Industry',
                         color='Count', color_continuous_scale='Teal',
                         template='plotly_dark')
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            # Average salary by industry
            if 'average_salary_usd' in df.columns:
                ind_sal = df.groupby('industry')['average_salary_usd'].mean().sort_values(ascending=False).reset_index()
                ind_sal.columns = ['Industry', 'Avg Salary']
                fig = px.bar(ind_sal, y='Industry', x='Avg Salary', orientation='h',
                             title='Average Salary by Industry',
                             color='Avg Salary', color_continuous_scale='Reds',
                             template='plotly_dark')
                fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # AI risk by industry
        if 'ai_replacement_risk' in df.columns:
            ind_risk = df.groupby('industry')['ai_replacement_risk'].mean().sort_values(ascending=False).reset_index()
            ind_risk.columns = ['Industry', 'Avg AI Risk']
            fig = px.bar(ind_risk.head(15), y='Industry', x='Avg AI Risk', orientation='h',
                         title='Average AI Replacement Risk by Industry',
                         color='Avg AI Risk', color_continuous_scale='Oranges',
                         template='plotly_dark')
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Future demand by industry
        if 'future_demand_score' in df.columns:
            ind_demand = df.groupby('industry')['future_demand_score'].mean().sort_values(ascending=False).reset_index()
            ind_demand.columns = ['Industry', 'Avg Demand Score']
            fig = px.bar(ind_demand.head(15), y='Industry', x='Avg Demand Score', orientation='h',
                         title='Average Future Demand Score by Industry',
                         color='Avg Demand Score', color_continuous_scale='Greens',
                         template='plotly_dark')
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    elif selected_tab == "💼 Job Title":
        st.markdown('<div class="section-header">💼 Job Title Intelligence</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            # Top job titles by count
            job_counts = df['job_title'].value_counts().reset_index()
            job_counts.columns = ['Job Title', 'Count']
            fig = px.bar(job_counts.head(15), y='Job Title', x='Count', orientation='h',
                         title='Top 15 Job Titles by Frequency',
                         color='Count', color_continuous_scale='Purples',
                         template='plotly_dark')
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            # Average salary by job title (top 15)
            if 'average_salary_usd' in df.columns:
                job_sal = df.groupby('job_title')['average_salary_usd'].mean().sort_values(ascending=False).reset_index()
                job_sal.columns = ['Job Title', 'Avg Salary']
                fig = px.bar(job_sal.head(15), y='Job Title', x='Avg Salary', orientation='h',
                             title='Top 15 Highest Paying Job Titles',
                             color='Avg Salary', color_continuous_scale='Reds',
                             template='plotly_dark')
                fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Job title AI risk vs demand
        if 'ai_replacement_risk' in df.columns and 'future_demand_score' in df.columns:
            job_risk_demand = df.groupby('job_title')[['ai_replacement_risk', 'future_demand_score']].mean().reset_index()
            fig = px.scatter(job_risk_demand, x='ai_replacement_risk', y='future_demand_score',
                             hover_name='job_title', text='job_title',
                             title='AI Risk vs Demand by Job Title',
                             labels={'ai_replacement_risk': 'AI Replacement Risk', 
                                     'future_demand_score': 'Future Demand Score'},
                             template='plotly_dark')
            fig.update_traces(textposition='top center', textfont=dict(size=8))
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    elif selected_tab == "🌍 Country":
        st.markdown('<div class="section-header">🌍 Country Intelligence</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            # Country distribution
            country_counts = df['country'].value_counts().reset_index()
            country_counts.columns = ['Country', 'Count']
            fig = px.bar(country_counts, y='Country', x='Count', orientation='h',
                         title='Job Count by Country',
                         color='Count', color_continuous_scale='Blues',
                         template='plotly_dark')
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            # Average salary by country
            if 'average_salary_usd' in df.columns:
                country_sal = df.groupby('country')['average_salary_usd'].mean().sort_values(ascending=False).reset_index()
                country_sal.columns = ['Country', 'Avg Salary']
                fig = px.bar(country_sal.head(15), y='Country', x='Avg Salary', orientation='h',
                             title='Average Salary by Country',
                             color='Avg Salary', color_continuous_scale='Reds',
                             template='plotly_dark')
                fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Remote work possibility by country
        if 'remote_work_possibility' in df.columns:
            rw_country = df.groupby(['country', 'remote_work_possibility']).size().reset_index(name='Count')
            fig = px.bar(rw_country, x='country', y='Count', color='remote_work_possibility',
                         title='Remote Work Possibility by Country',
                         template='plotly_dark', barmode='group')
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    elif selected_tab == "💰 Salary":
        st.markdown('<div class="section-header">💰 Salary Intelligence</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            # Salary distribution
            if 'average_salary_usd' in df.columns:
                fig = px.histogram(df, x='average_salary_usd', nbins=50,
                                   title='Salary Distribution',
                                   labels={'average_salary_usd': 'Salary (USD)'},
                                   template='plotly_dark', color_discrete_sequence=['#023E8A'])
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                fig.add_vline(x=df['average_salary_usd'].mean(), line_dash="dash", line_color="#FFB703",
                              annotation_text=f"Mean: ${df['average_salary_usd'].mean():,.0f}")
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Salary by education level
            if 'education_level' in df.columns:
                edu_sal = df.groupby('education_level')['average_salary_usd'].mean().sort_values(ascending=False).reset_index()
                edu_sal.columns = ['Education', 'Avg Salary']
                fig = px.bar(edu_sal, y='Education', x='Avg Salary', orientation='h',
                             title='Average Salary by Education Level',
                             color='Avg Salary', color_continuous_scale='Greens',
                             template='plotly_dark')
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            # Salary vs Experience
            if 'years_experience' in df.columns and 'average_salary_usd' in df.columns:
                fig = px.scatter(df, x='years_experience', y='average_salary_usd',
                                 color='industry', hover_name='job_title',
                                 title='Salary vs Years of Experience',
                                 labels={'years_experience': 'Years Experience', 
                                         'average_salary_usd': 'Salary (USD)'},
                                 template='plotly_dark', opacity=0.6)
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Salary by company size
            if 'company_size' in df.columns:
                size_sal = df.groupby('company_size')['average_salary_usd'].mean().sort_values(ascending=False).reset_index()
                size_sal.columns = ['Company Size', 'Avg Salary']
                fig = px.bar(size_sal, y='Company Size', x='Avg Salary', orientation='h',
                             title='Average Salary by Company Size',
                             color='Avg Salary', color_continuous_scale='Purples',
                             template='plotly_dark')
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Salary vs AI risk
        if 'ai_replacement_risk' in df.columns and 'average_salary_usd' in df.columns:
            fig = px.scatter(df, x='ai_replacement_risk', y='average_salary_usd',
                             color='automation_level', hover_name='job_title',
                             title='Salary vs AI Replacement Risk',
                             labels={'ai_replacement_risk': 'AI Replacement Risk', 
                                     'average_salary_usd': 'Salary (USD)'},
                             template='plotly_dark', opacity=0.6)
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    elif selected_tab == "🧠 AI Impact":
        st.markdown('<div class="section-header">🧠 AI Impact Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            # AI risk distribution
            if 'ai_replacement_risk' in df.columns:
                fig = px.histogram(df, x='ai_replacement_risk', nbins=30,
                                   title='AI Replacement Risk Distribution',
                                   labels={'ai_replacement_risk': 'Risk Score'},
                                   template='plotly_dark', color_discrete_sequence=['#E63946'])
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Automation level
            if 'automation_level' in df.columns:
                auto_counts = df['automation_level'].value_counts().reset_index()
                auto_counts.columns = ['Automation Level', 'Count']
                fig = px.pie(auto_counts, values='Count', names='Automation Level',
                             title='Automation Level Distribution',
                             color_discrete_sequence=px.colors.sequential.Reds_r,
                             template='plotly_dark', hole=0.4)
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            # AI risk vs future demand
            if 'ai_replacement_risk' in df.columns and 'future_demand_score' in df.columns:
                fig = px.scatter(df, x='ai_replacement_risk', y='future_demand_score',
                                 color='automation_level', hover_name='job_title',
                                 title='AI Risk vs Future Demand',
                                 labels={'ai_replacement_risk': 'AI Replacement Risk', 
                                         'future_demand_score': 'Future Demand Score'},
                                 template='plotly_dark', opacity=0.6)
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Upskilling needed
            if 'upskilling_needed' in df.columns:
                up_counts = df['upskilling_needed'].value_counts().reset_index()
                up_counts.columns = ['Upskilling Needed', 'Count']
                fig = px.pie(up_counts, values='Count', names='Upskilling Needed',
                             title='Upskilling Needed',
                             color_discrete_sequence=px.colors.sequential.Blues_r,
                             template='plotly_dark', hole=0.4)
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Correlation heatmap of numeric columns related to AI impact
        impact_cols = ['ai_replacement_risk', 'future_demand_score', 'average_salary_usd', 
                       'job_growth_2030', 'work_hours_per_week', 'performance_score', 'job_satisfaction']
        impact_cols = [c for c in impact_cols if c in df.columns]
        if len(impact_cols) >= 2:
            corr = df[impact_cols].corr()
            fig = px.imshow(corr, text_auto='.2f', title='Correlation Matrix (AI Impact)',
                            color_continuous_scale=['#E63946', '#161B22', '#06D6A0'],
                            zmin=-1, zmax=1, template='plotly_dark')
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    elif selected_tab == "🛠️ Skills":
        st.markdown('<div class="section-header">🛠️ Skills Intelligence</div>', unsafe_allow_html=True)
        
        # Extract skills
        if 'required_skills' in df.columns:
            skills_freq = get_skills_freq(df, 'required_skills')
            if not skills_freq.empty:
                col1, col2 = st.columns(2)
                with col1:
                    # Top skills
                    fig = px.bar(skills_freq.head(20).reset_index().head(20),
                                 y='index', x=0, orientation='h',
                                 title='Top 20 Required Skills',
                                 labels={'index': 'Skill', 0: 'Frequency'},
                                 template='plotly_dark',
                                 color=0, color_continuous_scale='Viridis')
                    fig.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                with col2:
                    # Word cloud alternative: bar chart of bottom top skills (just to have something)
                    # Show skills by average salary (if possible)
                    # For simplicity, show skills frequency by AI risk? not directly.
                    # Let's show skills that appear in high-paying jobs?
                    # Could be complex, so just show top skills again but with a different color.
                    fig = px.bar(skills_freq.head(20).reset_index().head(20),
                                 y='index', x=0, orientation='h',
                                 title='Top 20 Required Skills (Alternate)',
                                 labels={'index': 'Skill', 0: 'Frequency'},
                                 template='plotly_dark',
                                 color=0, color_continuous_scale='Plasma')
                    fig.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("No skills data found.")
        else:
            st.info("Column 'required_skills' not found.")
        
        # Skills vs Salary (if we can map skills to average salary of jobs containing that skill)
        if 'required_skills' in df.columns and 'average_salary_usd' in df.columns:
            # Compute average salary for each skill
            skill_salary = {}
            for idx, row in df.iterrows():
                skills = str(row['required_skills']).split(',')
                salary = row['average_salary_usd']
                for s in skills:
                    s = s.strip()
                    if s:
                        if s not in skill_salary:
                            skill_salary[s] = []
                        skill_salary[s].append(salary)
            import numpy as np
            skill_avg_sal = {k: np.mean(v) for k, v in skill_salary.items() if len(v) >= 5}
            if skill_avg_sal:
                df_skill_sal = pd.DataFrame(list(skill_avg_sal.items()), columns=['Skill', 'Avg Salary']).sort_values('Avg Salary', ascending=False)
                fig = px.bar(df_skill_sal.head(20), y='Skill', x='Avg Salary', orientation='h',
                             title='Top 20 Skills by Average Salary (min 5 occurrences)',
                             color='Avg Salary', color_continuous_scale='Reds',
                             template='plotly_dark')
                fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    elif selected_tab == "📈 ML Prediction":
        st.markdown('<div class="section-header">📈 Machine Learning — Salary Prediction</div>', unsafe_allow_html=True)
        
        st.info("""
        **Gradient Boosting Regressor** — trained to predict Average Salary from available features.
        Features: Years Experience, AI Replacement Risk, Future Demand Score, Automation Level, 
        Job Growth, Work Hours, Company Size, Performance Score, etc.
        """)
        
        # Prepare data for ML
        # We'll keep it simple: encode categorical features, train a model, and show evaluation.
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        # Select features
        feature_cols = ['years_experience', 'ai_replacement_risk', 'future_demand_score', 
                        'job_growth_2030', 'work_hours_per_week', 'performance_score', 
                        'job_satisfaction', 'automation_level', 'company_size', 
                        'education_level', 'industry', 'country']
        # Keep only those present
        feature_cols = [c for c in feature_cols if c in df.columns]
        target = 'average_salary_usd'
        if target not in df.columns:
            st.warning("Target column 'average_salary_usd' not found.")
        else:
            # Prepare data
            X = df[feature_cols].copy()
            y = df[target]
            
            # Encode categorical
            cat_cols = X.select_dtypes(include='object').columns
            for c in cat_cols:
                le = LabelEncoder()
                X[c] = le.fit_transform(X[c].astype(str))
            
            # Impute missing
            X = X.fillna(X.median())
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Pipeline
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42))
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("R² Score", f"{r2:.4f}")
            col2.metric("MAE", f"${mae:,.0f}")
            col3.metric("RMSE", f"${rmse:,.0f}")
            
            # Feature importance
            importances = pipe.named_steps['model'].feature_importances_
            feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
            
            st.markdown("### Feature Importances")
            fig = px.bar(feat_imp.head(10).reset_index(), y='index', x=0, orientation='h',
                         title='Top 10 Feature Importances',
                         labels={'index': 'Feature', 0: 'Importance'},
                         template='plotly_dark', color=0, color_continuous_scale='Blues')
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Actual vs Predicted
            fig = px.scatter(x=y_test, y=y_pred, title='Actual vs Predicted Salary',
                             labels={'x': 'Actual Salary ($)', 'y': 'Predicted Salary ($)'},
                             template='plotly_dark', opacity=0.5)
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            fig.add_shape(type='line', x0=lims[0], y0=lims[0], x1=lims[1], y1=lims[1],
                          line=dict(color='#FFB703', dash='dash'))
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # ---------- FOOTER ----------
    st.markdown("""
    <div class="footer">
        🤖 AI Impact on Jobs 2030 Dashboard &nbsp;·&nbsp; 
        Built with Streamlit & Plotly &nbsp;·&nbsp; 
        Data: AI_Impact_on_Jobs_2030
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()