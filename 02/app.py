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

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Car Market Intelligence Dashboard",
    page_icon="🚗",
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
    """Load the car market dataset from the local directory."""
    # Look for CSV files in the current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    # If no CSV in current dir, check subdirectories
    if not csv_files:
        for root, dirs, files in os.walk('.'):
            for f in files:
                if f.endswith('.csv'):
                    csv_files.append(os.path.join(root, f))
    
    if not csv_files:
        st.error("❌ No CSV files found in the current directory. Please ensure the dataset is present.")
        return None
    
    # Load the largest CSV file
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
    """Load the car image from the local directory."""
    image_paths = ['car.png', 'car.jpg', 'car.jpeg']
    for path in image_paths:
        if os.path.exists(path):
            return Image.open(path)
    
    # Check subdirectories
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.lower() in ['car.png', 'car.jpg', 'car.jpeg']:
                return Image.open(os.path.join(root, f))
    
    return None

# ---------- DATA PROCESSING FUNCTIONS ----------
def clean_column_names(df):
    """Clean column names to snake_case."""
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(r'[\s]+', '_', regex=True)
                  .str.replace(r'[^a-z0-9_]', '', regex=True))
    return df

def find_col(df, *candidates):
    """Find a column by candidate names."""
    for c in candidates:
        if c in df.columns:
            return c
        matches = [col for col in df.columns if c in col]
        if matches:
            return matches[0]
    return None

def coerce_numeric(df, exclude_hint=None):
    """Coerce string columns to numeric where possible."""
    exclude_hint = exclude_hint or []
    converted = []
    for c in df.select_dtypes(include='object').columns:
        if any(h in c for h in exclude_hint):
            continue
        cleaned = df[c].astype(str).str.replace(r'[\$,\s]', '', regex=True)
        numeric = pd.to_numeric(cleaned, errors='coerce')
        if numeric.notna().sum() >= 0.5 * df[c].notna().sum() and numeric.notna().sum() > 0:
            df[c] = numeric
            converted.append(c)
    return df, converted

def process_data(df):
    """Process the raw data for analysis."""
    df = clean_column_names(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Coerce numeric
    text_hints = ['name', 'make', 'model', 'trim', 'desc', 'type', 'fuel', 
                  'drive', 'transmission', 'body', 'color', 'style', 'cat', 
                  'class', 'origin']
    df, converted = coerce_numeric(df, exclude_hint=text_hints)
    
    # Identify key columns
    MSRP_COL = find_col(df, 'msrp', 'trim_msrp', 'price', 'retail_price', 'list_price')
    INVOICE_COL = find_col(df, 'invoice', 'trim_invoice', 'invoice_price', 'dealer_price')
    MAKE_COL = find_col(df, 'make_name', 'make', 'manufacturer', 'brand')
    MODEL_COL = find_col(df, 'model_name', 'model', 'vehicle_model')
    TRIM_COL = find_col(df, 'trim_name', 'trim', 'trim_level', 'variant')
    YEAR_COL = find_col(df, 'trim_year', 'year', 'model_year', 'vehicle_year')
    
    # Fill missing values
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for c in num_cols:
        if df[c].isna().any():
            df[c].fillna(df[c].median(), inplace=True)
    
    for c in cat_cols:
        if df[c].isna().any():
            df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else 'Unknown', inplace=True)
    
    # Create price tier
    if MSRP_COL:
        def price_tier(price):
            if price < 20000:
                return 'Budget (<$20K)'
            elif price < 40000:
                return 'Mid-Range ($20K-$40K)'
            elif price < 70000:
                return 'Premium ($40K-$70K)'
            elif price < 100000:
                return 'Luxury ($70K-$100K)'
            else:
                return 'Ultra-Luxury (>$100K)'
        df['price_tier'] = df[MSRP_COL].apply(price_tier)
    
    # Create price gap
    if MSRP_COL and INVOICE_COL:
        df['price_gap'] = df[MSRP_COL] - df[INVOICE_COL]
        df['margin_pct'] = (df['price_gap'] / df[MSRP_COL] * 100).round(2)
    
    return df, MSRP_COL, INVOICE_COL, MAKE_COL, MODEL_COL, TRIM_COL, YEAR_COL

# ---------- VISUALIZATION FUNCTIONS ----------
def create_kpi_cards(df, MSRP_COL, INVOICE_COL, MAKE_COL, MODEL_COL, TRIM_COL, YEAR_COL):
    """Create KPI metric cards."""
    kpis = {}
    kpis['Total Records'] = f"{len(df):,}"
    kpis['Manufacturers'] = f"{df[MAKE_COL].nunique():,}" if MAKE_COL else "N/A"
    kpis['Models'] = f"{df[MODEL_COL].nunique():,}" if MODEL_COL else "N/A"
    kpis['Trims'] = f"{df[TRIM_COL].nunique():,}" if TRIM_COL else "N/A"
    
    if MSRP_COL:
        msrp = df[MSRP_COL].dropna()
        kpis['Avg MSRP'] = f"${msrp.mean():,.0f}"
        kpis['Median MSRP'] = f"${msrp.median():,.0f}"
    
    if INVOICE_COL:
        inv = df[INVOICE_COL].dropna()
        kpis['Avg Invoice'] = f"${inv.mean():,.0f}"
    
    if MSRP_COL and INVOICE_COL:
        gap = df[MSRP_COL] - df[INVOICE_COL]
        kpis['Avg Price Gap'] = f"${gap.mean():,.0f}"
    
    if YEAR_COL:
        kpis['Year Range'] = f"{int(df[YEAR_COL].min())} – {int(df[YEAR_COL].max())}"
    
    return kpis

def plot_manufacturer_volume(df, MAKE_COL, top_n=20):
    """Plot manufacturer volume as bar chart."""
    if not MAKE_COL:
        return None
    counts = df[MAKE_COL].value_counts().head(top_n).reset_index()
    counts.columns = ['Manufacturer', 'Count']
    counts['Market Share %'] = (counts['Count'] / counts['Count'].sum() * 100).round(2)
    
    fig = px.bar(
        counts,
        y='Manufacturer',
        x='Count',
        orientation='h',
        title=f'Top {top_n} Manufacturers by Vehicle Count',
        color='Count',
        color_continuous_scale='Blues',
        text='Count',
        template='plotly_dark'
    )
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Vehicle Count',
        yaxis_title='',
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(22,27,34,0)',
        paper_bgcolor='rgba(22,27,34,0)',
        font=dict(color='#E6EDF3')
    )
    fig.update_traces(textposition='outside', textfont=dict(size=10, color='#E6EDF3'))
    return fig

def plot_manufacturer_pricing(df, MAKE_COL, MSRP_COL, top_n=20):
    """Plot manufacturer average MSRP."""
    if not MAKE_COL or not MSRP_COL:
        return None
    
    mfr_price = df.groupby(MAKE_COL)[MSRP_COL].agg(
        Avg_MSRP='mean', Count='count'
    ).round(0).reset_index()
    mfr_price = mfr_price[mfr_price['Count'] >= 3].sort_values('Avg_MSRP', ascending=False).head(top_n)
    
    overall_median = df[MSRP_COL].median()
    mfr_price['Segment'] = mfr_price['Avg_MSRP'].apply(
        lambda x: 'Luxury' if x > overall_median * 2 else
                  'Premium' if x > overall_median * 1.3 else
                  'Mid-Range' if x >= overall_median * 0.7 else 'Budget'
    )
    
    segment_colors = {'Luxury': '#FFB703', 'Premium': '#FB8500', 'Mid-Range': '#06D6A0', 'Budget': '#ADB5BD'}
    colors = [segment_colors.get(s, '#023E8A') for s in mfr_price['Segment']]
    
    fig = px.bar(
        mfr_price,
        y='Manufacturer',
        x='Avg_MSRP',
        orientation='h',
        title='Average MSRP by Manufacturer',
        color='Segment',
        color_discrete_map=segment_colors,
        text='Avg_MSRP',
        template='plotly_dark'
    )
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Average MSRP ($)',
        yaxis_title='',
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(22,27,34,0)',
        paper_bgcolor='rgba(22,27,34,0)',
        font=dict(color='#E6EDF3'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside', textfont=dict(size=10))
    return fig

def plot_price_tier_pie(df):
    """Plot price tier distribution as pie chart."""
    if 'price_tier' not in df.columns:
        return None
    
    tier_counts = df['price_tier'].value_counts().reset_index()
    tier_counts.columns = ['Tier', 'Count']
    
    tier_colors = {'Budget (<$20K)': '#06D6A0', 'Mid-Range ($20K-$40K)': '#023E8A',
                   'Premium ($40K-$70K)': '#FB8500', 'Luxury ($70K-$100K)': '#FFB703',
                   'Ultra-Luxury (>$100K)': '#7209B7'}
    
    fig = px.pie(
        tier_counts,
        values='Count',
        names='Tier',
        title='Market Price Tier Distribution',
        color='Tier',
        color_discrete_map=tier_colors,
        template='plotly_dark',
        hole=0.4
    )
    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(22,27,34,0)',
        paper_bgcolor='rgba(22,27,34,0)',
        font=dict(color='#E6EDF3'),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5)
    )
    fig.update_traces(textinfo='percent+label', textfont_size=11)
    return fig

def plot_msrp_distribution(df, MSRP_COL):
    """Plot MSRP distribution with histogram and KDE."""
    if not MSRP_COL:
        return None
    
    data = df[MSRP_COL].dropna()
    data_clip = data.clip(0, data.quantile(0.98))
    
    fig = px.histogram(
        data_clip,
        nbins=60,
        title='MSRP Distribution',
        labels={'value': 'MSRP ($)', 'count': 'Frequency'},
        template='plotly_dark',
        color_discrete_sequence=['#023E8A']
    )
    fig.add_vline(x=data.median(), line_dash="dash", line_color="#FFB703",
                  annotation_text=f"Median: ${data.median():,.0f}", annotation_position="top")
    fig.add_vline(x=data.mean(), line_dash="dash", line_color="#FB8500",
                  annotation_text=f"Mean: ${data.mean():,.0f}", annotation_position="bottom")
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(22,27,34,0)',
        paper_bgcolor='rgba(22,27,34,0)',
        font=dict(color='#E6EDF3'),
        xaxis=dict(tickprefix='$', tickformat=',.0f')
    )
    return fig

def plot_msrp_vs_invoice(df, MSRP_COL, INVOICE_COL):
    """Plot MSRP vs Invoice scatter."""
    if not MSRP_COL or not INVOICE_COL:
        return None
    
    fig = px.scatter(
        df,
        x=INVOICE_COL,
        y=MSRP_COL,
        title='MSRP vs Invoice Price',
        labels={INVOICE_COL: 'Invoice ($)', MSRP_COL: 'MSRP ($)'},
        template='plotly_dark',
        color_discrete_sequence=['#06D6A0'],
        opacity=0.4,
        trendline='ols'
    )
    max_val = max(df[MSRP_COL].max(), df[INVOICE_COL].max())
    fig.add_shape(
        type='line',
        x0=0, y0=0,
        x1=max_val, y1=max_val,
        line=dict(color='#FFB703', dash='dash', width=2)
    )
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(22,27,34,0)',
        paper_bgcolor='rgba(22,27,34,0)',
        font=dict(color='#E6EDF3'),
        xaxis=dict(tickprefix='$', tickformat=',.0f'),
        yaxis=dict(tickprefix='$', tickformat=',.0f')
    )
    return fig

def plot_price_gap(df):
    """Plot price gap distribution."""
    if 'price_gap' not in df.columns:
        return None
    
    data = df['price_gap'].dropna()
    data_clip = data.clip(data.quantile(0.01), data.quantile(0.99))
    
    fig = px.histogram(
        data_clip,
        nbins=50,
        title='Price Gap Distribution (MSRP – Invoice)',
        labels={'value': 'Price Gap ($)', 'count': 'Frequency'},
        template='plotly_dark',
        color_discrete_sequence=['#FB8500']
    )
    fig.add_vline(x=data.mean(), line_dash="dash", line_color="#FFB703",
                  annotation_text=f"Mean: ${data.mean():,.0f}", annotation_position="top")
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(22,27,34,0)',
        paper_bgcolor='rgba(22,27,34,0)',
        font=dict(color='#E6EDF3'),
        xaxis=dict(tickprefix='$', tickformat=',.0f')
    )
    return fig

def plot_gap_by_manufacturer(df, MAKE_COL):
    """Plot price gap by manufacturer."""
    if 'price_gap' not in df.columns or not MAKE_COL:
        return None
    
    gap_by_make = df.groupby(MAKE_COL)['price_gap'].mean().sort_values(ascending=False).head(20).reset_index()
    gap_by_make.columns = ['Manufacturer', 'Avg Price Gap']
    
    fig = px.bar(
        gap_by_make,
        y='Manufacturer',
        x='Avg Price Gap',
        orientation='h',
        title='Average Price Gap by Manufacturer (Top 20)',
        color='Avg Price Gap',
        color_continuous_scale='Reds',
        text='Avg Price Gap',
        template='plotly_dark'
    )
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Average Price Gap ($)',
        yaxis_title='',
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(22,27,34,0)',
        paper_bgcolor='rgba(22,27,34,0)',
        font=dict(color='#E6EDF3')
    )
    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    return fig

def plot_model_popularity(df, MODEL_COL, top_n=20):
    """Plot most popular models."""
    if not MODEL_COL:
        return None
    
    counts = df[MODEL_COL].value_counts().head(top_n).reset_index()
    counts.columns = ['Model', 'Count']
    
    fig = px.bar(
        counts,
        y='Model',
        x='Count',
        orientation='h',
        title=f'Top {top_n} Models by Trim Count',
        color='Count',
        color_continuous_scale='Teal',
        text='Count',
        template='plotly_dark'
    )
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Number of Trims',
        yaxis_title='',
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(22,27,34,0)',
        paper_bgcolor='rgba(22,27,34,0)',
        font=dict(color='#E6EDF3')
    )
    fig.update_traces(textposition='outside', textfont=dict(size=10))
    return fig

def plot_models_per_manufacturer(df, MAKE_COL, MODEL_COL):
    """Plot models per manufacturer."""
    if not MAKE_COL or not MODEL_COL:
        return None
    
    models_per_make = df.groupby(MAKE_COL)[MODEL_COL].nunique().sort_values(ascending=False).head(15).reset_index()
    models_per_make.columns = ['Manufacturer', 'Unique Models']
    
    fig = px.bar(
        models_per_make,
        y='Manufacturer',
        x='Unique Models',
        orientation='h',
        title='Most Diversified Manufacturers (Unique Models)',
        color='Unique Models',
        color_continuous_scale='Tealgrn',
        text='Unique Models',
        template='plotly_dark'
    )
    fig.update_layout(
        height=450,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Number of Unique Models',
        yaxis_title='',
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(22,27,34,0)',
        paper_bgcolor='rgba(22,27,34,0)',
        font=dict(color='#E6EDF3')
    )
    fig.update_traces(textposition='outside', textfont=dict(size=10))
    return fig

def plot_trim_popularity(df, TRIM_COL, top_n=20):
    """Plot most common trims."""
    if not TRIM_COL:
        return None
    
    counts = df[TRIM_COL].value_counts().head(top_n).reset_index()
    counts.columns = ['Trim', 'Count']
    
    fig = px.bar(
        counts,
        y='Trim',
        x='Count',
        orientation='h',
        title=f'Top {top_n} Most Common Trims',
        color='Count',
        color_continuous_scale='Purples',
        text='Count',
        template='plotly_dark'
    )
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Frequency',
        yaxis_title='',
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(22,27,34,0)',
        paper_bgcolor='rgba(22,27,34,0)',
        font=dict(color='#E6EDF3')
    )
    fig.update_traces(textposition='outside', textfont=dict(size=9))
    return fig

def plot_year_trend(df, YEAR_COL, MSRP_COL):
    """Plot year trend for vehicle count and average MSRP."""
    if not YEAR_COL:
        return None
    
    year_dist = df[YEAR_COL].dropna().value_counts().sort_index().reset_index()
    year_dist.columns = ['Year', 'Count']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar chart for count
    fig.add_trace(
        go.Bar(
            x=year_dist['Year'],
            y=year_dist['Count'],
            name='Vehicle Count',
            marker_color='#023E8A',
            opacity=0.8
        ),
        secondary_y=False
    )
    
    # Line chart for average MSRP
    if MSRP_COL:
        year_msrp = df.groupby(YEAR_COL)[MSRP_COL].mean().dropna().reset_index()
        year_msrp.columns = ['Year', 'Avg_MSRP']
        
        fig.add_trace(
            go.Scatter(
                x=year_msrp['Year'],
                y=year_msrp['Avg_MSRP'],
                name='Avg MSRP',
                mode='lines+markers',
                line=dict(color='#FFB703', width=2.5),
                marker=dict(size=6, color='#FFB703'),
                yaxis='y2'
            ),
            secondary_y=True
        )
    
    fig.update_layout(
        title='Vehicle Count & Average MSRP by Year',
        template='plotly_dark',
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(22,27,34,0)',
        paper_bgcolor='rgba(22,27,34,0)',
        font=dict(color='#E6EDF3'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    fig.update_yaxes(title_text="Vehicle Count", secondary_y=False, color='#8B949E')
    fig.update_yaxes(title_text="Avg MSRP ($)", secondary_y=True, color='#FFB703', tickprefix='$')
    fig.update_xaxes(title_text="Year")
    
    return fig

def plot_competitive_landscape(df, MAKE_COL, MSRP_COL, MODEL_COL):
    """Plot competitive landscape bubble chart."""
    if not MAKE_COL or not MSRP_COL:
        return None
    
    comp = df.groupby(MAKE_COL).agg(
        Avg_MSRP=(MSRP_COL, 'mean'),
        Count=(MSRP_COL, 'count')
    ).reset_index()
    
    if MODEL_COL:
        comp['Models'] = df.groupby(MAKE_COL)[MODEL_COL].nunique().values
    
    comp = comp[comp['Count'] >= 3].sort_values('Count', ascending=False)
    
    fig = px.scatter(
        comp,
        x='Count',
        y='Avg_MSRP',
        size='Count',
        color='Avg_MSRP',
        hover_name=MAKE_COL,
        text=MAKE_COL,
        title='Competitive Landscape — Volume vs Pricing',
        labels={'Count': 'Vehicle Count (Market Presence)', 'Avg_MSRP': 'Average MSRP ($)'},
        template='plotly_dark',
        color_continuous_scale='RdYlGn',
        size_max=50
    )
    fig.update_traces(
        textposition='top center',
        textfont=dict(size=10, color='#E6EDF3'),
        marker=dict(line=dict(width=1, color='#30363D'))
    )
    fig.update_layout(
        height=550,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(22,27,34,0)',
        paper_bgcolor='rgba(22,27,34,0)',
        font=dict(color='#E6EDF3'),
        xaxis=dict(gridcolor='#30363D'),
        yaxis=dict(gridcolor='#30363D', tickprefix='$', tickformat=',.0f')
    )
    return fig

def plot_correlation_heatmap(df, NUM_COLS):
    """Plot correlation heatmap."""
    if len(NUM_COLS) < 2:
        return None
    
    corr_cols = [c for c in NUM_COLS if c not in ['make_id', 'model_id', 'trim_id'] and df[c].notna().sum() > 20][:10]
    if len(corr_cols) < 2:
        return None
    
    corr_matrix = df[corr_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        title='Correlation Heatmap',
        color_continuous_scale=['#E63946', '#161B22', '#06D6A0'],
        zmin=-1, zmax=1,
        template='plotly_dark',
        aspect='auto'
    )
    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(22,27,34,0)',
        paper_bgcolor='rgba(22,27,34,0)',
        font=dict(color='#E6EDF3')
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))
    return fig

def plot_top_models_priced(df, MODEL_COL, MAKE_COL, MSRP_COL):
    """Plot highest and lowest priced models."""
    if not MODEL_COL or not MSRP_COL or not MAKE_COL:
        return None
    
    model_price = df.groupby([MAKE_COL, MODEL_COL])[MSRP_COL].agg(
        Avg_MSRP='mean', Count='count'
    ).round(0).reset_index()
    model_price.columns = ['Make', 'Model', 'Avg_MSRP', 'Count']
    model_price = model_price[model_price['Count'] >= 3]
    model_price['Label'] = model_price['Make'] + ' ' + model_price['Model']
    
    # Highest priced
    highest = model_price.nlargest(12, 'Avg_MSRP')
    fig_high = px.bar(
        highest,
        y='Label',
        x='Avg_MSRP',
        orientation='h',
        title='Highest Priced Models (Avg MSRP)',
        color='Avg_MSRP',
        color_continuous_scale='Oranges',
        text='Avg_MSRP',
        template='plotly_dark'
    )
    fig_high.update_layout(
        height=450,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Average MSRP ($)',
        yaxis_title='',
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(22,27,34,0)',
        paper_bgcolor='rgba(22,27,34,0)',
        font=dict(color='#E6EDF3')
    )
    fig_high.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    
    # Lowest priced
    lowest = model_price.nsmallest(12, 'Avg_MSRP')
    fig_low = px.bar(
        lowest,
        y='Label',
        x='Avg_MSRP',
        orientation='h',
        title='Lowest Priced Models (Avg MSRP)',
        color='Avg_MSRP',
        color_continuous_scale='Greens',
        text='Avg_MSRP',
        template='plotly_dark'
    )
    fig_low.update_layout(
        height=450,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Average MSRP ($)',
        yaxis_title='',
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(22,27,34,0)',
        paper_bgcolor='rgba(22,27,34,0)',
        font=dict(color='#E6EDF3')
    )
    fig_low.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    
    return fig_high, fig_low

def plot_trim_year_analysis(df, TRIM_COL, YEAR_COL, MSRP_COL):
    """Plot trim year analysis."""
    if not YEAR_COL or not TRIM_COL:
        return None
    
    year_dist = df[YEAR_COL].dropna().value_counts().sort_index().reset_index()
    year_dist.columns = ['Year', 'Count']
    
    fig = px.bar(
        year_dist,
        x='Year',
        y='Count',
        title='Vehicle Count by Model Year',
        labels={'Year': 'Year', 'Count': 'Count'},
        template='plotly_dark',
        color_discrete_sequence=['#023E8A']
    )
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(22,27,34,0)',
        paper_bgcolor='rgba(22,27,34,0)',
        font=dict(color='#E6EDF3'),
        xaxis=dict(tickangle=45)
    )
    return fig

# ---------- MAIN APP ----------
def main():
    # Load data
    data = load_data()
    if data is None:
        st.error("❌ Failed to load data. Please ensure the CSV dataset is in the current directory.")
        return
    
    df_raw, file_path = data
    st.success(f"✅ Data loaded successfully: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    
    # Process data
    df, MSRP_COL, INVOICE_COL, MAKE_COL, MODEL_COL, TRIM_COL, YEAR_COL = process_data(df_raw)
    
    # Get numeric columns
    NUM_COLS = df.select_dtypes(include=np.number).columns.tolist()
    
    # Load car image
    car_img = load_image()
    
    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.markdown("### 🚗 Car Market Intelligence")
        st.markdown("---")
        
        # Display car image
        if car_img is not None:
            st.image(car_img, use_container_width=True, caption="🚘 Automotive Market Analysis")
        else:
            st.info("📷 car.png not found in the current directory.")
        
        st.markdown("---")
        st.markdown("### 📊 Dashboard Navigation")
        
        tabs = [
            "🏠 Overview",
            "🏭 Manufacturer",
            "🚗 Model",
            "🔧 Trim",
            "💰 Pricing",
            "🌍 Market",
            "📈 Statistics",
            "🤖 ML Prediction"
        ]
        
        # Use radio buttons for navigation
        selected_tab = st.radio("", tabs, index=0, label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### 📋 Dataset Info")
        st.markdown(f"- **Records:** {len(df):,}")
        st.markdown(f"- **Features:** {df.shape[1]}")
        st.markdown(f"- **Manufacturers:** {df[MAKE_COL].nunique() if MAKE_COL else 'N/A'}")
        st.markdown(f"- **Models:** {df[MODEL_COL].nunique() if MODEL_COL else 'N/A'}")
        st.markdown(f"- **Trims:** {df[TRIM_COL].nunique() if TRIM_COL else 'N/A'}")
        
        st.markdown("---")
        st.markdown("### 🔍 Key Columns")
        if MSRP_COL:
            st.markdown(f"- **MSRP:** `{MSRP_COL}`")
        if INVOICE_COL:
            st.markdown(f"- **Invoice:** `{INVOICE_COL}`")
        if MAKE_COL:
            st.markdown(f"- **Make:** `{MAKE_COL}`")
        if MODEL_COL:
            st.markdown(f"- **Model:** `{MODEL_COL}`")
        if TRIM_COL:
            st.markdown(f"- **Trim:** `{TRIM_COL}`")
        if YEAR_COL:
            st.markdown(f"- **Year:** `{YEAR_COL}`")
        
        st.markdown("---")
        st.markdown("💡 **Tip:** Hover over charts for details. Use zoom/pan for deeper exploration.")
        
        st.markdown("---")
        st.markdown("© 2026 Car Market Intelligence")
    
    # ---------- MAIN CONTENT ----------
    # Get KPIs
    kpis = create_kpi_cards(df, MSRP_COL, INVOICE_COL, MAKE_COL, MODEL_COL, TRIM_COL, YEAR_COL)
    
    # ----- KPI ROW (shown on all pages) -----
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    kpi_items = list(kpis.items())
    cols = [col1, col2, col3, col4, col5, col6]
    
    for i, (label, value) in enumerate(kpi_items[:6]):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="display:flex;align-items:center;">
                    <span class="metric-value">{value}</span>
                </div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ----- TAB CONTENT -----
    if selected_tab == "🏠 Overview":
        st.markdown('<div class="section-header">📊 Executive Dashboard — Overview</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price tier pie
            fig_pie = plot_price_tier_pie(df)
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
            
            # MSRP distribution
            fig_msrp = plot_msrp_distribution(df, MSRP_COL)
            if fig_msrp:
                st.plotly_chart(fig_msrp, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            # Manufacturer volume
            fig_vol = plot_manufacturer_volume(df, MAKE_COL, top_n=15)
            if fig_vol:
                st.plotly_chart(fig_vol, use_container_width=True, config={'displayModeBar': False})
            
            # MSRP vs Invoice
            fig_scatter = plot_msrp_vs_invoice(df, MSRP_COL, INVOICE_COL)
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})
        
        # Year trend (full width)
        fig_year = plot_year_trend(df, YEAR_COL, MSRP_COL)
        if fig_year:
            st.plotly_chart(fig_year, use_container_width=True, config={'displayModeBar': False})
    
    elif selected_tab == "🏭 Manufacturer":
        st.markdown('<div class="section-header">🏭 Manufacturer Intelligence</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_vol = plot_manufacturer_volume(df, MAKE_COL, top_n=20)
            if fig_vol:
                st.plotly_chart(fig_vol, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            fig_price = plot_manufacturer_pricing(df, MAKE_COL, MSRP_COL, top_n=20)
            if fig_price:
                st.plotly_chart(fig_price, use_container_width=True, config={'displayModeBar': False})
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig_gap = plot_gap_by_manufacturer(df, MAKE_COL)
            if fig_gap:
                st.plotly_chart(fig_gap, use_container_width=True, config={'displayModeBar': False})
        
        with col4:
            fig_models = plot_models_per_manufacturer(df, MAKE_COL, MODEL_COL)
            if fig_models:
                st.plotly_chart(fig_models, use_container_width=True, config={'displayModeBar': False})
        
        # Manufacturer data table
        st.markdown('<div class="sub-header">📋 Manufacturer Summary</div>', unsafe_allow_html=True)
        if MAKE_COL:
            mfr_summary = df.groupby(MAKE_COL).agg(
                Count=(MAKE_COL, 'count'),
                Models=(MODEL_COL, 'nunique') if MODEL_COL else None,
                Avg_MSRP=(MSRP_COL, 'mean') if MSRP_COL else None,
                Median_MSRP=(MSRP_COL, 'median') if MSRP_COL else None
            ).round(0).sort_values('Count', ascending=False).reset_index()
            mfr_summary.columns = ['Manufacturer', 'Vehicle Count', 'Unique Models', 'Avg MSRP', 'Median MSRP']
            st.dataframe(mfr_summary.head(25), use_container_width=True, hide_index=True)
    
    elif selected_tab == "🚗 Model":
        st.markdown('<div class="section-header">🚗 Model Intelligence</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_model_pop = plot_model_popularity(df, MODEL_COL, top_n=20)
            if fig_model_pop:
                st.plotly_chart(fig_model_pop, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            fig_models_mfr = plot_models_per_manufacturer(df, MAKE_COL, MODEL_COL)
            if fig_models_mfr:
                st.plotly_chart(fig_models_mfr, use_container_width=True, config={'displayModeBar': False})
        
        # Highest and Lowest priced models
        figs = plot_top_models_priced(df, MODEL_COL, MAKE_COL, MSRP_COL)
        if figs:
            fig_high, fig_low = figs
            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(fig_high, use_container_width=True, config={'displayModeBar': False})
            with col4:
                st.plotly_chart(fig_low, use_container_width=True, config={'displayModeBar': False})
    
    elif selected_tab == "🔧 Trim":
        st.markdown('<div class="section-header">🔧 Trim-Level Intelligence</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_trim = plot_trim_popularity(df, TRIM_COL, top_n=20)
            if fig_trim:
                st.plotly_chart(fig_trim, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            fig_year = plot_trim_year_analysis(df, TRIM_COL, YEAR_COL, MSRP_COL)
            if fig_year:
                st.plotly_chart(fig_year, use_container_width=True, config={'displayModeBar': False})
        
        # Year trend with MSRP
        fig_year_trend = plot_year_trend(df, YEAR_COL, MSRP_COL)
        if fig_year_trend:
            st.plotly_chart(fig_year_trend, use_container_width=True, config={'displayModeBar': False})
        
        # Trim data
        if TRIM_COL:
            st.markdown('<div class="sub-header">📋 Top Trims</div>', unsafe_allow_html=True)
            trim_data = df[TRIM_COL].value_counts().head(30).reset_index()
            trim_data.columns = ['Trim', 'Frequency']
            st.dataframe(trim_data, use_container_width=True, hide_index=True)
    
    elif selected_tab == "💰 Pricing":
        st.markdown('<div class="section-header">💰 Pricing Intelligence</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_msrp = plot_msrp_distribution(df, MSRP_COL)
            if fig_msrp:
                st.plotly_chart(fig_msrp, use_container_width=True, config={'displayModeBar': False})
            
            fig_pie = plot_price_tier_pie(df)
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            fig_scatter = plot_msrp_vs_invoice(df, MSRP_COL, INVOICE_COL)
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})
            
            fig_gap = plot_price_gap(df)
            if fig_gap:
                st.plotly_chart(fig_gap, use_container_width=True, config={'displayModeBar': False})
        
        # Price gap by manufacturer
        fig_gap_mfr = plot_gap_by_manufacturer(df, MAKE_COL)
        if fig_gap_mfr:
            st.plotly_chart(fig_gap_mfr, use_container_width=True, config={'displayModeBar': False})
        
        # Pricing statistics
        st.markdown('<div class="sub-header">📊 Pricing Statistics</div>', unsafe_allow_html=True)
        if MSRP_COL:
            msrp_stats = df[MSRP_COL].describe().round(2).reset_index()
            msrp_stats.columns = ['Statistic', 'Value']
            msrp_stats['Value'] = msrp_stats['Value'].apply(lambda x: f'${x:,.2f}')
            st.dataframe(msrp_stats, use_container_width=True, hide_index=True)
    
    elif selected_tab == "🌍 Market":
        st.markdown('<div class="section-header">🌍 Market Intelligence</div>', unsafe_allow_html=True)
        
        # Competitive landscape
        fig_comp = plot_competitive_landscape(df, MAKE_COL, MSRP_COL, MODEL_COL)
        if fig_comp:
            st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_vol = plot_manufacturer_volume(df, MAKE_COL, top_n=15)
            if fig_vol:
                st.plotly_chart(fig_vol, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            fig_pie = plot_price_tier_pie(df)
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
        
        # Market concentration
        st.markdown('<div class="sub-header">📈 Market Concentration</div>', unsafe_allow_html=True)
        if MAKE_COL:
            shares = (df[MAKE_COL].value_counts() / len(df) * 100)
            hhi = (shares ** 2).sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Herfindahl-Hirschman Index (HHI)", f"{hhi:.1f}")
            with col2:
                if hhi < 1000:
                    conc = "Competitive"
                    color = "#06D6A0"
                elif hhi < 2500:
                    conc = "Moderately Concentrated"
                    color = "#FFB703"
                else:
                    conc = "Highly Concentrated"
                    color = "#E63946"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display:flex;align-items:center;">
                        <span style="font-size:24px;font-weight:700;color:{color};">{conc}</span>
                    </div>
                    <div class="metric-label">Market Concentration</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.metric("Manufacturers", f"{df[MAKE_COL].nunique():,}")
    
    elif selected_tab == "📈 Statistics":
        st.markdown('<div class="section-header">📈 Statistical Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_corr = plot_correlation_heatmap(df, NUM_COLS)
            if fig_corr:
                st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            # MSRP boxplot
            if MSRP_COL:
                fig_box = px.box(
                    df,
                    y=MSRP_COL,
                    title='MSRP Box Plot',
                    template='plotly_dark',
                    color_discrete_sequence=['#023E8A']
                )
                fig_box.update_layout(
                    height=450,
                    margin=dict(l=10, r=10, t=50, b=10),
                    plot_bgcolor='rgba(22,27,34,0)',
                    paper_bgcolor='rgba(22,27,34,0)',
                    font=dict(color='#E6EDF3'),
                    yaxis=dict(tickprefix='$', tickformat=',.0f')
                )
                st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})
        
        # Descriptive statistics
        st.markdown('<div class="sub-header">📊 Descriptive Statistics — All Numeric Columns</div>', unsafe_allow_html=True)
        if NUM_COLS:
            desc = df[NUM_COLS].describe().T
            desc['skewness'] = df[NUM_COLS].skew()
            desc['kurtosis'] = df[NUM_COLS].kurtosis()
            st.dataframe(desc, use_container_width=True)
        
        # Distribution plots for numeric columns
        st.markdown('<div class="sub-header">📊 Distribution of Numeric Variables</div>', unsafe_allow_html=True)
        plot_cols = [c for c in NUM_COLS if c not in ['make_id', 'model_id', 'trim_id'] and df[c].notna().sum() > 10][:9]
        if plot_cols:
            cols = st.columns(3)
            for i, c in enumerate(plot_cols):
                with cols[i % 3]:
                    data = df[c].dropna().clip(df[c].quantile(0.01), df[c].quantile(0.99))
                    fig = px.histogram(
                        data,
                        nbins=30,
                        title=c.replace('_', ' ').title(),
                        template='plotly_dark',
                        color_discrete_sequence=['#7209B7'],
                        opacity=0.8
                    )
                    fig.update_layout(
                        height=250,
                        margin=dict(l=5, r=5, t=30, b=5),
                        plot_bgcolor='rgba(22,27,34,0)',
                        paper_bgcolor='rgba(22,27,34,0)',
                        font=dict(color='#E6EDF3', size=10),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    elif selected_tab == "🤖 ML Prediction":
        st.markdown('<div class="section-header">🤖 Machine Learning — MSRP Prediction</div>', unsafe_allow_html=True)
        
        st.info("""
        **Gradient Boosting Regressor** — trained to predict vehicle MSRP from available features.
        The model uses manufacturer, model, trim, year, and engineered features.
        """)
        
        # Feature engineering
        df_ml = df.copy()
        new_features = []
        
        if MSRP_COL and INVOICE_COL:
            df_ml['price_gap'] = df_ml[MSRP_COL] - df_ml[INVOICE_COL]
            df_ml['margin_pct'] = (df_ml['price_gap'] / df_ml[MSRP_COL] * 100).round(4)
            new_features += ['price_gap', 'margin_pct']
        
        if YEAR_COL:
            CURRENT_YEAR = 2025
            df_ml['vehicle_age'] = CURRENT_YEAR - df_ml[YEAR_COL].fillna(CURRENT_YEAR)
            new_features.append('vehicle_age')
        
        if MAKE_COL and MSRP_COL:
            make_avg = df_ml.groupby(MAKE_COL)[MSRP_COL].transform('mean')
            make_std = df_ml.groupby(MAKE_COL)[MSRP_COL].transform('std').fillna(1)
            df_ml['make_avg_msrp'] = make_avg
            df_ml['make_price_index'] = (df_ml[MSRP_COL] - make_avg) / make_std.replace(0, 1)
            new_features += ['make_avg_msrp', 'make_price_index']
        
        if MODEL_COL and MSRP_COL:
            model_avg = df_ml.groupby(MODEL_COL)[MSRP_COL].transform('mean')
            df_ml['model_avg_msrp'] = model_avg
            new_features.append('model_avg_msrp')
        
        # Encode categorical columns
        from sklearn.preprocessing import LabelEncoder
        cat_encode_cols = [c for c in [MAKE_COL, MODEL_COL, TRIM_COL] if c]
        for c in cat_encode_cols:
            if c in df_ml.columns:
                le = LabelEncoder()
                df_ml[f'{c}_enc'] = le.fit_transform(df_ml[c].astype(str).fillna('Unknown'))
                new_features.append(f'{c}_enc')
        
        # Prepare features
        candidate_features = [f for f in new_features if f in df_ml.columns and f != MSRP_COL]
        id_hints = ['_id', 'make_id', 'model_id', 'trim_id']
        for c in NUM_COLS:
            if c != MSRP_COL and c not in candidate_features and not any(h in c for h in id_hints):
                candidate_features.append(c)
        
        leak_cols = [INVOICE_COL, 'price_gap', 'margin_pct'] if INVOICE_COL else ['price_gap', 'margin_pct']
        candidate_features = [c for c in candidate_features if c not in leak_cols and c in df_ml.columns]
        
        # Remove any features with all NaN or single value
        valid_features = []
        for c in candidate_features:
            if df_ml[c].notna().sum() > 10 and df_ml[c].nunique() > 1:
                valid_features.append(c)
        candidate_features = valid_features
        
        st.markdown(f"**Selected Features:** {len(candidate_features)} features")
        with st.expander("View features"):
            st.write(candidate_features)
        
        # Prepare data
        ml_df = df_ml[candidate_features + [MSRP_COL]].dropna(subset=[MSRP_COL])
        ml_df = ml_df.fillna(ml_df.median(numeric_only=True))
        
        X = ml_df[candidate_features]
        y = ml_df[MSRP_COL]
        
        st.markdown(f"**Dataset:** {X.shape[0]:,} samples × {X.shape[1]} features")
        
        if len(X) >= 20:
            from sklearn.model_selection import train_test_split, cross_val_score, KFold
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            import numpy as np
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42
            )
            
            with st.spinner("Training model..."):
                # Pipeline
                pipe_gbm = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('model', GradientBoostingRegressor(
                        n_estimators=200, learning_rate=0.07, max_depth=4,
                        subsample=0.85, min_samples_leaf=5, random_state=42))
                ])
                
                # Cross-validation
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_r2 = cross_val_score(pipe_gbm, X_train, y_train, cv=kf, scoring='r2')
                cv_rmse = np.sqrt(-cross_val_score(pipe_gbm, X_train, y_train, cv=kf, scoring='neg_mean_squared_error'))
                
                # Fit and evaluate
                pipe_gbm.fit(X_train, y_train)
                y_pred = pipe_gbm.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.nan))) * 100
                
                feat_imp = pd.Series(
                    pipe_gbm.named_steps['model'].feature_importances_,
                    index=candidate_features
                ).sort_values(ascending=False)
            
            # Results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{r2:.4f}", delta=f"{r2*100:.1f}% variance")
            with col2:
                st.metric("MAE", f"${mae:,.0f}")
            with col3:
                st.metric("RMSE", f"${rmse:,.0f}")
            with col4:
                st.metric("MAPE", f"{mape:.2f}%")
            
            st.markdown(f"**CV R²:** {cv_r2.mean():.4f} ± {cv_r2.std():.4f}  |  **CV RMSE:** ${cv_rmse.mean():,.0f} ± ${cv_rmse.std():,.0f}")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Actual vs Predicted
                fig_actual = px.scatter(
                    x=y_test, y=y_pred,
                    title='Actual vs Predicted MSRP',
                    labels={'x': 'Actual MSRP ($)', 'y': 'Predicted MSRP ($)'},
                    template='plotly_dark',
                    color_discrete_sequence=['#06D6A0'],
                    opacity=0.5
                )
                lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
                fig_actual.add_shape(
                    type='line', x0=lims[0], y0=lims[0],
                    x1=lims[1], y1=lims[1],
                    line=dict(color='#FFB703', dash='dash', width=2)
                )
                fig_actual.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=50, b=10),
                    plot_bgcolor='rgba(22,27,34,0)',
                    paper_bgcolor='rgba(22,27,34,0)',
                    font=dict(color='#E6EDF3'),
                    xaxis=dict(tickprefix='$', tickformat=',.0f'),
                    yaxis=dict(tickprefix='$', tickformat=',.0f')
                )
                st.plotly_chart(fig_actual, use_container_width=True, config={'displayModeBar': False})
            
            with col2:
                # Residual plot
                residuals = y_test - y_pred
                fig_res = px.scatter(
                    x=y_pred, y=residuals,
                    title='Residual Plot',
                    labels={'x': 'Predicted MSRP ($)', 'y': 'Residual ($)'},
                    template='plotly_dark',
                    color_discrete_sequence=['#FB8500'],
                    opacity=0.5
                )
                fig_res.add_hline(y=0, line_dash="dash", line_color="#FFB703")
                fig_res.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=50, b=10),
                    plot_bgcolor='rgba(22,27,34,0)',
                    paper_bgcolor='rgba(22,27,34,0)',
                    font=dict(color='#E6EDF3'),
                    xaxis=dict(tickprefix='$', tickformat=',.0f'),
                    yaxis=dict(tickprefix='$', tickformat=',.0f')
                )
                st.plotly_chart(fig_res, use_container_width=True, config={'displayModeBar': False})
            
            # Feature importance
            st.markdown('<div class="sub-header">🔑 Feature Importances</div>', unsafe_allow_html=True)
            
            fig_imp = px.bar(
                feat_imp.head(12).reset_index(),
                y='index',
                x=0,
                orientation='h',
                title='Top 12 Feature Importances',
                labels={'index': 'Feature', 0: 'Importance'},
                template='plotly_dark',
                color=0,
                color_continuous_scale='Blues',
                text=0
            )
            fig_imp.update_layout(
                height=450,
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title='Importance Score',
                yaxis_title='',
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=50, b=10),
                plot_bgcolor='rgba(22,27,34,0)',
                paper_bgcolor='rgba(22,27,34,0)',
                font=dict(color='#E6EDF3')
            )
            fig_imp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})
            
            # Residual distribution
            st.markdown('<div class="sub-header">📊 Residual Analysis</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    residuals,
                    nbins=40,
                    title='Residual Distribution',
                    labels={'value': 'Residual ($)'},
                    template='plotly_dark',
                    color_discrete_sequence=['#7209B7']
                )
                fig_hist.add_vline(x=residuals.mean(), line_dash="dash", line_color="#FFB703",
                                   annotation_text=f"Mean: ${residuals.mean():,.0f}")
                fig_hist.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=50, b=10),
                    plot_bgcolor='rgba(22,27,34,0)',
                    paper_bgcolor='rgba(22,27,34,0)',
                    font=dict(color='#E6EDF3')
                )
                st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
            
            with col2:
                # Q-Q plot
                from scipy import stats
                (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist='norm')
                fig_qq = px.scatter(
                    x=osm, y=osr,
                    title=f'Q-Q Plot — Residuals (r={r:.4f})',
                    labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'},
                    template='plotly_dark',
                    color_discrete_sequence=['#06D6A0'],
                    opacity=0.6
                )
                line_x = np.array([min(osm), max(osm)])
                line_y = slope * line_x + intercept
                fig_qq.add_scatter(
                    x=line_x, y=line_y,
                    mode='lines',
                    line=dict(color='#FFB703', width=2),
                    name='Reference'
                )
                fig_qq.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=50, b=10),
                    plot_bgcolor='rgba(22,27,34,0)',
                    paper_bgcolor='rgba(22,27,34,0)',
                    font=dict(color='#E6EDF3'),
                    showlegend=False
                )
                st.plotly_chart(fig_qq, use_container_width=True, config={'displayModeBar': False})
        else:
            st.warning("⚠️ Insufficient data for ML training (need ≥ 20 samples).")
    
    # ---------- FOOTER ----------
    st.markdown("""
    <div class="footer">
        🚗 Comprehensive Car Market Intelligence Dashboard &nbsp;·&nbsp; 
        Built with Streamlit & Plotly &nbsp;·&nbsp; 
        Data: Comprehensive Car Market Analysis
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()