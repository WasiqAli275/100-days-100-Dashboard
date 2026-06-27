import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# ──────────────────────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET (mirrors notebook)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=10000, seed=42):
    """Generate synthetic student performance dataset."""
    np.random.seed(seed)
    
    study_hours = np.clip(np.random.normal(5.0, 2.0, n), 0, 12)
    attendance_pct = np.clip(np.random.normal(75.0, 15.0, n), 20, 100)
    sleep_hours = np.clip(np.random.normal(7.0, 1.5, n), 3, 12)
    internet_usage = np.clip(np.random.normal(4.0, 2.0, n), 0, 12)
    assignments_comp = np.clip(np.random.normal(70.0, 20.0, n), 0, 100)
    previous_scores = np.clip(np.random.normal(65.0, 15.0, n), 20, 100)
    
    exam_score = (
        0.30 * study_hours * 5
        + 0.25 * attendance_pct * 0.5
        + 0.25 * previous_scores * 0.5
        + 0.10 * assignments_comp * 0.4
        - 0.05 * internet_usage * 2
        + np.random.normal(0, 5, n)
    )
    exam_score = np.clip(exam_score, 20, 100)
    
    placement_prob = 1 / (1 + np.exp(-(exam_score - 60) / 8))
    placement = np.where(np.random.rand(n) < placement_prob, 'Placed', 'Not Placed')
    
    df = pd.DataFrame({
        'Study Hours': study_hours,
        'Attendance Percentage': attendance_pct,
        'Sleep Hours': sleep_hours,
        'Internet Usage': internet_usage,
        'Assignments Completed': assignments_comp,
        'Previous Scores': previous_scores,
        'Exam Score': exam_score,
        'Placement Status': placement
    })
    return df

# ──────────────────────────────────────────────────────────────
# 2. HELPER FUNCTIONS FOR PLOTS
# ──────────────────────────────────────────────────────────────
def plot_histogram(df, col, title, color='#667eea', xlabel=''):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[col],
        marker_color=color,
        opacity=0.75,
        hovertemplate='%{x:.1f} → %{y} students<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, weight='bold')),
        xaxis_title=xlabel or col,
        yaxis_title='Count',
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI, sans-serif')
    )
    return fig

def plot_scatter(df, x_col, y_col, title, color='#667eea'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers',
        marker=dict(color=color, size=5, opacity=0.5),
        hovertemplate=f'{x_col}: %{{x:.1f}}<br>{y_col}: %{{y:.1f}}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, weight='bold')),
        xaxis_title=x_col,
        yaxis_title=y_col,
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI, sans-serif')
    )
    return fig

def plot_pie(df, col, title, colors=['#3BB273', '#E84855']):
    counts = df[col].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        marker=dict(colors=colors),
        textinfo='percent+label',
        textposition='inside',
        hole=0.4,
        hovertemplate='%{label}<br>%{value} students (%{percent})<extra></extra>'
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, weight='bold')),
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        font=dict(family='Segoe UI, sans-serif')
    )
    return fig

def plot_heatmap(df, cols, title):
    corr = df[cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=[
            [0, '#2E86AB'],
            [0.5, '#f7f7f7'],
            [1, '#E84855']
        ],
        zmin=-1,
        zmax=1,
        text=corr.round(2).values,
        texttemplate='%{text}',
        textfont=dict(size=9),
        hovertemplate='%{x} ↔ %{y}<br>r = %{z:.2f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, weight='bold')),
        margin=dict(l=80, r=20, t=40, b=60),
        xaxis=dict(tickangle=-30),
        yaxis=dict(autorange='reversed'),
        font=dict(family='Segoe UI, sans-serif', size=10)
    )
    return fig

def plot_bar_horizontal(df, col, title, color='#667eea', limit=10):
    counts = df[col].value_counts().head(limit)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts.values,
        y=counts.index,
        orientation='h',
        marker_color=color,
        opacity=0.8,
        hovertemplate='%{y}: %{x} students<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, weight='bold')),
        xaxis_title='Count',
        yaxis=dict(tickfont=dict(size=10), automargin=True),
        margin=dict(l=10, r=20, t=40, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI, sans-serif')
    )
    return fig

def plot_feature_importance(df):
    # Use absolute correlation with 'Exam Score'
    cols = ['Study Hours', 'Attendance Percentage', 'Sleep Hours', 'Internet Usage', 'Assignments Completed', 'Previous Scores']
    corr = df[cols + ['Exam Score']].corr()
    imp = corr['Exam Score'][cols].abs().sort_values(ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=imp.values,
        y=imp.index,
        orientation='h',
        marker_color='#F18F01',
        opacity=0.8,
        hovertemplate='%{y}: %{x:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text='Feature Importance (|Correlation with Exam Score|)', font=dict(size=14, weight='bold')),
        xaxis_title='Absolute Correlation',
        yaxis=dict(tickfont=dict(size=10), automargin=True),
        margin=dict(l=10, r=20, t=40, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI, sans-serif')
    )
    return fig, imp.index[0] if len(imp) > 0 else ''

# ──────────────────────────────────────────────────────────────
# 3. STREAMLIT APP
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Custom CSS to mimic card style
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 18px;
        padding: 25px 30px;
        margin-bottom: 30px;
        box-shadow: 0 10px 35px rgba(102, 126, 234, 0.25);
        display: flex;
        align-items: center;
        gap: 20px;
        flex-wrap: wrap;
    }
    .main-header img {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #fff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3436;
        margin: 0;
    }
    .main-header h1 span {
        color: #667eea;
    }
    .main-header p {
        font-size: 1.05rem;
        color: #4a4a4a;
        margin: 5px 0 0 0;
    }
    .main-header .badge {
        display: inline-block;
        background: rgba(102, 126, 234, 0.25);
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-top: 6px;
    }
    .stat-card {
        background: #fff;
        border-radius: 14px;
        padding: 15px 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.04);
        transition: transform 0.2s;
    }
    .stat-card:hover {
        transform: translateY(-3px);
    }
    .stat-card .number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-card .label {
        font-size: 0.85rem;
        color: #7f8c8d;
        margin-top: 4px;
    }
    .card {
        background: #fff;
        border-radius: 14px;
        padding: 18px 18px 12px 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.04);
        transition: box-shadow 0.2s;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .card:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.10);
    }
    .card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }
    .card-header .icon {
        font-size: 1.6rem;
        flex-shrink: 0;
    }
    .card-header h3 {
        font-size: 0.95rem;
        font-weight: 600;
        color: #2d3436;
        margin: 0;
        line-height: 1.3;
    }
    .card-header h3 .highlight {
        color: #667eea;
    }
    .card-footer {
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid #f0f0f0;
        font-size: 0.82rem;
        color: #7f8c8d;
        line-height: 1.5;
    }
    .card-footer strong {
        color: #2d3436;
    }
    .footer {
        text-align: center;
        padding: 20px 0 10px 0;
        font-size: 0.9rem;
        color: #7f8c8d;
        border-top: 1px solid #e9ecef;
        margin-top: 20px;
    }
    .footer span {
        color: #667eea;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load data
df = generate_dataset()

# Header
col1, col2 = st.columns([1, 5])
with col1:
    # Try to load image
    if os.path.exists("student.png"):
        st.image("student.png", width=80)
    else:
        st.markdown("📸")
with col2:
    st.markdown("""
    <div>
        <h1>📊 <span>Student Performance</span> Intelligence Dashboard</h1>
        <p>Comprehensive analysis of student behavior, academic performance, and placement outcomes.</p>
        <span class="badge">🎯 9 Key Insights &bull; Interactive Visualizations</span>
    </div>
    """, unsafe_allow_html=True)

# Stats row
total = len(df)
placed = df[df['Placement Status'] == 'Placed'].shape[0]
avg_score = df['Exam Score'].mean()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="stat-card"><div class="number">{total:,}</div><div class="label">Total Records</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-card"><div class="number">7</div><div class="label">Features</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="stat-card"><div class="number">{placed}</div><div class="label">Placed</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="stat-card"><div class="number">{avg_score:.1f}</div><div class="label">Avg Exam Score</div></div>', unsafe_allow_html=True)

# Helper to create a card
def create_card(icon, title, plot_func, footer_html, key):
    with st.container():
        st.markdown(f"""
        <div class="card">
            <div class="card-header">
                <span class="icon">{icon}</span>
                <h3>{title}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Place plot
        fig = plot_func()
        st.plotly_chart(fig, use_container_width=True, key=key)
        # Footer
        st.markdown(f'<div class="card-footer">{footer_html}</div>', unsafe_allow_html=True)

# Compute stats for footers
def compute_stats():
    stats = {}
    # Exam score
    exam = df['Exam Score']
    stats['exam_mean'] = exam.mean()
    stats['exam_std'] = exam.std()
    # Placement counts
    stats['placed_count'] = placed
    stats['not_placed'] = total - placed
    # Correlations
    stats['corr_study_exam'] = df['Study Hours'].corr(df['Exam Score'])
    stats['corr_attend_exam'] = df['Attendance Percentage'].corr(df['Exam Score'])
    stats['corr_prev_exam'] = df['Previous Scores'].corr(df['Exam Score'])
    # Sleep
    stats['sleep_mean'] = df['Sleep Hours'].mean()
    stats['sleep_median'] = df['Sleep Hours'].median()
    # Internet
    stats['internet_mean'] = df['Internet Usage'].mean()
    stats['internet_median'] = df['Internet Usage'].median()
    # Top feature
    cols = ['Study Hours', 'Attendance Percentage', 'Sleep Hours', 'Internet Usage', 'Assignments Completed', 'Previous Scores']
    corr = df[cols + ['Exam Score']].corr()
    imp = corr['Exam Score'][cols].abs().sort_values(ascending=False)
    stats['top_feature'] = imp.index[0] if len(imp) > 0 else ''
    return stats

stats = compute_stats()

# Define card configurations
cards = [
    {
        "icon": "📈",
        "title": "Exam Score Distribution",
        "plot": lambda: plot_histogram(df, 'Exam Score', 'Exam Score Distribution', '#667eea', 'Exam Score'),
        "footer": f"📊 <strong>Mean:</strong> {stats['exam_mean']:.1f} &nbsp;|&nbsp; <strong>Std:</strong> {stats['exam_std']:.1f}",
        "key": "hist_exam"
    },
    {
        "icon": "🎯",
        "title": "Placement Status",
        "plot": lambda: plot_pie(df, 'Placement Status', 'Placement Status', ['#3BB273', '#E84855']),
        "footer": f"✅ <strong>Placed:</strong> {stats['placed_count']} &nbsp;|&nbsp; ❌ <strong>Not Placed:</strong> {stats['not_placed']}",
        "key": "pie_placement"
    },
    {
        "icon": "📚",
        "title": "Study Hours vs Exam Score",
        "plot": lambda: plot_scatter(df, 'Study Hours', 'Exam Score', 'Study Hours → Exam Score', '#2E86AB'),
        "footer": f"⏱️ <strong>Correlation:</strong> {stats['corr_study_exam']:.3f}",
        "key": "scatter_study"
    },
    {
        "icon": "🏫",
        "title": "Attendance vs Exam Score",
        "plot": lambda: plot_scatter(df, 'Attendance Percentage', 'Exam Score', 'Attendance → Exam Score', '#17A398'),
        "footer": f"📊 <strong>Correlation:</strong> {stats['corr_attend_exam']:.3f}",
        "key": "scatter_attend"
    },
    {
        "icon": "📖",
        "title": "Previous Scores vs Exam Score",
        "plot": lambda: plot_scatter(df, 'Previous Scores', 'Exam Score', 'Previous Scores → Exam Score', '#7B2D8B'),
        "footer": f"📈 <strong>Correlation:</strong> {stats['corr_prev_exam']:.3f}",
        "key": "scatter_prev"
    },
    {
        "icon": "🔥",
        "title": "Feature Correlation Heatmap",
        "plot": lambda: plot_heatmap(df, ['Study Hours', 'Attendance Percentage', 'Sleep Hours', 'Internet Usage', 'Assignments Completed', 'Previous Scores', 'Exam Score'], 'Correlation Heatmap'),
        "footer": "🔍 <strong>Key:</strong> Dark blue = negative, white = neutral, red = positive",
        "key": "heatmap"
    },
    {
        "icon": "⚖️",
        "title": "Feature Importance (Exam Score)",
        "plot": lambda: plot_feature_importance(df)[0],
        "footer": f"🏆 <strong>Top predictor:</strong> {stats['top_feature']}",
        "key": "importance"
    },
    {
        "icon": "😴",
        "title": "Sleep Hours Distribution",
        "plot": lambda: plot_histogram(df, 'Sleep Hours', 'Sleep Hours Distribution', '#A29BFE', 'Sleep Hours'),
        "footer": f"💤 <strong>Mean:</strong> {stats['sleep_mean']:.1f} &nbsp;|&nbsp; <strong>Median:</strong> {stats['sleep_median']:.1f}",
        "key": "hist_sleep"
    },
    {
        "icon": "🌐",
        "title": "Internet Usage Distribution",
        "plot": lambda: plot_histogram(df, 'Internet Usage', 'Internet Usage Distribution', '#00B894', 'Internet Usage (hrs)'),
        "footer": f"🌍 <strong>Mean:</strong> {stats['internet_mean']:.1f} &nbsp;|&nbsp; <strong>Median:</strong> {stats['internet_median']:.1f}",
        "key": "hist_internet"
    }
]

# Arrange cards in a grid: 3 columns
rows = [cards[i:i+3] for i in range(0, len(cards), 3)]
for row in rows:
    cols = st.columns(3)
    for idx, card in enumerate(row):
        with cols[idx]:
            create_card(card["icon"], card["title"], card["plot"], card["footer"], card["key"])

# Footer
st.markdown("""
<div class="footer">
    📊 Student Performance Dashboard &bull; Built with <span>Streamlit & Plotly</span> &bull; 9 Interactive Insights
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET (mirrors notebook)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=10000, seed=42):
    """Generate synthetic student performance dataset."""
    np.random.seed(seed)
    
    study_hours = np.clip(np.random.normal(5.0, 2.0, n), 0, 12)
    attendance_pct = np.clip(np.random.normal(75.0, 15.0, n), 20, 100)
    sleep_hours = np.clip(np.random.normal(7.0, 1.5, n), 3, 12)
    internet_usage = np.clip(np.random.normal(4.0, 2.0, n), 0, 12)
    assignments_comp = np.clip(np.random.normal(70.0, 20.0, n), 0, 100)
    previous_scores = np.clip(np.random.normal(65.0, 15.0, n), 20, 100)
    
    exam_score = (
        0.30 * study_hours * 5
        + 0.25 * attendance_pct * 0.5
        + 0.25 * previous_scores * 0.5
        + 0.10 * assignments_comp * 0.4
        - 0.05 * internet_usage * 2
        + np.random.normal(0, 5, n)
    )
    exam_score = np.clip(exam_score, 20, 100)
    
    placement_prob = 1 / (1 + np.exp(-(exam_score - 60) / 8))
    placement = np.where(np.random.rand(n) < placement_prob, 'Placed', 'Not Placed')
    
    df = pd.DataFrame({
        'Study Hours': study_hours,
        'Attendance Percentage': attendance_pct,
        'Sleep Hours': sleep_hours,
        'Internet Usage': internet_usage,
        'Assignments Completed': assignments_comp,
        'Previous Scores': previous_scores,
        'Exam Score': exam_score,
        'Placement Status': placement
    })
    return df

# ──────────────────────────────────────────────────────────────
# 2. HELPER FUNCTIONS FOR PLOTS
# ──────────────────────────────────────────────────────────────
def plot_histogram(df, col, title, color='#667eea', xlabel=''):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[col],
        marker_color=color,
        opacity=0.75,
        hovertemplate='%{x:.1f} → %{y} students<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, weight='bold')),
        xaxis_title=xlabel or col,
        yaxis_title='Count',
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI, sans-serif')
    )
    return fig

def plot_scatter(df, x_col, y_col, title, color='#667eea'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers',
        marker=dict(color=color, size=5, opacity=0.5),
        hovertemplate=f'{x_col}: %{{x:.1f}}<br>{y_col}: %{{y:.1f}}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, weight='bold')),
        xaxis_title=x_col,
        yaxis_title=y_col,
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI, sans-serif')
    )
    return fig

def plot_pie(df, col, title, colors=['#3BB273', '#E84855']):
    counts = df[col].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        marker=dict(colors=colors),
        textinfo='percent+label',
        textposition='inside',
        hole=0.4,
        hovertemplate='%{label}<br>%{value} students (%{percent})<extra></extra>'
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, weight='bold')),
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        font=dict(family='Segoe UI, sans-serif')
    )
    return fig

def plot_heatmap(df, cols, title):
    corr = df[cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=[
            [0, '#2E86AB'],
            [0.5, '#f7f7f7'],
            [1, '#E84855']
        ],
        zmin=-1,
        zmax=1,
        text=corr.round(2).values,
        texttemplate='%{text}',
        textfont=dict(size=9),
        hovertemplate='%{x} ↔ %{y}<br>r = %{z:.2f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, weight='bold')),
        margin=dict(l=80, r=20, t=40, b=60),
        xaxis=dict(tickangle=-30),
        yaxis=dict(autorange='reversed'),
        font=dict(family='Segoe UI, sans-serif', size=10)
    )
    return fig

def plot_bar_horizontal(df, col, title, color='#667eea', limit=10):
    counts = df[col].value_counts().head(limit)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts.values,
        y=counts.index,
        orientation='h',
        marker_color=color,
        opacity=0.8,
        hovertemplate='%{y}: %{x} students<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, weight='bold')),
        xaxis_title='Count',
        yaxis=dict(tickfont=dict(size=10), automargin=True),
        margin=dict(l=10, r=20, t=40, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI, sans-serif')
    )
    return fig

def plot_feature_importance(df):
    # Use absolute correlation with 'Exam Score'
    cols = ['Study Hours', 'Attendance Percentage', 'Sleep Hours', 'Internet Usage', 'Assignments Completed', 'Previous Scores']
    corr = df[cols + ['Exam Score']].corr()
    imp = corr['Exam Score'][cols].abs().sort_values(ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=imp.values,
        y=imp.index,
        orientation='h',
        marker_color='#F18F01',
        opacity=0.8,
        hovertemplate='%{y}: %{x:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text='Feature Importance (|Correlation with Exam Score|)', font=dict(size=14, weight='bold')),
        xaxis_title='Absolute Correlation',
        yaxis=dict(tickfont=dict(size=10), automargin=True),
        margin=dict(l=10, r=20, t=40, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI, sans-serif')
    )
    return fig, imp.index[0] if len(imp) > 0 else ''

# ──────────────────────────────────────────────────────────────
# 3. STREAMLIT APP
# ──────────────────────────────────────────────────────────────
# st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Custom CSS to mimic card style
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 18px;
        padding: 25px 30px;
        margin-bottom: 30px;
        box-shadow: 0 10px 35px rgba(102, 126, 234, 0.25);
        display: flex;
        align-items: center;
        gap: 20px;
        flex-wrap: wrap;
    }
    .main-header img {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #fff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3436;
        margin: 0;
    }
    .main-header h1 span {
        color: #667eea;
    }
    .main-header p {
        font-size: 1.05rem;
        color: #4a4a4a;
        margin: 5px 0 0 0;
    }
    .main-header .badge {
        display: inline-block;
        background: rgba(102, 126, 234, 0.25);
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-top: 6px;
    }
    .stat-card {
        background: #fff;
        border-radius: 14px;
        padding: 15px 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.04);
        transition: transform 0.2s;
    }
    .stat-card:hover {
        transform: translateY(-3px);
    }
    .stat-card .number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-card .label {
        font-size: 0.85rem;
        color: #7f8c8d;
        margin-top: 4px;
    }
    .card {
        background: #fff;
        border-radius: 14px;
        padding: 18px 18px 12px 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.04);
        transition: box-shadow 0.2s;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .card:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.10);
    }
    .card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }
    .card-header .icon {
        font-size: 1.6rem;
        flex-shrink: 0;
    }
    .card-header h3 {
        font-size: 0.95rem;
        font-weight: 600;
        color: #2d3436;
        margin: 0;
        line-height: 1.3;
    }
    .card-header h3 .highlight {
        color: #667eea;
    }
    .card-footer {
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid #f0f0f0;
        font-size: 0.82rem;
        color: #7f8c8d;
        line-height: 1.5;
    }
    .card-footer strong {
        color: #2d3436;
    }
    .footer {
        text-align: center;
        padding: 20px 0 10px 0;
        font-size: 0.9rem;
        color: #7f8c8d;
        border-top: 1px solid #e9ecef;
        margin-top: 20px;
    }
    .footer span {
        color: #667eea;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load data
df = generate_dataset()

# Header
col1, col2 = st.columns([1, 5])
with col1:
    # Try to load image
    if os.path.exists("student.png"):
        st.image("student.png", width=80)
    else:
        st.markdown("📸")
with col2:
    st.markdown("""
    <div>
        <h1>📊 <span>Student Performance</span> Intelligence Dashboard</h1>
        <p>Comprehensive analysis of student behavior, academic performance, and placement outcomes.</p>
        <span class="badge">🎯 9 Key Insights &bull; Interactive Visualizations</span>
    </div>
    """, unsafe_allow_html=True)

# Stats row
total = len(df)
placed = df[df['Placement Status'] == 'Placed'].shape[0]
avg_score = df['Exam Score'].mean()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="stat-card"><div class="number">{total:,}</div><div class="label">Total Records</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-card"><div class="number">7</div><div class="label">Features</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="stat-card"><div class="number">{placed}</div><div class="label">Placed</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="stat-card"><div class="number">{avg_score:.1f}</div><div class="label">Avg Exam Score</div></div>', unsafe_allow_html=True)

# Helper to create a card
def create_card(icon, title, plot_func, footer_html, key):
    with st.container():
        st.markdown(f"""
        <div class="card">
            <div class="card-header">
                <span class="icon">{icon}</span>
                <h3>{title}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Place plot
        fig = plot_func()
        st.plotly_chart(fig, use_container_width=True, key=key)
        # Footer
        st.markdown(f'<div class="card-footer">{footer_html}</div>', unsafe_allow_html=True)

# Compute stats for footers
def compute_stats():
    stats = {}
    # Exam score
    exam = df['Exam Score']
    stats['exam_mean'] = exam.mean()
    stats['exam_std'] = exam.std()
    # Placement counts
    stats['placed_count'] = placed
    stats['not_placed'] = total - placed
    # Correlations
    stats['corr_study_exam'] = df['Study Hours'].corr(df['Exam Score'])
    stats['corr_attend_exam'] = df['Attendance Percentage'].corr(df['Exam Score'])
    stats['corr_prev_exam'] = df['Previous Scores'].corr(df['Exam Score'])
    # Sleep
    stats['sleep_mean'] = df['Sleep Hours'].mean()
    stats['sleep_median'] = df['Sleep Hours'].median()
    # Internet
    stats['internet_mean'] = df['Internet Usage'].mean()
    stats['internet_median'] = df['Internet Usage'].median()
    # Top feature
    cols = ['Study Hours', 'Attendance Percentage', 'Sleep Hours', 'Internet Usage', 'Assignments Completed', 'Previous Scores']
    corr = df[cols + ['Exam Score']].corr()
    imp = corr['Exam Score'][cols].abs().sort_values(ascending=False)
    stats['top_feature'] = imp.index[0] if len(imp) > 0 else ''
    return stats

stats = compute_stats()

# Define card configurations
cards = [
    {
        "icon": "📈",
        "title": "Exam Score Distribution",
        "plot": lambda: plot_histogram(df, 'Exam Score', 'Exam Score Distribution', '#667eea', 'Exam Score'),
        "footer": f"📊 <strong>Mean:</strong> {stats['exam_mean']:.1f} &nbsp;|&nbsp; <strong>Std:</strong> {stats['exam_std']:.1f}",
        "key": "Exam_score"
    },
    {
        "icon": "🎯",
        "title": "Placement Status",
        "plot": lambda: plot_pie(df, 'Placement Status', 'Placement Status', ['#3BB273', '#E84855']),
        "footer": f"✅ <strong>Placed:</strong> {stats['placed_count']} &nbsp;|&nbsp; ❌ <strong>Not Placed:</strong> {stats['not_placed']}",
        "key": "placement_status"
    },
    {
        "icon": "📚",
        "title": "Study Hours vs Exam Score",
        "plot": lambda: plot_scatter(df, 'Study Hours', 'Exam Score', 'Study Hours → Exam Score', '#2E86AB'),
        "footer": f"⏱️ <strong>Correlation:</strong> {stats['corr_study_exam']:.3f}",
        "key": "study_hours"
    },
    {
        "icon": "🏫",
        "title": "Attendance vs Exam Score",
        "plot": lambda: plot_scatter(df, 'Attendance Percentage', 'Exam Score', 'Attendance → Exam Score', '#17A398'),
        "footer": f"📊 <strong>Correlation:</strong> {stats['corr_attend_exam']:.3f}",
        "key": "attendance"
    },
    {
        "icon": "📖",
        "title": "Previous Scores vs Exam Score",
        "plot": lambda: plot_scatter(df, 'Previous Scores', 'Exam Score', 'Previous Scores → Exam Score', '#7B2D8B'),
        "footer": f"📈 <strong>Correlation:</strong> {stats['corr_prev_exam']:.3f}",
        "key": "prev_scores"
    },
    {
        "icon": "🔥",
        "title": "Feature Correlation Heatmap",
        "plot": lambda: plot_heatmap(df, ['Study Hours', 'Attendance Percentage', 'Sleep Hours', 'Internet Usage', 'Assignments Completed', 'Previous Scores', 'Exam Score'], 'Correlation Heatmap'),
        "footer": "🔍 <strong>Key:</strong> Dark blue = negative, white = neutral, red = positive",
        "key": "scatter"
    },
    {
        "icon": "⚖️",
        "title": "Feature Importance (Exam Score)",
        "plot": lambda: plot_feature_importance(df)[0],
        "footer": f"🏆 <strong>Top predictor:</strong> {stats['top_feature']}",
        "key": "feature"
    },
    {
        "icon": "😴",
        "title": "Sleep Hours Distribution",
        "plot": lambda: plot_histogram(df, 'Sleep Hours', 'Sleep Hours Distribution', '#A29BFE', 'Sleep Hours'),
        "footer": f"💤 <strong>Mean:</strong> {stats['sleep_mean']:.1f} &nbsp;|&nbsp; <strong>Median:</strong> {stats['sleep_median']:.1f}",
        "key": "sleep_hours"
    },
    {
        "icon": "🌐",
        "title": "Internet Usage Distribution",
        "plot": lambda: plot_histogram(df, 'Internet Usage', 'Internet Usage Distribution', '#00B894', 'Internet Usage (hrs)'),
        "footer": f"🌍 <strong>Mean:</strong> {stats['internet_mean']:.1f} &nbsp;|&nbsp; <strong>Median:</strong> {stats['internet_median']:.1f}",
        "key": "internet_usage"
    }
]

# Arrange cards in a grid: 3 columns
rows = [cards[i:i+3] for i in range(0, len(cards), 3)]
for row in rows:
    cols = st.columns(3)
    for idx, card in enumerate(row):
        with cols[idx]:
            create_card(card["icon"], card["title"], card["plot"], card["footer"], card["key"])

# Footer
st.markdown("""
<div class="footer">
    📊 Student Performance Dashboard &bull; Built with <span>Streamlit & Plotly</span> &bull; 9 Interactive Insights
</div>
""", unsafe_allow_html=True)