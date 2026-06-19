import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================
st.set_page_config(page_title="FIFA World Cup 2026 Analytics", layout="wide")

# =============================================================================
# 2. DATA LOADING WITH CACHING
# =============================================================================
@st.cache_data
def load_matches():
    try:
        df = pd.read_csv("matches.csv")
        return df
    except FileNotFoundError:
        st.error("matches.csv not found. Please ensure the file is in the same directory.")
        return pd.DataFrame()

@st.cache_data
def load_players():
    try:
        df = pd.read_csv("players.csv")
        # Fix age column: extract numeric age from strings like '27-003' (age-days)
        if 'age' in df.columns:
            df['age_num'] = df['age'].astype(str).str.split('-').str[0]
            df['age_num'] = pd.to_numeric(df['age_num'], errors='coerce')
        else:
            df['age_num'] = np.nan
        return df
    except FileNotFoundError:
        st.error("players.csv not found. Please ensure the file is in the same directory.")
        return pd.DataFrame()

@st.cache_data
def load_teams():
    try:
        df = pd.read_csv("teams.csv")
        return df
    except FileNotFoundError:
        st.error("teams.csv not found. Please ensure the file is in the same directory.")
        return pd.DataFrame()

# =============================================================================
# 3. LOAD DATA
# =============================================================================
matches_df = load_matches()
players_df = load_players()
teams_df = load_teams()

# Check if data loaded
if matches_df.empty or players_df.empty or teams_df.empty:
    st.error("Some datasets are missing. Please check file paths.")
    st.stop()

# =============================================================================
# 4. SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.title("⚽ FIFA World Cup 2026")
st.sidebar.markdown("---")

# Try to load local image for sidebar
try:
    st.sidebar.image("example.jpg", use_container_width=True)
except:
    st.sidebar.info("⚽ Image not found (example.jpg)")

analysis_choice = st.sidebar.radio(
    "Choose Analysis",
    [
        "Executive Dashboard",
        "Player Analytics",
        "Team Analytics",
        "Match Analytics",
        "Tournament Intelligence",
        "Statistical Analysis",
        "Machine Learning Prediction",
        "Insights & Recommendations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("FIFA World Cup 2026 Analytics Dashboard")

# =============================================================================
# 5. GLOBAL BANNER (shows on all pages)
# =============================================================================
st.markdown("""
<div style="background-color: #e3f2fd; padding: 25px; border-radius: 10px; border-left: 6px solid #1976d2; margin: 20px 0;">
<p style="font-size: 16px; margin: 0; color: #1565c0;">
<strong>🌍 The Biggest Football Tournament!</strong><br>
FIFA World Cup 2026 is the <strong>first-ever 48-team World Cup</strong>, co-hosted by:<br>
🇨🇦 <strong>Canada</strong> | 🇲🇽 <strong>Mexico</strong> | 🇺🇸 <strong>United States</strong>
</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 6. PAGE FUNCTIONS (Each returns multiple plots and insights)
# =============================================================================

def executive_dashboard():
    st.header("📊 Executive Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", len(matches_df))
    with col2:
        st.metric("Total Players", len(players_df))
    with col3:
        st.metric("Total Teams", len(teams_df['team'].unique()))
    with col4:
        total_goals = matches_df['home_score'].sum() + matches_df['away_score'].sum()
        st.metric("Total Goals", total_goals)
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        avg_age = players_df['age_num'].mean() if 'age_num' in players_df else None
        st.metric("Average Age", f"{avg_age:.1f}" if avg_age else "N/A")
    with col6:
        avg_possession = None
        if 'home_possession' in matches_df and 'away_possession' in matches_df:
            avg_possession = (matches_df['home_possession'].mean() + matches_df['away_possession'].mean()) / 2
        st.metric("Avg Possession", f"{avg_possession:.1f}%" if avg_possession else "N/A")
    with col7:
        total_assists = players_df['assists'].sum() if 'assists' in players_df else None
        st.metric("Total Assists", total_assists if total_assists else "N/A")
    with col8:
        top_scorer = None
        if 'goals' in players_df and not players_df['goals'].isna().all():
            top_scorer = players_df.loc[players_df['goals'].idxmax(), 'player'] if not players_df['goals'].empty else None
        st.metric("Top Scorer", top_scorer if top_scorer else "N/A")
    
    st.markdown("---")
    
    # 9 plots
    # 1. Goals per match distribution
    if 'home_score' in matches_df and 'away_score' in matches_df:
        matches_df['total_goals'] = matches_df['home_score'] + matches_df['away_score']
        fig1 = px.histogram(matches_df, x='total_goals', nbins=20, title='Goals per Match Distribution')
        st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Possession distribution home vs away
    if 'home_possession' in matches_df and 'away_possession' in matches_df:
        fig2 = px.box(matches_df, y=['home_possession', 'away_possession'], title='Possession Distribution (Home vs Away)')
        st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Home wins vs away wins vs draws
    home_wins = (matches_df['home_score'] > matches_df['away_score']).sum() if 'home_score' in matches_df and 'away_score' in matches_df else 0
    away_wins = (matches_df['away_score'] > matches_df['home_score']).sum() if 'home_score' in matches_df and 'away_score' in matches_df else 0
    draws = len(matches_df) - home_wins - away_wins if not matches_df.empty else 0
    fig3 = px.pie(values=[home_wins, away_wins, draws], names=['Home Wins', 'Away Wins', 'Draws'], title='Match Outcomes')
    st.plotly_chart(fig3, use_container_width=True)
    
    # 4. Goals by round
    if 'round' in matches_df and 'home_score' in matches_df and 'away_score' in matches_df:
        round_goals = matches_df.groupby('round')[['home_score', 'away_score']].sum().reset_index()
        round_goals['total'] = round_goals['home_score'] + round_goals['away_score']
        fig4 = px.bar(round_goals, x='round', y='total', title='Goals per Round')
        st.plotly_chart(fig4, use_container_width=True)
    
    # 5. Average goals per gameweek
    if 'gameweek' in matches_df and 'home_score' in matches_df and 'away_score' in matches_df:
        matches_df['total_goals'] = matches_df['home_score'] + matches_df['away_score']
        gw_goals = matches_df.groupby('gameweek')['total_goals'].mean().reset_index()
        fig5 = px.line(gw_goals, x='gameweek', y='total_goals', title='Average Goals per Gameweek')
        st.plotly_chart(fig5, use_container_width=True)
    
    # 6. Top scoring teams (from matches)
    if 'home_score' in matches_df and 'away_score' in matches_df and 'home_team' in matches_df and 'away_team' in matches_df:
        home_goals = matches_df.groupby('home_team')['home_score'].sum().reset_index().rename(columns={'home_team':'team', 'home_score':'goals'})
        away_goals = matches_df.groupby('away_team')['away_score'].sum().reset_index().rename(columns={'away_team':'team', 'away_score':'goals'})
        all_goals = home_goals.merge(away_goals, on='team', suffixes=('_home', '_away')).fillna(0)
        all_goals['total'] = all_goals['goals_home'] + all_goals['goals_away']
        top_teams = all_goals.nlargest(10, 'total')
        fig6 = px.bar(top_teams, x='total', y='team', orientation='h', title='Top 10 Teams by Goals Scored')
        st.plotly_chart(fig6, use_container_width=True)
    
    # 7. Referee analysis (most matches)
    if 'referee' in matches_df:
        ref_counts = matches_df['referee'].value_counts().reset_index()
        ref_counts.columns = ['Referee', 'Matches']
        fig7 = px.bar(ref_counts.head(10), x='Matches', y='Referee', orientation='h', title='Top 10 Referees by Matches Officiated')
        st.plotly_chart(fig7, use_container_width=True)
    
    # 8. Attendance distribution
    if 'attendance' in matches_df:
        fig8 = px.histogram(matches_df, x='attendance', nbins=20, title='Attendance Distribution')
        st.plotly_chart(fig8, use_container_width=True)
    
    # 9. Formation usage (from home_formation and away_formation)
    if 'home_formation' in matches_df and 'away_formation' in matches_df:
        formations = pd.concat([matches_df['home_formation'], matches_df['away_formation']]).value_counts().reset_index()
        formations.columns = ['Formation', 'Count']
        fig9 = px.bar(formations.head(10), x='Count', y='Formation', orientation='h', title='Top 10 Formations Used')
        st.plotly_chart(fig9, use_container_width=True)

def player_analytics():
    st.header("👤 Player Analytics")
    df = players_df.copy()
    
    # 1. Position distribution
    if 'position' in df:
        pos_counts = df['position'].value_counts().reset_index()
        pos_counts.columns = ['Position', 'Count']
        fig1 = px.bar(pos_counts, x='Position', y='Count', title='Player Distribution by Position')
        st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Age distribution (numeric)
    if 'age_num' in df:
        fig2 = px.histogram(df, x='age_num', nbins=20, title='Age Distribution', labels={'age_num':'Age'})
        st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Top 10 goalscorers
    if 'goals' in df and not df['goals'].isna().all():
        top_goals = df.nlargest(10, 'goals')[['player', 'team', 'goals']]
        fig3 = px.bar(top_goals, x='goals', y='player', orientation='h', title='Top 10 Goalscorers', color='team')
        st.plotly_chart(fig3, use_container_width=True)
    
    # 4. Top 10 assist providers
    if 'assists' in df and not df['assists'].isna().all():
        top_assists = df.nlargest(10, 'assists')[['player', 'team', 'assists']]
        fig4 = px.bar(top_assists, x='assists', y='player', orientation='h', title='Top 10 Assist Providers', color='team')
        st.plotly_chart(fig4, use_container_width=True)
    
    # 5. Top 10 minutes played
    if 'minutes' in df and not df['minutes'].isna().all():
        top_minutes = df.nlargest(10, 'minutes')[['player', 'team', 'minutes']]
        fig5 = px.bar(top_minutes, x='minutes', y='player', orientation='h', title='Top 10 Minutes Played', color='team')
        st.plotly_chart(fig5, use_container_width=True)
    
    # 6. Goals vs shots scatter
    if 'goals' in df and 'shots' in df:
        fig6 = px.scatter(df, x='shots', y='goals', hover_data=['player', 'team'], title='Goals vs Shots')
        st.plotly_chart(fig6, use_container_width=True)
    
    # 7. Goals per 90 distribution
    if 'goals_per90' in df:
        fig7 = px.histogram(df, x='goals_per90', nbins=20, title='Goals per 90 Distribution')
        st.plotly_chart(fig7, use_container_width=True)
    
    # 8. Club representation
    if 'club' in df:
        club_counts = df['club'].value_counts().reset_index()
        club_counts.columns = ['Club', 'Count']
        fig8 = px.bar(club_counts.head(10), x='Count', y='Club', orientation='h', title='Top 10 Clubs with Most Players')
        st.plotly_chart(fig8, use_container_width=True)
    
    # 9. Team representation (players per team)
    if 'team' in df:
        team_counts = df['team'].value_counts().reset_index()
        team_counts.columns = ['Team', 'Count']
        fig9 = px.bar(team_counts, x='Team', y='Count', title='Number of Players per Team')
        st.plotly_chart(fig9, use_container_width=True)

def team_analytics():
    st.header("🏆 Team Analytics")
    df = teams_df.copy()
    
    # 1. Goals per 90 ranking
    if 'goals_per90' in df:
        off_rank = df.nlargest(10, 'goals_per90')[['team', 'goals_per90']]
        fig1 = px.bar(off_rank, x='goals_per90', y='team', orientation='h', title='Top 10 Attacking Teams (Goals per 90)', color='team')
        st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Defensive ranking (goals against per 90)
    if 'goals_against_per90' in df:
        def_rank = df.nsmallest(10, 'goals_against_per90')[['team', 'goals_against_per90']]
        fig2 = px.bar(def_rank, x='goals_against_per90', y='team', orientation='h', title='Top 10 Defensive Teams (Goals Conceded per 90)', color='team')
        st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Possession leaders
    if 'possession' in df:
        poss_rank = df.nlargest(10, 'possession')[['team', 'possession']]
        fig3 = px.bar(poss_rank, x='possession', y='team', orientation='h', title='Top 10 Possession Leaders', color='team')
        st.plotly_chart(fig3, use_container_width=True)
    
    # 4. Squad utilization (players_used)
    if 'players_used' in df:
        util_rank = df.nlargest(10, 'players_used')[['team', 'players_used']]
        fig4 = px.bar(util_rank, x='players_used', y='team', orientation='h', title='Most Players Used', color='team')
        st.plotly_chart(fig4, use_container_width=True)
    
    # 5. Average age vs goals scatter
    if 'avg_age' in df and 'goals' in df:
        fig5 = px.scatter(df, x='avg_age', y='goals', hover_data=['team'], title='Average Age vs Goals Scored')
        st.plotly_chart(fig5, use_container_width=True)
    
    # 6. Possession vs goals
    if 'possession' in df and 'goals' in df:
        fig6 = px.scatter(df, x='possession', y='goals', hover_data=['team'], title='Possession vs Goals Scored')
        st.plotly_chart(fig6, use_container_width=True)
    
    # 7. Shots vs goals
    if 'shots' in df and 'goals' in df:
        fig7 = px.scatter(df, x='shots', y='goals', hover_data=['team'], title='Shots vs Goals')
        st.plotly_chart(fig7, use_container_width=True)
    
    # 8. Goals conceded vs possession
    if 'possession' in df and 'goals_against' in df:
        fig8 = px.scatter(df, x='possession', y='goals_against', hover_data=['team'], title='Possession vs Goals Conceded')
        st.plotly_chart(fig8, use_container_width=True)
    
    # 9. Heatmap of key metrics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig9 = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix of Team Metrics")
        st.plotly_chart(fig9, use_container_width=True)

def match_analytics():
    st.header("⚽ Match Analytics")
    df = matches_df.copy()
    
    # 1. Match outcomes per round
    if 'round' in df and 'home_score' in df and 'away_score' in df:
        df['outcome'] = np.where(df['home_score'] > df['away_score'], 'Home Win',
                                 np.where(df['home_score'] < df['away_score'], 'Away Win', 'Draw'))
        outcome_round = df.groupby(['round', 'outcome']).size().reset_index(name='count')
        fig1 = px.bar(outcome_round, x='round', y='count', color='outcome', title='Match Outcomes per Round', barmode='group')
        st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Goals per match distribution
    if 'home_score' in df and 'away_score' in df:
        df['total_goals'] = df['home_score'] + df['away_score']
        fig2 = px.histogram(df, x='total_goals', nbins=20, title='Goals per Match Distribution')
        st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Home vs Away goals
    if 'home_score' in df and 'away_score' in df:
        home_goals = df['home_score'].sum()
        away_goals = df['away_score'].sum()
        fig3 = px.bar(x=['Home', 'Away'], y=[home_goals, away_goals], title='Total Goals: Home vs Away')
        st.plotly_chart(fig3, use_container_width=True)
    
    # 4. Possession difference vs goal difference
    if 'home_possession' in df and 'away_possession' in df and 'home_score' in df and 'away_score' in df:
        df['possession_diff'] = df['home_possession'] - df['away_possession']
        df['goal_diff'] = df['home_score'] - df['away_score']
        fig4 = px.scatter(df, x='possession_diff', y='goal_diff', title='Possession Difference vs Goal Difference',
                          hover_data=['home_team', 'away_team'])
        st.plotly_chart(fig4, use_container_width=True)
    
    # 5. Formation usage
    if 'home_formation' in df and 'away_formation' in df:
        formations = pd.concat([df['home_formation'], df['away_formation']]).value_counts().reset_index()
        formations.columns = ['Formation', 'Count']
        fig5 = px.bar(formations.head(10), x='Count', y='Formation', orientation='h', title='Top 10 Formations Used')
        st.plotly_chart(fig5, use_container_width=True)
    
    # 6. Goals per gameweek trend
    if 'gameweek' in df and 'home_score' in df and 'away_score' in df:
        df['total_goals'] = df['home_score'] + df['away_score']
        gw_goals = df.groupby('gameweek')['total_goals'].mean().reset_index()
        fig6 = px.line(gw_goals, x='gameweek', y='total_goals', title='Average Goals per Gameweek')
        st.plotly_chart(fig6, use_container_width=True)
    
    # 7. Referee cards (yellow + red) analysis
    if 'home_cards_yellow' in df and 'away_cards_yellow' in df:
        df['total_yellow'] = df['home_cards_yellow'] + df['away_cards_yellow']
        ref_yellow = df.groupby('referee')['total_yellow'].sum().reset_index().nlargest(10, 'total_yellow')
        fig7 = px.bar(ref_yellow, x='total_yellow', y='referee', orientation='h', title='Top 10 Referees by Yellow Cards')
        st.plotly_chart(fig7, use_container_width=True)
    
    # 8. Attendance by round
    if 'attendance' in df and 'round' in df:
        fig8 = px.box(df, x='round', y='attendance', title='Attendance Distribution by Round')
        st.plotly_chart(fig8, use_container_width=True)
    
    # 9. Match intensity (fouls) vs cards
    if 'home_fouls' in df and 'away_fouls' in df and 'home_cards_yellow' in df and 'away_cards_yellow' in df:
        df['total_fouls'] = df['home_fouls'] + df['away_fouls']
        df['total_yellow'] = df['home_cards_yellow'] + df['away_cards_yellow']
        fig9 = px.scatter(df, x='total_fouls', y='total_yellow', title='Fouls vs Yellow Cards', hover_data=['home_team', 'away_team'])
        st.plotly_chart(fig9, use_container_width=True)

def tournament_intelligence():
    st.header("🌍 Tournament Intelligence")
    df = teams_df.copy()
    
    # 1. Strength index (composite)
    if all(col in df.columns for col in ['goals', 'goals_against', 'possession']):
        df['goals_norm'] = (df['goals'] - df['goals'].min()) / (df['goals'].max() - df['goals'].min())
        df['defense_norm'] = 1 - (df['goals_against'] - df['goals_against'].min()) / (df['goals_against'].max() - df['goals_against'].min())
        df['poss_norm'] = (df['possession'] - df['possession'].min()) / (df['possession'].max() - df['possession'].min())
        df['strength'] = 0.4 * df['goals_norm'] + 0.3 * df['defense_norm'] + 0.3 * df['poss_norm']
        strength_rank = df.nlargest(10, 'strength')[['team', 'strength']]
        fig1 = px.bar(strength_rank, x='strength', y='team', orientation='h', title='Top 10 Strongest Nations (Composite Score)')
        st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Efficiency (goals per shot)
    if 'goals' in df and 'shots' in df:
        df['efficiency'] = df['goals'] / df['shots']
        eff_rank = df.nlargest(10, 'efficiency')[['team', 'efficiency']]
        fig2 = px.bar(eff_rank, x='efficiency', y='team', orientation='h', title='Top 10 Most Efficient Teams (Goals per Shot)')
        st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Youngest squads
    if 'avg_age' in df:
        young_rank = df.nsmallest(10, 'avg_age')[['team', 'avg_age']]
        fig3 = px.bar(young_rank, x='avg_age', y='team', orientation='h', title='Youngest Squads (Emerging Nations)')
        st.plotly_chart(fig3, use_container_width=True)
    
    # 4. Goals per 90 vs shots per 90
    if 'goals_per90' in df and 'shots_per90' in df:
        fig4 = px.scatter(df, x='shots_per90', y='goals_per90', hover_data=['team'], title='Goals per 90 vs Shots per 90')
        st.plotly_chart(fig4, use_container_width=True)
    
    # 5. Goals against per 90 vs possession
    if 'goals_against_per90' in df and 'possession' in df:
        fig5 = px.scatter(df, x='goals_against_per90', y='possession', hover_data=['team'], title='Goals Conceded per 90 vs Possession')
        st.plotly_chart(fig5, use_container_width=True)
    
    # 6. Squad depth (players used vs goals)
    if 'players_used' in df and 'goals' in df:
        fig6 = px.scatter(df, x='players_used', y='goals', hover_data=['team'], title='Players Used vs Goals')
        st.plotly_chart(fig6, use_container_width=True)
    
    # 7. Age vs efficiency
    if 'avg_age' in df and 'efficiency' in df:
        fig7 = px.scatter(df, x='avg_age', y='efficiency', hover_data=['team'], title='Average Age vs Efficiency')
        st.plotly_chart(fig7, use_container_width=True)
    
    # 8. Possession vs strength
    if 'possession' in df and 'strength' in df:
        fig8 = px.scatter(df, x='possession', y='strength', hover_data=['team'], title='Possession vs Strength Index')
        st.plotly_chart(fig8, use_container_width=True)
    
    # 9. Defensive vs offensive balance
    if 'goals_per90' in df and 'goals_against_per90' in df:
        fig9 = px.scatter(df, x='goals_per90', y='goals_against_per90', hover_data=['team'], title='Offensive (x) vs Defensive (y) Performance')
        st.plotly_chart(fig9, use_container_width=True)

def statistical_analysis():
    st.header("📈 Statistical Analysis")
    df = teams_df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        # 1. Pearson correlation heatmap
        corr_pearson = df[numeric_cols].corr()
        fig1 = px.imshow(corr_pearson, text_auto=True, aspect="auto", title="Pearson Correlation Matrix")
        st.plotly_chart(fig1, use_container_width=True)
        
        # 2. Spearman correlation heatmap
        corr_spearman = df[numeric_cols].corr(method='spearman')
        fig2 = px.imshow(corr_spearman, text_auto=True, aspect="auto", title="Spearman Correlation Matrix")
        st.plotly_chart(fig2, use_container_width=True)
        
        # 3. Distribution of selected metric
        selected_col = st.selectbox("Select a metric to view distribution", numeric_cols)
        if selected_col:
            fig3 = px.histogram(df, x=selected_col, nbins=20, title=f'Distribution of {selected_col}')
            st.plotly_chart(fig3, use_container_width=True)
        
        # 4. Scatter matrix (first 5 numeric columns)
        cols_to_plot = numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
        if len(cols_to_plot) >= 2:
            fig4 = px.scatter_matrix(df, dimensions=cols_to_plot, title="Scatter Matrix")
            st.plotly_chart(fig4, use_container_width=True)
        
        # 5. Boxplot for each numeric column (side by side)
        fig5 = px.box(df, y=numeric_cols, title="Boxplots of Numeric Metrics")
        st.plotly_chart(fig5, use_container_width=True)
        
        # 6. Pairplot with selected columns (already done as scatter matrix)
        
        # 7. Correlation with goals (if exists)
        if 'goals' in numeric_cols:
            corr_goals = df[numeric_cols].corr()['goals'].drop('goals').sort_values(ascending=False)
            corr_goals_df = corr_goals.reset_index()
            corr_goals_df.columns = ['Metric', 'Correlation']
            fig7 = px.bar(corr_goals_df, x='Correlation', y='Metric', orientation='h', title='Correlation with Goals')
            st.plotly_chart(fig7, use_container_width=True)
        
        # 8. Correlation with possession
        if 'possession' in numeric_cols:
            corr_poss = df[numeric_cols].corr()['possession'].drop('possession').sort_values(ascending=False)
            corr_poss_df = corr_poss.reset_index()
            corr_poss_df.columns = ['Metric', 'Correlation']
            fig8 = px.bar(corr_poss_df, x='Correlation', y='Metric', orientation='h', title='Correlation with Possession')
            st.plotly_chart(fig8, use_container_width=True)
        
        # 9. Heatmap of selected columns (sub-correlation)
        selected_corr_cols = st.multiselect("Select columns for custom correlation", numeric_cols, default=numeric_cols[:5] if len(numeric_cols)>=5 else numeric_cols)
        if len(selected_corr_cols) > 1:
            fig9 = px.imshow(df[selected_corr_cols].corr(), text_auto=True, aspect="auto", title="Custom Correlation Heatmap")
            st.plotly_chart(fig9, use_container_width=True)
    else:
        st.warning("Not enough numeric columns for statistical analysis.")

def ml_prediction():
    st.header("🤖 Machine Learning Prediction")
    st.write("This section uses a Random Forest model to predict match outcomes and then ranks teams by win probability to determine the champion.")

    if matches_df.empty or teams_df.empty:
        st.error("Insufficient data for ML prediction.")
        return

    # ── Required columns ────────────────────────────────────────────────────────
    required_team_cols = ['goals', 'goals_against', 'possession', 'shots', 'shots_on_target']
    required_match_cols = ['home_team', 'away_team', 'home_score', 'away_score']

    missing_team  = [c for c in required_team_cols  if c not in teams_df.columns]
    missing_match = [c for c in required_match_cols if c not in matches_df.columns]
    if missing_team:
        st.error(f"Missing team columns: {missing_team}")
        return
    if missing_match:
        st.error(f"Missing match columns: {missing_match}")
        return

    # ── Build ML training data ───────────────────────────────────────────────────
    team_stats = teams_df[['team'] + required_team_cols].copy()

    home_stats = team_stats.add_prefix('home_')
    home_stats.rename(columns={'home_team': 'team'}, inplace=True)

    away_stats = team_stats.add_prefix('away_')
    away_stats.rename(columns={'away_team': 'team'}, inplace=True)

    df_ml = matches_df.merge(home_stats, left_on='home_team', right_on='team', how='inner')
    df_ml = df_ml.merge(away_stats, left_on='away_team', right_on='team', how='inner')

    if df_ml.empty:
        st.error("No match data after merging. Check team name consistency between datasets.")
        return

    # ── Define feature columns ────────────────────────────────────────────────────
    # Build the candidate list from required_team_cols, then keep ONLY those that
    # actually landed in df_ml after both merges.  This single list drives BOTH
    # model training AND champion-prediction vectors, so their widths are always
    # identical — regardless of which columns exist in the user's CSV files.
    candidate_feature_cols = (
        [f'home_{c}' for c in required_team_cols] +
        [f'away_{c}' for c in required_team_cols]
    )
    feature_cols = [c for c in candidate_feature_cols if c in df_ml.columns]

    # Derive the stat names that survived (used later to build prediction vectors)
    # e.g. if 'shots_on_target' was missing, surviving_stat_cols won't include it
    surviving_stat_cols = [c for c in required_team_cols if f'home_{c}' in feature_cols]

    df_ml['target'] = (df_ml['home_score'] > df_ml['away_score']).astype(int)

    X = df_ml[feature_cols]   # width = len(surviving_stat_cols) * 2
    y = df_ml['target']

    # ── Impute / scale / train ────────────────────────────────────────────────────
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)

    # ── KPI row ────────────────────────────────────────────────────────────────────
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("🎯 Model Accuracy", f"{acc:.2%}")
    with k2:
        st.metric("📊 Training Samples", len(X_train))
    with k3:
        st.metric("🔬 Test Samples", len(X_test))
    with k4:
        st.metric("🌲 Features Used", len(feature_cols))

    st.markdown("---")

    # ── PLOT 1 – Feature Importance (horizontal, multi-colour) ────────────────────
    st.subheader("📌 Plot 1 – Top Feature Importances")
    importances = model.feature_importances_
    feat_imp_df = (
        pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
        .sort_values('Importance', ascending=False)
        .head(10)
    )
    fig1 = px.bar(
        feat_imp_df, x='Importance', y='Feature', orientation='h',
        title='Top 10 Feature Importances',
        color='Importance',
        color_continuous_scale='Viridis',
        text=feat_imp_df['Importance'].map(lambda v: f"{v:.3f}")
    )
    fig1.update_layout(coloraxis_showscale=False, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig1, use_container_width=True)

    # ── PLOT 2 – Confusion Matrix ─────────────────────────────────────────────────
    st.subheader("📌 Plot 2 – Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig2 = px.imshow(
        cm, text_auto=True, aspect='auto',
        title='Confusion Matrix (Actual vs Predicted)',
        labels=dict(x='Predicted', y='Actual'),
        x=['Away Win / Draw', 'Home Win'],
        y=['Away Win / Draw', 'Home Win'],
        color_continuous_scale='Blues'
    )
    fig2.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    # ── PLOT 3 – Predicted probability distribution ────────────────────────────────
    st.subheader("📌 Plot 3 – Predicted Probability Distribution")
    prob_df = pd.DataFrame({'Win Probability': y_prob, 'Actual': y_test.values})
    prob_df['Actual Label'] = prob_df['Actual'].map({1: 'Home Win', 0: 'Away Win / Draw'})
    fig3 = px.histogram(
        prob_df, x='Win Probability', color='Actual Label',
        nbins=30, barmode='overlay', opacity=0.7,
        title='Distribution of Predicted Win Probabilities',
        color_discrete_map={'Home Win': '#2ecc71', 'Away Win / Draw': '#e74c3c'}
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Champion Prediction ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏆 Champion Prediction via Round-Robin Simulation")

    # Build lookup using ONLY surviving_stat_cols — the same columns that went
    # into X / imputer / scaler / model.  Concatenating home + away gives exactly
    # len(feature_cols) values every time → zero shape-mismatch risk.
    team_stats_dict = {}
    for _, row in team_stats.iterrows():
        team_stats_dict[row['team']] = row[surviving_stat_cols].values  # len = k

    teams_list = [t for t in team_stats_dict
                  if t in set(df_ml['home_team']).union(df_ml['away_team'])]

    win_probs = []
    for team in teams_list:
        probs = []
        home_vec = team_stats_dict[team]                        # shape (k,)
        for opp in teams_list:
            if opp == team:
                continue
            away_vec = team_stats_dict[opp]                     # shape (k,)
            fv = np.concatenate([home_vec, away_vec]).reshape(1, -1)  # shape (1, 2k)
            # 2k == len(feature_cols) — guaranteed match with the fitted imputer
            fv_imputed = imputer.transform(fv)
            fv_scaled  = scaler.transform(fv_imputed)
            prob = model.predict_proba(fv_scaled)[0][1]
            probs.append(prob)
        win_probs.append({'Team': team, 'Avg Win Probability': np.mean(probs) if probs else 0.0})

    win_probs_df = (
        pd.DataFrame(win_probs)
        .sort_values('Avg Win Probability', ascending=False)
        .reset_index(drop=True)
    )
    win_probs_df['Rank'] = win_probs_df.index + 1

    # ── PLOT 4 – Top 10 Contenders bar chart ─────────────────────────────────────
    st.subheader("📌 Plot 4 – Top 10 Champion Contenders")
    top10 = win_probs_df.head(10).copy()
    top10['colour'] = top10['Avg Win Probability']
    fig4 = px.bar(
        top10, x='Avg Win Probability', y='Team', orientation='h',
        title='Top 10 Teams by Average Win Probability',
        color='Avg Win Probability',
        color_continuous_scale='RdYlGn',
        text=top10['Avg Win Probability'].map(lambda v: f"{v:.1%}")
    )
    fig4.update_traces(textposition='outside')
    fig4.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False,
        xaxis_tickformat='.0%'
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Champion banner
    champ = win_probs_df.iloc[0]
    st.success(f"🏆 Predicted Champion: **{champ['Team']}**  —  Avg Win Probability: **{champ['Avg Win Probability']:.2%}**")

    # ── PLOT 5 – Win-probability distribution (histogram) ─────────────────────────
    st.subheader("📌 Plot 5 – Win Probability Distribution Across All Teams")
    fig5 = px.histogram(
        win_probs_df, x='Avg Win Probability', nbins=20,
        title='Distribution of Average Win Probabilities',
        color_discrete_sequence=['#9b59b6'],
        marginal='rug'
    )
    fig5.update_layout(xaxis_tickformat='.0%')
    st.plotly_chart(fig5, use_container_width=True)

    # ── PLOT 6 – Violin plot of win probabilities by quartile tier ────────────────
    st.subheader("📌 Plot 6 – Violin Plot of Win Probabilities")
    win_probs_df['Tier'] = pd.qcut(
        win_probs_df['Avg Win Probability'], q=4,
        labels=['Bottom 25%', 'Lower-Mid', 'Upper-Mid', 'Top 25%']
    )
    fig6 = px.violin(
        win_probs_df, x='Tier', y='Avg Win Probability', box=True, points='all',
        title='Win Probability Violin Plot by Performance Tier',
        color='Tier',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig6.update_layout(yaxis_tickformat='.0%', showlegend=False)
    st.plotly_chart(fig6, use_container_width=True)

    # ── PLOT 7 – Scatter: home goals vs away goals coloured by win-prob ───────────
    st.subheader("📌 Plot 7 – Team Attack vs Defence coloured by Win Probability")
    scatter_df = team_stats.merge(
        win_probs_df[['Team', 'Avg Win Probability']], left_on='team', right_on='Team', how='inner'
    )
    if not scatter_df.empty:
        fig7 = px.scatter(
            scatter_df, x='goals', y='goals_against',
            color='Avg Win Probability',
            size='possession',
            hover_name='team',
            color_continuous_scale='RdYlGn_r',
            title='Goals Scored vs Goals Conceded (size = Possession, colour = Win Prob)',
            labels={'goals': 'Goals Scored', 'goals_against': 'Goals Conceded'}
        )
        fig7.update_layout(coloraxis_colorbar_tickformat='.0%')
        st.plotly_chart(fig7, use_container_width=True)

    # ── PLOT 8 – Line chart: cumulative ranking by win probability ─────────────────
    st.subheader("📌 Plot 8 – Cumulative Win Probability Across Rankings")
    win_probs_df_sorted = win_probs_df.sort_values('Avg Win Probability', ascending=False).reset_index(drop=True)
    win_probs_df_sorted['Cumulative Probability'] = win_probs_df_sorted['Avg Win Probability'].cumsum()
    fig8 = px.line(
        win_probs_df_sorted, x=win_probs_df_sorted.index + 1,
        y='Cumulative Probability',
        hover_name='Team',
        markers=True,
        title='Cumulative Win Probability by Team Rank',
        labels={'x': 'Team Rank', 'Cumulative Probability': 'Cumulative Win Prob'}
    )
    fig8.update_traces(line_color='#e67e22', marker=dict(color='#c0392b', size=6))
    st.plotly_chart(fig8, use_container_width=True)

    # ── PLOT 9 – Full ranking table with colour-coded bar ─────────────────────────
    st.subheader("📌 Plot 9 – Full Team Rankings")
    fig9 = px.bar(
        win_probs_df, x='Avg Win Probability', y='Team',
        orientation='h',
        title='All Teams Ranked by Average Win Probability',
        color='Avg Win Probability',
        color_continuous_scale='Turbo',
        text=win_probs_df['Avg Win Probability'].map(lambda v: f"{v:.1%}")
    )
    fig9.update_traces(textposition='outside')
    fig9.update_layout(
        height=max(500, len(win_probs_df) * 22),
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False,
        xaxis_tickformat='.0%'
    )
    st.plotly_chart(fig9, use_container_width=True)

    # ── Classification Report ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Model Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Blues'))

    # ── Full ranking dataframe ────────────────────────────────────────────────────
    st.subheader("📋 Full Team Rankings Table")
    st.dataframe(
        win_probs_df[['Rank', 'Team', 'Avg Win Probability']]
        .style.format({'Avg Win Probability': '{:.2%}'})
        .background_gradient(subset=['Avg Win Probability'], cmap='RdYlGn')
    )

def insights_recommendations():
    st.header("💡 Insights & Recommendations")
    st.markdown("---")
    
    insights = []
    
    # Top scorer team
    if 'goals' in teams_df and not teams_df['goals'].isna().all():
        top_scoring_team = teams_df.loc[teams_df['goals'].idxmax(), 'team']
        insights.append(f"**Top Scoring Team:** {top_scoring_team} with {teams_df['goals'].max()} goals.")
    
    # Best defense
    if 'goals_against' in teams_df and not teams_df['goals_against'].isna().all():
        best_defense = teams_df.loc[teams_df['goals_against'].idxmin(), 'team']
        insights.append(f"**Best Defense:** {best_defense} with only {teams_df['goals_against'].min()} goals conceded.")
    
    # Highest possession
    if 'possession' in teams_df and not teams_df['possession'].isna().all():
        poss_leader = teams_df.loc[teams_df['possession'].idxmax(), 'team']
        insights.append(f"**Highest Possession:** {poss_leader} with {teams_df['possession'].max()}% average possession.")
    
    # Youngest team
    if 'avg_age' in teams_df and not teams_df['avg_age'].isna().all():
        youngest = teams_df.loc[teams_df['avg_age'].idxmin(), 'team']
        insights.append(f"**Youngest Squad:** {youngest} with average age {teams_df['avg_age'].min():.1f} years.")
    
    # Most efficient (goals per shot)
    if 'goals' in teams_df and 'shots' in teams_df:
        teams_df['efficiency'] = teams_df['goals'] / teams_df['shots']
        most_efficient = teams_df.loc[teams_df['efficiency'].idxmax(), 'team']
        insights.append(f"**Most Efficient Attack:** {most_efficient} with {teams_df['efficiency'].max():.2f} goals per shot.")
    
    # Home advantage
    if 'home_score' in matches_df and 'away_score' in matches_df:
        home_wins = (matches_df['home_score'] > matches_df['away_score']).sum()
        total = len(matches_df)
        home_win_pct = home_wins / total if total > 0 else 0
        insights.append(f"**Home Advantage:** {home_win_pct:.1%} of matches won by home team.")
    
    for i, ins in enumerate(insights):
        st.write(f"- {ins}")
    
    st.subheader("Recommendations")
    recs = []
    if 'possession' in teams_df and 'goals' in teams_df:
        corr = teams_df[['possession', 'goals']].corr().iloc[0,1]
        if corr > 0.5:
            recs.append("Teams with high possession tend to score more goals. Focus on maintaining possession.")
        else:
            recs.append("Possession does not strongly correlate with goals. Consider counter-attacking strategies.")
    
    if 'avg_age' in teams_df and 'goals' in teams_df:
        corr_age_goals = teams_df[['avg_age', 'goals']].corr().iloc[0,1]
        if corr_age_goals < -0.3:
            recs.append("Younger teams tend to score more goals. Invest in youth development.")
        else:
            recs.append("Experience (age) may be beneficial for goal scoring. Balance youth and experience.")
    
    if not recs:
        recs.append("Analyze your team's strengths and weaknesses using the provided metrics.")
    
    for r in recs:
        st.write(f"- {r}")
    
    st.subheader("Final Thoughts")
    st.write("The FIFA World Cup 2026 is shaping up to be an exciting tournament with a mix of traditional powerhouses and emerging nations. Keep an eye on teams that combine strong defense with efficient attack.")

# =============================================================================
# 7. RENDER SELECTED ANALYSIS
# =============================================================================
if analysis_choice == "Executive Dashboard":
    executive_dashboard()
elif analysis_choice == "Player Analytics":
    player_analytics()
elif analysis_choice == "Team Analytics":
    team_analytics()
elif analysis_choice == "Match Analytics":
    match_analytics()
elif analysis_choice == "Tournament Intelligence":
    tournament_intelligence()
elif analysis_choice == "Statistical Analysis":
    statistical_analysis()
elif analysis_choice == "Machine Learning Prediction":
    ml_prediction()
elif analysis_choice == "Insights & Recommendations":
    insights_recommendations()

