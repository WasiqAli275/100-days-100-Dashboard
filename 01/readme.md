# ⚽ FIFA World Cup 2026 Analytics Dashboard

An interactive, production‑ready dashboard built with **Streamlit** and **Plotly** to explore, analyze, and predict outcomes of the FIFA World Cup 2026. This tool provides comprehensive insights into players, teams, matches, and tournament dynamics – all powered by real tournament data.

---

## 🚀 Features

- **Executive Dashboard** – Key performance indicators (KPIs), match outcome distributions, goal trends, attendance, referee analysis, and formation usage.
- **Player Analytics** – Position distribution, age profile, top goal scorers, assist leaders, minutes played, club representation, and goalkeeper stats.
- **Team Analytics** – Attacking/defensive rankings, possession leaders, squad depth, and correlations between key team metrics.
- **Match Analytics** – Round‑by‑round outcomes, home vs. away performance, possession vs. goal difference, formation popularity, and referee card tendencies.
- **Tournament Intelligence** – Composite strength index, efficiency (goals per shot), emerging nations (youngest squads), and tactical balance analysis.
- **Statistical Analysis** – Pearson & Spearman correlation heatmaps, distribution plots, scatter matrices, and custom correlation exploration.
- **Machine Learning Prediction** – Uses **Random Forest** to predict match outcomes and simulates a round‑robin to estimate each team's win probability, ultimately predicting the **World Cup Champion**.
- **Insights & Recommendations** – Summarizes key findings and provides actionable strategic advice based on data patterns.

---

## 📁 Datasets

The dashboard automatically loads three CSV files located in the same directory as `app.py`:

- `matches.csv` – Full match statistics (scores, formations, possession, shots, cards, etc.)
- `players.csv` – Detailed player information (demographics, performance metrics, per‑90 stats, goalkeeper specifics)
- `teams.csv` – Aggregated team statistics (offensive, defensive, possession, squad usage, etc.)

> **Note:** The data is already provided; no external download is required.

---

## 🖥️ Installation & Running

### 1. Clone or download the repository

Ensure `app.py`, the three CSV files, and `requirements.txt` are in the same folder.

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate