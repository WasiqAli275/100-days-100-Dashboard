# 🤖 AI Impact on Jobs 2030 — Interactive Dashboard
 
<div align="center">
 

<!-- ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge) -->
 
**A professional, dark-themed, multi-tab Streamlit dashboard for exploring the AI Impact on Jobs 2030 dataset.**
 
</div>
 
---
 
## 📊 Why This Dataset is Exceptional for Dashboard-Style Visualization
 
The **AI Impact on Jobs 2030** dataset is one of the most naturally suited datasets for rich, interactive dashboard design. Here's a detailed breakdown of why:
 
---
 
### 🧩 1. Rich Multi-Dimensional Structure
 
The dataset spans **multiple analytical dimensions simultaneously** — jobs, industries, countries, salaries, and AI risk scores — meaning every chart tells a different part of the same story. This multi-axis nature makes it ideal for tabbed dashboards where each tab can explore a completely different slice without ever running out of meaningful content.
 
| Dimension | What It Enables |
|---|---|
| **Job Titles** | Rankings, salary comparisons, risk profiles |
| **Industries** | Sector-level aggregations, competitive benchmarking |
| **Countries** | Geographic distribution, regional salary gaps |
| **Salary (USD)** | Distribution charts, histograms, box plots |
| **AI Replacement Risk** | Risk scoring, scatter plots, heatmaps |
| **Future Demand Score** | Growth projections, forward-looking indicators |
| **Automation Level** | Categorical segmentation, pie/donut charts |
| **Skills** | Frequency parsing, skill-to-salary mapping |
 
> **Dashboard implication:** You never need to reuse the same chart type twice. Each dimension naturally calls for a different visual treatment.
 
---
 
### 📈 2. Continuous + Categorical Column Balance
 
The dataset contains a healthy mix of **quantitative** and **qualitative** columns — a critical property for dashboard design:
 
#### Continuous / Numeric Columns
- `average_salary_usd` → histograms, KDE, violin plots, scatter axes
- `ai_replacement_risk` → risk gauges, distributions, correlation axes
- `future_demand_score` → trend indicators, scatter overlays
- `job_growth_2030` → bar rankings, trend lines
- `years_experience` → scatter vs salary, bubble sizing
- `work_hours_per_week` → workload comparisons
- `performance_score` → performance benchmarks
- `job_satisfaction` → sentiment scoring visuals
 
#### Categorical Columns
- `industry`, `job_title`, `country` → group-by aggregations, ranked bars
- `automation_level` → pie/donut segmentation
- `education_level` → salary stratification
- `company_size` → size-based salary analysis
- `remote_work_possibility` → binary split comparisons
- `upskilling_needed` → yes/no donut charts
- `hiring_trend_2026` → forward-looking categorical trend bar
 
> **Dashboard implication:** The continuous columns power scatter plots, histograms, and correlation heatmaps. The categorical columns power ranked bar charts, pie/donut charts, and group-by aggregations — giving every chart module a natural data source.
 
---
 
### 🔗 3. Natural Cross-Dimensional Relationships
 
The dataset contains strong, naturally interpretable relationships between columns that make scatter plots and correlation heatmaps instantly meaningful:
 
```
AI Replacement Risk  ←→  Future Demand Score      (risk vs opportunity)
Years Experience     ←→  Average Salary USD        (career progression)
Automation Level     ←→  Job Satisfaction          (automation effect)
AI Replacement Risk  ←→  Average Salary USD        (premium for resilience)
Future Demand Score  ←→  Job Growth 2030           (demand alignment)
```
 
These relationships are not trivial — they tell a **coherent narrative** about the future of work, making every chart in the dashboard feel like part of a single analytical story rather than disconnected visuals.
 
> **Dashboard implication:** A single correlation heatmap across these columns becomes an executive insight on its own. Scatter plots between any two numeric columns reveal immediately actionable intelligence.
 
---
 
### 🏢 4. Hierarchical Grouping Structure
 
The dataset has a natural **three-level grouping hierarchy**:
 
```
Country
  └── Industry
        └── Job Title
              └── Individual Record
```
 
This hierarchy is perfect for:
- **Treemaps** — hierarchical area charts showing volume at each level
- **Sunburst charts** — radial hierarchy exploration
- **Drill-down filters** — sidebar country → industry → job title chaining
- **Aggregated bar charts** — average salary or AI risk at each level
 
> **Dashboard implication:** A single sidebar filter cascade (Country → Industry → Job Title) immediately transforms the entire dashboard, making it feel like an enterprise BI tool.
 
---
 
### 💰 5. Salary as a Universal Connector
 
`average_salary_usd` acts as a **universal linking variable** that connects every other column in the dataset. Almost any question can be answered through salary:
 
- *Which industries pay the most?* → Bar chart: `industry` × `avg salary`
- *Does AI risk affect salary?* → Scatter: `ai_replacement_risk` × `salary`
- *Does education level matter?* → Box plot: `education_level` × `salary`
- *Remote vs on-site salary gap?* → Bar: `remote_work_possibility` × `salary`
- *Which skills command premium pay?* → Bar: `skill` × `avg salary of jobs requiring it`
- *Does company size matter?* → Bar: `company_size` × `salary`
 
> **Dashboard implication:** A dedicated Salary tab with 6+ chart types all sourced from the same column creates one of the richest analytical sections possible from a single variable.
 
---
 
### 🧠 6. Forward-Looking Predictive Columns
 
Unlike most static datasets, this one contains **future-oriented columns** that make ML prediction and forecasting naturally compelling:
 
| Column | Dashboard Value |
|---|---|
| `job_growth_2030` | Acts as a built-in forecast target |
| `future_demand_score` | Creates a "bull/bear" job market signal |
| `ai_replacement_risk` | Becomes the central risk metric for the entire narrative |
| `hiring_trend_2026` | Short-term hiring signal — near-term forecast |
 
> **Dashboard implication:** A dedicated ML tab where users see a Gradient Boosting model predict salary in real-time — trained on AI risk, demand score, and experience — makes the dashboard feel analytically advanced without needing external data.
 
---
 
### 🛠️ 7. Skills Column — Unique Parsing Opportunity
 
The `required_skills` column stores comma-separated skill tags per job record. This is one of the most visually interesting columns in the dataset because it requires **text parsing before visualization**, creating a uniquely rich analytics opportunity:
 
```python
# Example: "Python, SQL, Machine Learning, TensorFlow"
# Parsed into individual skill frequency counts
# → Bar chart of top 20 skills
# → Skill-to-salary mapping (avg salary of jobs requiring each skill)
# → Skill co-occurrence matrix (which skills appear together)
```
 
This unlocks chart types unavailable from simple numeric columns:
- **Ranked skill frequency bars** — most in-demand skills
- **Skill-salary bars** — which skills command a pay premium
- **Skills per industry** — sector-specific skill landscape
 
> **Dashboard implication:** A dedicated Skills tab backed by runtime NLP parsing creates a layer of analytical depth that feels sophisticated and genuinely useful to job seekers, recruiters, and workforce planners alike.
 
---
 
### 🌍 8. Geographic Diversity
 
With records spanning multiple countries, the dataset enables:
 
- **Country-level salary comparison** — horizontal bar chart ranked by avg salary
- **Remote work by country** — stacked/grouped bar showing adoption rates
- **AI risk by geography** — which countries have higher automation exposure
- **Job count by country** — market size indicators
 
> **Dashboard implication:** A Country tab that immediately answers "where in the world should I look for jobs?" is one of the most compelling panels a workforce dashboard can offer.
 
---
 
### 🎯 9. Ideal KPI Cardinality for Executive Summary Cards
 
The dataset naturally produces **8–12 high-value KPI numbers** that work perfectly as executive summary metric cards at the top of any dashboard:
 
```
Total Records          Unique Job Titles        Industries Covered
Countries Represented  Average Salary (USD)     Median Salary (USD)
Avg AI Replacement Risk  Avg Future Demand Score
```
 
These are not arbitrary — each KPI tells an immediately interpretable story and sets the context for the deeper tabs below.
 
> **Dashboard implication:** A 6-column KPI row at the top of the dashboard gives any stakeholder the full picture in under 5 seconds before they explore any chart.
 
---
 
### ⚡ 10. Performance Characteristics
 
| Property | Value | Dashboard Benefit |
|---|---|---|
| Row count | Manageable (thousands range) | Fast Streamlit re-renders, no pagination needed |
| Column count | 15–25 | Wide enough for multi-tab depth, not so wide as to be overwhelming |
| Missing data | Minimal | Reliable aggregations without imputation artifacts |
| Data types | Mixed numeric + categorical | Full chart type coverage |
| String columns | Parseable, consistent | Group-by operations produce clean outputs |
 
> **Dashboard implication:** The dataset is large enough to produce statistically meaningful aggregations but small enough that `@st.cache_data` keeps the app snappy with no perceptible lag.
 
---
 
## 🏗️ Dashboard Architecture
 
```
app.py
│
├── 📊 Sidebar
│   ├── Dataset info (records, features, job titles, industries, countries)
│   ├── Tab navigation radio (8 tabs)
│   └── Optional AI banner image (Ai.png)
│
├── 🔢 KPI Row (6 metric cards — always visible)
│
└── 📑 Tab Panels
    ├── 🏠 Overview         → Industry bars, salary histogram, AI risk scatter, remote work pie
    ├── 🏢 Industry         → Job count, avg salary, AI risk, future demand — all by industry
    ├── 💼 Job Title        → Frequency, salary rankings, AI risk vs demand scatter
    ├── 🌍 Country          → Distribution, salary comparison, remote work by country
    ├── 💰 Salary           → Distribution, education level, experience scatter, company size
    ├── 🧠 AI Impact        → Risk distribution, automation level, correlation heatmap
    ├── 🛠️ Skills           → Top skills bar, skill-salary mapping (runtime parsed)
    └── 📈 ML Prediction    → GBM salary predictor, feature importance, actual vs predicted
```
 
---
 
## 🚀 Quick Start
 
### 1. Install dependencies
 
```bash
pip install streamlit pandas numpy plotly scikit-learn pillow
```
 
### 2. Prepare your files
 
```
project/
├── app.py                        ← dashboard code
├── AI_Impact_on_Jobs_2030.csv    ← dataset
└── Ai.png                        ← optional sidebar image
```
 
### 3. Launch the dashboard
 
```bash
streamlit run app.py
```
 
The app opens at `http://localhost:8501` in your browser.
 
---
 
## 📦 Dependencies
 
| Library | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.28 | Dashboard framework |
| `pandas` | ≥ 2.0 | Data loading and transformation |
| `numpy` | ≥ 1.24 | Numeric operations |
| `plotly` | ≥ 5.18 | Interactive charts (express + graph_objects) |
| `scikit-learn` | ≥ 1.3 | ML pipeline, GBM, metrics |
| `pillow` | ≥ 10.0 | Sidebar image loading |
 
---
 
## 🎨 Design System
 
The dashboard uses a consistent **dark automotive/tech theme**:
 
| Token | Hex | Usage |
|---|---|---|
| Background | `#0D1117` | Page background |
| Panel | `#161B22` | Cards, chart containers |
| Border | `#30363D` | Dividers, card edges |
| Text Primary | `#E6EDF3` | Main text |
| Text Muted | `#8B949E` | Labels, subtitles |
| Accent Gold | `#FFB703` | Headers, highlights, active tabs |
| Accent Red | `#E63946` | Risk indicators, alerts |
| Accent Blue | `#023E8A` | Salary, positive metrics |
| Accent Teal | `#06D6A0` | Demand scores, growth |
 
---
 
## 📁 Dataset Column Reference
 
| Column | Type | Description |
|---|---|---|
| `job_title` | categorical | Job role name |
| `industry` | categorical | Sector classification |
| `country` | categorical | Geographic location |
| `years_experience` | numeric | Years in the field |
| `education_level` | categorical | Highest qualification |
| `company_size` | categorical | Small / Medium / Large / Enterprise |
| `average_salary_usd` | numeric | Annual compensation in USD |
| `ai_replacement_risk` | numeric (0–1) | Probability of AI automation |
| `future_demand_score` | numeric | Projected job demand index |
| `job_growth_2030` | numeric | Projected growth % by 2030 |
| `automation_level` | categorical | Low / Medium / High |
| `required_skills` | text (CSV) | Comma-separated skill tags |
| `remote_work_possibility` | categorical | Yes / No / Hybrid |
| `work_hours_per_week` | numeric | Standard weekly hours |
| `upskilling_needed` | categorical | Yes / No |
| `hiring_trend_2026` | categorical | Growing / Stable / Declining |
| `performance_score` | numeric | Role performance benchmark |
| `job_satisfaction` | numeric | Satisfaction index |
 
---
 
## 🔮 Potential Extensions
 
- **Live salary range filter** — slider in sidebar to dynamically filter all charts
- **Job recommender** — input your skills, get top-matched job titles
- **Country choropleth map** — geographic heatmap of AI risk or salary
- **Time-series projection** — synthetic growth curves from `job_growth_2030`
- **Skill gap analyser** — compare your skills vs top-demand skills
- **PDF export** — one-click executive report generation
 
---
 
## 📄 License
 
This project is released under the **MIT License**. Dataset usage is subject to the original source's terms.
 
---
 
<div align="center">
 
Built with ❤️ using **Streamlit** · **Plotly** · **Pandas** · **scikit-learn**
 
*Data insights for the future of work — 2030 and beyond*
 
</div>
 