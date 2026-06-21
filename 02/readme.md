# 🚗 Car Market Intelligence Dashboard
 
An interactive Streamlit web application for exploring, visualizing, and modeling a comprehensive car market dataset. It covers manufacturer/model/trim analysis, pricing intelligence, market concentration, descriptive statistics, and a built-in machine learning module for MSRP (price) prediction.
 
---
 
## 📌 Overview
 
This app automatically loads any CSV dataset placed in its working directory, cleans and processes it, and presents the results through a dark-themed, multi-tab dashboard built with **Streamlit** and **Plotly**. It's designed to work with raw, messy automotive datasets — auto-detecting key columns (make, model, trim, MSRP, invoice, year) so no manual configuration is required.
 
---
 
## ✨ Features
 
### 🔄 Automatic Data Handling
- Scans the working directory (and subdirectories) for `.csv` files and loads the largest one found.
- Cleans column names into `snake_case`.
- Removes duplicate rows.
- Coerces price-like text columns (e.g. `"$45,000"`) into numeric values.
- Auto-detects key columns: **MSRP**, **Invoice**, **Make**, **Model**, **Trim**, **Year** — using flexible name-matching heuristics.
- Fills missing numeric values with the median and missing categorical values with the mode.
- Derives new fields:
  - `price_tier` — Budget / Mid-Range / Premium / Luxury / Ultra-Luxury segmentation
  - `price_gap` — MSRP minus Invoice
  - `margin_pct` — Dealer margin percentage
### 🖥️ Dashboard Tabs (sidebar navigation)
| Tab | Description |
|---|---|
| 🏠 **Overview** | Executive KPIs, price-tier distribution, MSRP histogram, top manufacturers, MSRP vs. Invoice scatter, year trend |
| 🏭 **Manufacturer** | Volume by manufacturer, average MSRP by manufacturer, brand segmentation (Luxury/Premium/Mid-Range/Budget), model diversification |
| 🚗 **Model** | Most popular models, competitive landscape bubble chart |
| 🔧 **Trim** | Most common trims and trim-level pricing breakdown |
| 💰 **Pricing** | MSRP distribution, price-gap analysis, dealer margin statistics, gap by manufacturer |
| 🌍 **Market** | Market share, market concentration metrics |
| 📈 **Statistics** | Full descriptive statistics table, correlation heatmap across numeric features |
| 🤖 **ML Prediction** | Gradient Boosting regression model to predict MSRP, with evaluation metrics and diagnostic plots |
 
### 🤖 Machine Learning — MSRP Prediction
- Algorithm: **GradientBoostingRegressor** (scikit-learn), wrapped in a pipeline with median imputation and feature standardization.
- Train/test split (80/20) plus 5-fold cross-validation.
- Metrics reported: **R²**, **MAE**, **RMSE**, **MAPE**, with CV mean ± std.
- Diagnostics:
  - Actual vs. Predicted scatter plot
  - Residual plot
  - Feature importance ranking (top 12)
  - Residual distribution histogram
  - Q-Q plot for residual normality
- Requires a minimum of 20 valid samples to train.
### 🎨 UI / Styling
- Custom dark GitHub-style theme (`#0D1117` background, amber `#FFB703` accent).
- Card-based KPI metrics, styled tabs, styled dataframes, and Plotly charts themed for dark mode.
- Optional sidebar car image (`car.png` / `car.jpg` / `car.jpeg`) if present in the project directory.
---
 
## 🛠️ Tech Stack
 
| Component | Library |
|---|---|
| Web framework | [Streamlit](https://streamlit.io/) |
| Data processing | pandas, numpy |
| Visualization | Plotly Express, Plotly Graph Objects |
| Machine learning | scikit-learn (GradientBoostingRegressor, pipelines, metrics) |
| Statistics | scipy.stats (Q-Q plot / normality check) |
| Image handling | Pillow (PIL) |
 
---
 
## 📂 Project Structure
 
```
.
├── app.py              # Main Streamlit application
├── *.csv                # Car market dataset (auto-detected — largest CSV in directory)
├── car.png / car.jpg    # Optional sidebar image
└── README.md
```
 
> The app does **not** require a specific dataset filename. It automatically finds and loads the largest `.csv` file in the project directory (or subdirectories), so multiple datasets can be swapped in without code changes.
 
---
 
## 📊 Dataset
 
This project is built around a **comprehensive car market analysis dataset**, typically including fields such as:
- Manufacturer / Make
- Model and Trim names
- Model Year
- MSRP (manufacturer's suggested retail price)
- Invoice / dealer price
- Additional vehicle specs (numeric features used for correlation analysis and ML)
The app is schema-flexible — column names don't need to match exactly; it searches for common naming patterns (e.g. `msrp`, `price`, `retail_price`, `make`, `manufacturer`, `model_year`, etc.) to identify the relevant fields automatically.
 
---
 
## 🚀 Getting Started
 
### Prerequisites
- Python 3.9+
### Installation
```bash
pip install streamlit pandas numpy plotly pillow scikit-learn scipy
```
 
### Run the app
```bash
streamlit run app.py
```
 
Place your dataset CSV (and optionally `car.png`) in the same directory as `app.py` before launching. The app will pick it up automatically on load.
 
---
 
## 📈 Roadmap / Ideas for Extension
- Add interactive filters (year range, manufacturer multiselect, price range slider) to narrow the dataset across all tabs.
- Add model comparison (e.g. Random Forest, XGBoost, Linear Regression) alongside Gradient Boosting.
- Export filtered data / charts as downloadable reports.
- Add geographic / regional market analysis if location data is available.
- Cache trained ML model to avoid retraining on every interaction.
---
 
## 📄 License
Specify your project license here (e.g. MIT).
 
---
 
*Built with Streamlit & Plotly · Comprehensive Car Market Analysis*
 