# 📊 Student Performance Intelligence Dashboard

A fully interactive data dashboard built with **Streamlit** and **Plotly** that visualizes key insights from student performance data. This project is a direct implementation of the exploratory analysis and visualizations from the original Jupyter Notebook (`deep-predictive-analysis-of-student-performance.ipynb`), transformed into a modern, responsive web application.

---

## 📌 Overview

The dashboard provides a comprehensive view of student behavior, academic performance, and placement outcomes. It presents **9 key insights** through interactive charts, including histograms, scatter plots, a pie chart, a heatmap, and feature importance analysis. All visualizations are built with Plotly, allowing users to hover, zoom, and explore the data dynamically.

This project evolved through three stages:
1. **Original Jupyter Notebook** – deep predictive analysis with static Matplotlib/Seaborn plots.
2. **HTML Dashboard** – a self‑contained HTML file using Plotly.js for interactivity.
3. **Streamlit App** – a Python‑based web application that offers the same functionality with easier deployment and customization.

The final product is the **Streamlit app** (`app.py`), which generates a synthetic dataset (mimicking the original data) and renders all visualizations in a clean, card‑based layout.

---

## ✨ Features

- **9 Interactive Visualizations** covering:
  - Exam score distribution
  - Placement status (pie chart)
  - Study hours vs exam score
  - Attendance vs exam score
  - Previous scores vs exam score
  - Correlation heatmap of all numeric features
  - Feature importance (absolute correlation with exam score)
  - Sleep hours distribution
  - Internet usage distribution

- **Dynamic Statistics** – each chart card includes a footer with relevant statistics (mean, standard deviation, correlations, counts, etc.) updated from the dataset.

- **Responsive Grid Layout** – adapts to any screen size, stacking cards on smaller devices.

- **Synthetic Data Generation** – if no dataset is provided, the app generates a realistic 10,000‑record dataset mirroring the original notebook’s schema.

- **Custom CSS Styling** – a pastel gradient header, card shadows, and hover effects give the dashboard a polished, professional look.

---

## 🧰 Tech Stack

| Tool/Library | Purpose |
|--------------|---------|
| **Python 3.8+** | Core language |
| **Streamlit** | Web app framework |
| **Plotly** | Interactive charts |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical operations |
| **Pillow (optional)** | Image handling (for `student.png`) |

---

## 📂 Dataset

The dashboard uses a **synthetic dataset** generated programmatically. It contains 8 columns:

| Column Name | Description |
|-------------|-------------|
| `Study Hours` | Hours studied per week (0–12) |
| `Attendance Percentage` | Class attendance (%) |
| `Sleep Hours` | Average sleep per night (3–12) |
| `Internet Usage` | Daily internet usage (hours) |
| `Assignments Completed` | Percentage of assignments submitted |
| `Previous Scores` | Prior academic performance |
| `Exam Score` | Final exam score (target variable) |
| `Placement Status` | `'Placed'` or `'Not Placed'` (based on exam score) |

The dataset is generated with a fixed random seed for reproducibility.

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone or download** the repository.
2. **Install dependencies**:
   ```bash
   pip install streamlit plotly pandas numpy