# 🔍 Crime Pattern Intelligence System

A machine learning system that predicts crime frequency across Vancouver neighbourhoods,
built to support data-driven resource allocation for law enforcement and urban planners.

---

## 📌 Overview

Given neighbourhood, crime type, season, and era features, this model predicts expected
crime volume using a regression pipeline trained on real Vancouver crime incident records.
Built with Python, scikit-learn, and XGBoost — with a Streamlit app powered by a
Groq model narrative insight layer.

---

## 📂 Dataset

- **Source:** Vancouver Open Data — Crime Incidents
- **~93,756 rows** of crime incident records
- **13 columns** covering neighbourhood, crime type, date/time, and location
- **Target variable:** `crime_count` — aggregated incident frequency per group

---

## ⚙️ Preprocessing & Feature Engineering

Raw incident records were aggregated and enriched before modelling.

**Encoding:**
- `NEIGHBOURHOOD_ENC` — Label Encoding applied to neighbourhood
- `TYPE_encoded` — Target Encoding applied to crime type

**Engineered Features:**
- `SEASON` — derived from month to capture seasonal crime patterns
- `CRIME_ERA` — derived from year to capture long-term trend shifts

**Target Transformation:**
- `log1p()` applied to `crime_count` due to high skewness (skew = 6.0)
- Predictions reverse-transformed using `np.expm1()` at inference time

---

## 🤖 Models

| Model | Role | Tuning |
|-------|------|--------|
| Linear Regression | Baseline | None |
| Random Forest | Intermediate | GridSearchCV |
| XGBoost | **Final model** | RandomizedSearchCV |

**Final model R²: ~0.68**
Model persisted with `joblib` for reuse in the Streamlit app.

---

## 🛠️ Tech Stack

- Python
- pandas, numpy, scikit-learn, XGBoost, joblib
- Streamlit (frontend app)
- Groq API (narrative insight generation)
- GitHub + Streamlit Cloud (deployment)

---

## 🚀 How to Run

1. Clone the repo
```bash
git clone https://github.com/<BuildWithFemi>/crime-pattern-intelligence.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Add your Azure OpenAI credentials to a `.env` file (local)
   or to `secrets.toml` under `.streamlit/` for Streamlit Cloud

4. Launch the app
```bash
streamlit run app.py
```

---

## 🌐 Live Demo

👉 [View on Streamlit Cloud](<https://fbi-crime-investigation-firstproject1.streamlit.app>) ← update with link

---

*Built by [Femi](https://github.com/<BuildWithFemi>) — Data Science Portfolio Project*
