# 🎬 Netflix Catalog Analytics, Forecasting & Recommendation Engine

An end-to-end data science project analyzing **8,790 Netflix catalog records** to uncover content acquisition trends, forecast release volumes, classify media formats, and power a weighted content-based recommendation engine—deployed via an interactive **Streamlit** web application.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Tech Stack](#-tech-stack)
- [Project Architecture & Pipeline](#-project-architecture--pipeline)
- [Key Business Insights](#-key-business-insights)
- [Strategic Business Recommendations](#-strategic-business-recommendations)
- [Setup & Installation](#-setup--installation)

---

## 🎯 Project Overview
Streaming platforms face two competing pressures: **controlling content acquisition costs** while **maximizing subscriber retention through personalized discovery**. This project addresses both challenges across six structured tasks:
1. **Data Cleaning & Deduplication:** Identifying and removing duplicate catalog entries (ignoring unique `show_id` keys) and standardizing datetime/categorical schemas.
2. **Exploratory Data Analysis (EDA):** Auditing content format splits, maturity ratings, release surges, and geographic production dominance.
3. **Content-Based Recommendation System:** Engineering a **TF-IDF + Cosine Similarity** recommender with **2x feature weighting** on genres and directors, supporting dynamic content-type filtering (`Movie` vs. `TV Show`).
4. **Time-Series Forecasting:** Modeling annual release trajectories (`2021–2025`) and monthly platform additions using **Holt’s Damped Trend & Seasonal Exponential Smoothing**.
5. **Supervised Machine Learning Classification:** Building a **Scikit-Learn Pipeline** (`ColumnTransformer` + `RandomForestClassifier`) evaluated with 5-fold cross-validation and ROC-AUC diagnostics to predict content formats from metadata.
6. **Interactive Executive Dashboard:** Delivering all visual diagnostics, forecasts, and live ML inference through a **Streamlit** application.

---

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning & NLP:** Scikit-Learn (`TfidfVectorizer`, `OneHotEncoder`, `StandardScaler`, `ColumnTransformer`, `Pipeline`, `RandomForestClassifier`, `cosine_similarity`)
* **Time-Series Modeling:** Statsmodels (`ExponentialSmoothing`)
* **Visualization & UI:** Plotly Express, Plotly Graph Objects, Matplotlib, Seaborn, Streamlit

---

## 🏗️ Project Architecture & Pipeline

```text
├── Auspify Project.ipynb    # Full exploratory notebook (Tasks 1–6)
├── app.py                   # Interactive Streamlit web application
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation & business strategy
```

---

## 📊 Key Business Insights

1. **Catalog Asymmetry & Audience Targeting**
   * **Movies dominate the catalog at ~69.6%**, while **TV Shows account for ~30.4%**. Although movies provide rapid library breadth, serialized TV shows—a higher-retention format—are significantly under-indexed.
   * **Mature Content Focus:** **TV-MA** is the highest-volume maturity rating on the platform, demonstrating a core content strategy heavily anchored in adult-oriented programming over family or general-audience titles.

2. **Geographic Concentration & Supply Chain Risk**
   * The **United States** leads total content production by a wide margin, indicating heavy reliance on North American studios and licensing pipelines.
   * High genre representation in *International Movies* and *Dramas* points to strong global catalog demand that can be leveraged to diversify regional production investments.

3. **Content Freshness & Acquisition Lag Dynamics**
   * Measuring **Acquisition Lag** (`year_added - release_year`) reveals a stark operational difference between formats:
     * **TV Shows** exhibit a much lower median acquisition lag, reflecting Netflix's aggressive push into day-and-date Original series.
     * **Movies** show a wider, older lag distribution, confirming that the film library relies substantially on licensing third-party back-catalog titles.

4. **Growth Stabilization & Seasonal Addition Cycles**
   * Catalog production surged sharply leading up to **2020**. However, **Holt’s Damped Trend Exponential Smoothing** forecasts a plateau in annual release volume across the **2021–2025** horizon, signaling a strategic shift from *volume expansion* to *subscriber retention and capital efficiency*.
   * Monthly additions (`2016–2021`) follow a distinct **12-month seasonal cycle**, with recurring Q3/Q4 spikes timed to capture peak holiday viewership.

---

## 💡 Strategic Business Recommendations

### 1. Rebalance Content Spend Toward Serialized TV Shows
* **Action:** Shift incremental acquisition budgets away from aging licensed movies toward multi-season TV series and limited series.
* **Business Impact:** Closing the **69.6% vs. 30.4%** format gap increases recurring weekly engagement and reduces monthly subscriber churn.

### 2. Deploy Weighted Metadata Similarity in Discovery Carousels
* **Action:** Integrate the **TF-IDF + Cosine Similarity** recommender into production "More Like This" rails, preserving the **2x feature weighting on Genres (`listed_in`) and Directors**.
* **Business Impact:** Double-weighting creative talent and sub-genres surfaces stylistically cohesive recommendations across different release years and countries, increasing catalog depth consumption beyond trending top-10 titles.

### 3. Enforce Acquisition Lag Caps on Third-Party Film Licensing
* **Action:** Use historical median acquisition lag benchmarks to cap licensing fees for older back-catalog movies, reallocating capital toward **0-year lag Originals** and exclusive regional releases.
* **Business Impact:** Improves content ROI during a forecasted plateau in total catalog growth (`2021–2025`), ensuring spend prioritizes high-acquisition-value titles.

### 4. Synchronize Marketing & Infrastructure with Seasonal Peaks
* **Action:** Align major promotional campaigns, lifecycle push notifications, and CDN capacity scaling with the **12-month Holt-Winters seasonal addition peaks**.
* **Business Impact:** Maximizes customer acquisition efficiency during high-volume release months while preventing discovery fatigue during quieter release months.

### 5. Automate Partner Ingestion QA with the Classification Pipeline
* **Action:** Deploy the **Random Forest Pipeline** (`Title` + `Director` TF-IDF + `Country`/`Rating` encoding) as an automated validation layer during partner metadata ingestion.
* **Business Impact:** Automatically flags or imputes misclassified content types (`Movie` vs. `TV Show`) before titles go live, reducing manual catalog operations overhead and preventing broken user-facing search filters.

---

## 🚀 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd netflix-catalog-analytics
   ```

2. **Install dependencies:**
   ```bash
   pip install streamlit pandas numpy scikit-learn statsmodels plotly matplotlib seaborn
   ```

3. **Launch the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
