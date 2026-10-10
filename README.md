# Urban Traffic Flow Optimization

A Python-based data science and traffic analysis system built to analyze historical traffic patterns, identify peak congestion periods, study weather and temporal impacts, engineer predictive features, and prepare for machine learning traffic-volume forecasting.

---

## 1. Project Objective

Urban roadway networks face dynamic congestion influenced by commuting schedules, holidays, and severe weather conditions. The objective of this project is to develop an end-to-end Python pipeline using the **Metro Interstate Traffic Volume** dataset to:
1. Ingest and validate historical hourly traffic records.
2. Characterize temporal traffic rhythms (rush hours, diurnal cycles, weekend vs. weekday variations).
3. Analyze environmental determinants (temperature, precipitation, cloud cover, and categorical weather states).
4. Perform feature engineering (cyclical time encodings, lag indicators, weather transformations).
5. Lay a reproducible foundation for regression-based traffic volume forecasting.

---

## 2. Dataset Source & Attribution

* **Source:** [UCI Machine Learning Repository - Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/urban+traffic)
* **DOI:** [10.24432/C5X60B](https://doi.org/10.24432/C5X60B)
* **Citation:** Hogue, J. (2019). *Metro Interstate Traffic Volume*. UCI Machine Learning Repository.
* **Geographic Coverage:** Hourly westbound traffic flow on **Interstate 94 (I-94)** for Minnesota Department of Transportation (MN DoT) ATR station 301, located roughly midway between **Minneapolis and St. Paul, Minnesota, USA**. *(Note: This dataset reflects Minneapolis–St. Paul traffic conditions and does not describe Pune traffic data).*
* **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)).

### Dataset Schema (9 Original Attributes)
| Column Name | Role | Data Type | Description |
|---|---|---|---|
| `holiday` | Feature | Categorical | US National holidays plus Minnesota State Fair (or `None`) |
| `temp` | Feature | Numeric (Continuous) | Average temperature in Kelvin |
| `rain_1h` | Feature | Numeric (Continuous) | Hourly rainfall in millimeters (mm) |
| `snow_1h` | Feature | Numeric (Continuous) | Hourly snowfall in millimeters (mm) |
| `clouds_all` | Feature | Integer | Cloud cover percentage (0–100%) |
| `weather_main` | Feature | Categorical | Short categorical weather summary (e.g., Clouds, Clear, Rain) |
| `weather_description` | Feature | Categorical | Granular weather description (e.g., light rain, overcast clouds) |
| `date_time` | Feature | DateTime | Timestamp of hourly recording (local CST time) |
| `traffic_volume` | Target | Integer | Hourly westbound traffic volume count |

---

## 3. Project Structure

```text
UrbanTrafficFlowOptimization/
│
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv       # Primary raw dataset (48,204 rows)
├── models/
│   └── traffic_volume_rf_model.joblib            # Production Random Forest pipeline & metadata
├── outputs/
│   ├── figures/                                  # Publication-quality diagnostic figures
│   │   ├── actual_vs_predicted.png               # Regression model test set fit comparison
│   │   ├── average_traffic_by_day.png
│   │   ├── average_traffic_by_hour.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_importance.png                # Top Gini impurity feature rankings
│   │   ├── monthly_traffic.png
│   │   ├── prediction_residuals.png              # Residual scatter and error distribution
│   │   ├── rainfall_vs_traffic.png
│   │   ├── temperature_vs_traffic.png
│   │   ├── traffic_over_time.png
│   │   ├── weather_traffic.png
│   │   └── weekday_vs_weekend.png
│   ├── correlation_matrix.csv                    # Feature Pearson correlation table
│   ├── daily_traffic_summary.csv                 # Monday–Sunday aggregated statistics
│   ├── eda_summary.txt                           # Detailed statistical findings and caveats
│   ├── hourly_traffic_summary.csv                 # 24-hour diurnal aggregated statistics
│   ├── model_metrics.csv                         # Out-of-sample MAE, RMSE, and R² metrics
│   ├── monthly_traffic_summary.csv                # January–December seasonal statistics
│   ├── weather_description_summary.csv           # Granular weather condition aggregates
│   ├── weather_main_summary.csv                  # Main weather condition aggregates
│   ├── weekday_vs_weekend_summary.csv            # Commute vs. weekend volume aggregates
│   └── yearly_traffic_summary.csv                 # Annual traffic volume statistics
├── reports/
│   ├── final_project_report.md                   # Comprehensive academic capstone report
│   ├── final_verification_report.md              # End-to-end QA reliability audit report
│   └── project_demo_guide.md                     # Step-by-step viva & live evaluation demo guide
├── src/
│   ├── __init__.py                               # Package initialization
│   ├── data_loader.py                            # Dataset loading, path resolution & schema validation
│   ├── eda.py                                    # Data cleaning, feature extraction, analysis & plotting
│   ├── inspector.py                              # Data inspection and descriptive statistics
│   └── model.py                                  # Regression modeling, evaluation, plotting & inference
├── .gitignore                                    # Excludes virtual environments, caches, OS files
├── app.py                                        # Interactive Streamlit Web GUI Dashboard
├── main.py                                       # Unified CLI entry point (--eda, --train, --predict)
├── requirements.txt                              # Verified project dependencies
└── README.md                                     # Project documentation
```

---

## 4. Required Software & Prerequisites

* **Operating System:** Windows 10/11, Linux, or macOS.
* **Python Runtime:** Python 3.10 to 3.11 (Python 3.11 is strongly recommended on Windows for pre-built scientific binary wheel stability).
* **Package Manager:** `pip` (bundled with Python).

---

## 5. Environment Setup & Dependency Installation

### Step 1: Clone or Navigate to the Workspace
```powershell
cd R:\UrbanTrafficFlowOptimization
```

### Step 2: Create a Local Virtual Environment (`.venv`)
Using Python 3.11:
```powershell
py -3.11 -m venv .venv
```
*(On Linux/macOS: `python3.11 -m venv .venv`)*

### Step 3: Activate the Virtual Environment
* **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
* **Windows (Command Prompt):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
* **Linux/macOS:**
  ```bash
  source .venv/bin/activate
  ```

### Step 4: Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verified packages installed:
* `pandas` (>=2.0.0, verified: 3.0.6)
* `numpy` (>=1.24.0, verified: 2.4.6)
* `matplotlib` (>=3.7.0, verified: 3.11.2)
* `seaborn` (>=0.12.0, verified: 0.13.2)
* `scikit-learn` (>=1.3.0, verified: 1.9.1)
* `jupyter` (>=1.0.0, verified: 1.1.1)
* `streamlit` (>=1.30.0, verified: 1.65.0)

---

## 6. How to Run the System

The project provides both an interactive Streamlit Web GUI and a unified Command-Line Interface (CLI). Paths resolve reliably regardless of current working directory.

### A. Launch Interactive Web GUI Dashboard
Start the professional Streamlit web dashboard:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
*(Or simply: `streamlit run app.py` when the virtual environment is activated)*

* **Local Web URL:** `http://localhost:8501`
* **Dashboard Pages:**
  1. **📊 Overview:** Historical observations counter (48,204 rows), mean and peak traffic KPIs, 6-year data date span, longitudinal traffic volume trend, diurnal 24-hour cycle ribbon, and peak period breakdown.
  2. **📈 Traffic Analysis:** Tabbed exploratory views covering Hourly & Daily rhythms, Weekday vs. Weekend contrast (-27.24% volume reduction), Monthly seasonality, Weather condition rankings, Temperature & Rainfall sensitivities, and Pearson Correlation Heatmap.
  3. **🚦 Predict Traffic:** Interactive prediction form supporting custom date, time, temperature (°C), weather category, rainfall (mm), snowfall (mm), cloud cover (%), and holiday status. Includes quick scenario presets (Morning Rush, Evening Rush, Overnight Low Flow, Weekend Afternoon) and real-time model inference with contextual traffic regime indicators and analytical disclaimers.
  4. **🎯 Model Performance:** Side-by-side comparison cards and tables for Baseline Mean, Linear Regression, and Random Forest models (MAE, RMSE, $R^2$), percentage improvement metrics, and diagnostic tabs for Actual vs. Predicted, Residual Diagnostics, and Gini Feature Importance.

---

### B. Command-Line Interface (CLI) Commands

The CLI in `main.py` supports all automated batch inspection, analysis, and prediction tasks:

#### 1. Run Initial Data Inspection (Default)
```powershell
python main.py
```
*(or explicitly using the virtual environment: `.\.venv\Scripts\python.exe main.py`)*

#### 2. Run Full Exploratory Data Analysis (EDA) Pipeline
```powershell
python main.py --eda
```
*(Cleans data, removes 17 duplicates, extracts temporal features, computes aggregations, generates 9 figures in `outputs/figures/`, and exports summary CSVs and `outputs/eda_summary.txt`)*

#### 3. Run Regression Model Training & Evaluation Pipeline
```powershell
python main.py --train
```
*(Sorts chronologically, fits preprocessing strictly on 80% training window, evaluates Baseline, Linear Regression, and Random Forest on out-of-sample test set, saves `outputs/model_metrics.csv`, generates 3 diagnostic figures, and saves fitted pipeline to `models/traffic_volume_rf_model.joblib`)*

#### 4. Run Traffic Volume Prediction (Inference via CLI)
Run model inference on a target timestamp:
```powershell
python main.py --predict "2018-10-15 17:00:00"
```
Or run the default multi-scenario prediction demonstration (morning peak, evening rush, overnight):
```powershell
python main.py --predict
```

#### 5. Specify a Custom Dataset Path
```powershell
python main.py --data-path "path/to/custom/Metro_Interstate_Traffic_Volume.csv" --train
```

---

## 7. Exploratory Data Analysis (EDA) Findings

Running `main.py --eda` executes the verified EDA pipeline:

### 1. Data Cleaning & Deduplication Audit
* **Initial Records:** 48,204 rows.
* **Exact Duplicate Rows:** Exactly **17 exact duplicate rows** (where all 9 column values are identical) were identified and removed, leaving **48,187 unique observations**. *(Note: 7,629 records share timestamps due to multi-condition concurrent weather reporting; these are valid distinct weather observations and are properly preserved).*
* **Raw CSV Immutability:** The raw CSV file [`data/Metro_Interstate_Traffic_Volume.csv`](file:///r:/UrbanTrafficFlowOptimization/data/Metro_Interstate_Traffic_Volume.csv) remains completely unmodified.
* **Zero-Traffic Records:** 2 valid zero-traffic observations (July 23, 2016 at 18:00 and 23:00) were confirmed intact post-cleaning.
* **Holiday Indicator:** `is_holiday` feature created from `holiday`. 48,143 null records were mapped to standard operational non-holidays (`0`), while 61 designated holiday events were mapped to `1`.

### 2. Temporal Traffic Patterns
* **Busiest Hour of Day:** **16:00 (4:00 PM)** with mean volume of **5,663.76 vehicles/hour**.
* **Quietest Hour of Day:** **03:00 (3:00 AM)** with mean volume of **371.09 vehicles/hour**.
* **Bimodal Rush Hours:** Weekdays exhibit two sharp commute surges at **07:00 (morning rush: 6,048 veh/h)** and **16:00 (evening rush: 6,204 veh/h)**.
* **Weekday vs. Weekend Contrast:**
  * **Weekday (Mon–Fri) Mean:** **3,533.29 vehicles/hour**
  * **Weekend (Sat–Sun) Mean:** **2,570.70 vehicles/hour** (~27.2% reduction relative to weekdays).
  * Weekends display a gentle unimodal midday plateau reaching ~4,350 veh/h between 12:00 and 16:00 with zero morning rush surge.
* **Busiest Day of Week:** **Friday** (mean: 3,655.33 veh/h), followed closely by Thursday (3,637.93 veh/h).
* **Quietest Day of Week:** **Sunday** (mean: 2,369.09 veh/h).

### 3. Weather & Environmental Observations
* **Primary Weather Breakdown:**
  * `Clouds`: 3,618.0 veh/h (N=15,158)
  * `Haze`: 3,502.1 veh/h (N=1,360)
  * `Rain`: 3,317.9 veh/h (N=5,672)
  * `Drizzle`: 3,292.2 veh/h (N=1,820)
  * `Clear`: 3,055.6 veh/h (N=13,384)
  * `Snow`: 3,016.3 veh/h (N=2,875)
  * `Thunderstorm`: 2,999.4 veh/h (N=1,033)
  * `Mist`: 2,933.3 veh/h (N=5,949)
  * `Fog`: 2,703.7 veh/h (N=912)
  * `Squall`: 2,061.8 veh/h (N=4)
* **Precipitation:** Comparing dry hours (rain = 0 mm, mean: 3,257 veh/h, N=44,720) with rainy hours (rain > 0 mm, mean: 3,292 veh/h, N=3,467) reveals that moderate rain alone does not reduce road volume. However, within active rainfall events, increasing rain intensity exhibits a negative slope (slope = -61.8 veh/h per mm).
* **Temperature Correlation:** Warmer temperatures correlate with higher traffic volume ($r = +0.130$), reflecting higher daytime and summer travel activity.

### 4. Pearson Correlation Highlights
* `hour`: **$r = +0.352$** (strongest positive linear driver)
* `is_weekend`: **$r = -0.219$** (strongest negative linear driver)
* `day_of_week`: **$r = -0.150$**
* `temp_c`: **$r = +0.130$**
* `clouds_all`: **$r = +0.067$**
* `is_holiday`: **$r = -0.043$**
* `rain_1h`: **$r = +0.005$**
* `snow_1h`: **$r = +0.001$**

---

## 8. Diagnostic Figures in `outputs/figures/`

All 12 publication-quality figures are automatically generated and saved with consistent styling:

| Figure | Category | Description |
|---|---|---|
| `traffic_over_time.png` | EDA | Longitudinal daily and 30-day moving average volume from 2012 to 2018; highlights the 2014–2015 tracking gap. |
| `average_traffic_by_hour.png` | EDA | Diurnal traffic volume across 24 hours with ±1 std ribbon and peak rush hour callouts. |
| `average_traffic_by_day.png` | EDA | Weekly profile across Monday through Sunday with weekday/weekend color-coding. |
| `weekday_vs_weekend.png` | EDA | Overlay of bimodal weekday curve vs. unimodal weekend curve over 24 hours. |
| `monthly_traffic.png` | EDA | Monthly seasonality across January through December showing peak summer/autumn travel. |
| `weather_traffic.png` | EDA | Horizontal bar ranking of traffic volume across `weather_main` categories with sample counts. |
| `temperature_vs_traffic.png` | EDA | Scatter plot and 5°C binned mean curve of volume vs. temperature in Celsius, annotating 0 K sensor artifacts. |
| `rainfall_vs_traffic.png` | EDA | Dual panel: Dry vs. Rainy bar comparison, plus scatter regression of volume vs. rainfall (0–60 mm). |
| `correlation_heatmap.png` | EDA | Lower-triangle annotated Pearson correlation matrix heatmap for numeric features. |
| `actual_vs_predicted.png` | Modeling | Out-of-sample scatter (Actual vs. Predicted) and 2-week continuous time series tracking. |
| `prediction_residuals.png` | Modeling | Residuals vs. fitted values scatter (homoscedasticity check) and residual error distribution histogram. |
| `feature_importance.png` | Modeling | Random Forest Gini impurity feature importances ranking top temporal and meteorological drivers. |

---

## 9. Machine Learning Regression Modeling

The predictive modeling module in `src/model.py` provides an end-to-end regression pipeline for forecasting hourly traffic volume.

### 1. Target Variable & Leakage Prevention
* **Target Variable:** `traffic_volume` (vehicles per hour).
* **Target Leakage Safeguards:**
  * Neither `traffic_volume` nor any future traffic metrics are included in the feature set.
  * Autoregressive lag features of `traffic_volume` were explicitly avoided to ensure the model functions as a pure environmental and schedule-based forecaster, preventing data leakage across collection discontinuities.

### 2. Feature Engineering & Preprocessing
The model ingests 17 engineered and raw attributes:
* **Calendar & Schedule:**
  * `hour` (0–23) and cyclical transformations `hour_sin`, `hour_cos` ($2\pi \cdot \text{hour} / 24$).
  * `day_of_week` (0–6) and cyclical transformations `day_sin`, `day_cos` ($2\pi \cdot \text{day} / 7$).
  * `month` (1–12) and cyclical transformations `month_sin`, `month_cos` ($2\pi \cdot \text{month} / 12$).
  * `year` (linear trend indicator).
  * `is_weekend` (binary flag: 1 for Saturday/Sunday, 0 for Monday–Friday).
  * `is_holiday` (binary flag: 1 for designated holidays, 0 for regular days).
* **Meteorological Features:**
  * `temp_c` (temperature in Celsius, converted from Kelvin). 0 Kelvin sensor artifacts (-273.15°C) are imputed using the **training partition median** ($10.0^\circ\text{C}$).
  * `rain_1h` (hourly precipitation in mm, physically winsorized/clipped at 100 mm based on training distribution).
  * `snow_1h` (hourly snowfall in mm).
  * `clouds_all` (cloud cover percentage, 0–100%).
  * `weather_main` (categorical weather summary, One-Hot Encoded with `handle_unknown="ignore"`).

* **Strict Training-Set Isolation:**
  * All scalers (`StandardScaler`), encoders (`OneHotEncoder`), and cleaning parameters (temperature median, rain clipping ceiling) are fitted exclusively on the training partition.
  * No test-set statistics leak into preprocessing.

### 3. Chronological Train / Test Split Strategy
Random shuffling is inappropriate for time-series and schedule-based traffic data because it introduces temporal lookahead leakage. The pipeline enforces a **chronological split**:
* Records are sorted strictly by `date_time`.
* **Training Partition (Oldest 80%):** 38,549 observations spanning **October 2, 2012 to November 1, 2017**.
* **Test Partition (Latest 20%):** 9,638 observations spanning **November 1, 2017 to September 30, 2018**.
* Test data is held out strictly for final unbiased evaluation; no hyperparameter tuning is conducted on the test period.

### 4. Regression Models Evaluated
1. **Baseline Model (Training Set Mean):**
   * Predicts a constant value equal to the mean training traffic volume ($\bar{y}_{\text{train}} = 3,257.77\text{ veh/h}$).
   * Serves as the zero-intelligence benchmark.
2. **Linear Regression:**
   * Ordinary Least Squares (OLS) with standardized numeric features and one-hot encoded weather categories.
   * Serves as the linear parametric baseline.
3. **Random Forest Regressor:**
   * Ensemble of 100 decision trees (`n_estimators=100`, `max_depth=18`, `min_samples_split=5`, `min_samples_leaf=4`, `random_state=42`, `n_jobs=-1`).
   * Captures high-order non-linear interactions between hour of day, day of week, and weather conditions.

### 5. Out-of-Sample Evaluation Metrics
The actual test set metrics (exported in [`outputs/model_metrics.csv`](file:///r:/UrbanTrafficFlowOptimization/outputs/model_metrics.csv)) are:

| Model | MAE (veh/h) | RMSE (veh/h) | $R^2$ Score | Interpretation |
|---|---|---|---|---|
| **Baseline (Train Mean)** | 1,722.50 | 1,967.61 | -0.000006 | Zero explanatory power; confirms heavy diurnal variance. |
| **Linear Regression** | 826.21 | 1,050.57 | 0.7149 | Captures linear rhythms and explains 71.5% of variance, but struggles with acute bimodal rush hours. |
| **Random Forest Regressor** | **300.69** | **532.98** | **0.9266** | **Explains 92.7% of out-of-sample variance; reduces MAE by 82.54% vs baseline and 63.61% vs linear regression.** |

* **Exact Error Reductions:**
  - Random Forest vs. Baseline: **82.54% MAE reduction**, **72.91% RMSE reduction**.
  - Linear Regression vs. Baseline: **52.03% MAE reduction**, **46.61% RMSE reduction**.
  - Random Forest vs. Linear Regression: **63.61% MAE reduction**, **49.27% RMSE reduction**.

### 6. Honest Evaluation & Key Model Insights
* **Why the Baseline Fails:** Traffic volume oscillates between under 400 veh/h (night) and over 6,000 veh/h (rush hour). A constant mean estimate yields an unacceptable MAE of 1,722.50 veh/h.
* **Linear vs. Non-linear Performance:** Linear Regression achieves a respectable $R^2$ of 0.7149 due to cyclical $\sin/\cos$ harmonic features. However, it overpredicts late-night traffic and blunts the sharpness of peak rush hours.
* **Superiority of Random Forest:** The Random Forest Regressor captures the sharp threshold shifts of morning and evening rush hours, weekday vs. weekend divergence, and nonlinear weather interactions, reducing RMSE by ~49.3% compared to Linear Regression.
* **Feature Importance & Causation Caveats:**
  - Random Forest impurity importance (Mean Decrease in Impurity / Gini importance) reflects how frequently an attribute was selected to partition variance in decision tree splits; **it does not prove physical causality**. Weather conditions or cloud cover do not "cause" traffic in isolation; they correlate with human travel behavior and daylight schedules.
  - Collinear features divide importance across tree splits: `hour_cos` (66.4%), `hour` (14.0%), and `hour_sin` (5.1%) split the diurnal signal, collectively accounting for **>85.5%** of model splits.
* **Generalization Over Calendar Years:**
  - The test set contains 7,949 samples in 2018—a calendar year unseen during the 2012–2017 training period.
  - Decision trees cannot extrapolate linear trends beyond training bounds ($year > 2017$). However, because `year` contributed only **0.5%** of feature importance, the model relies on recurring cyclical hour, day-of-week, and seasonal patterns, enabling it to generalize reliably across unseen future dates.
* **Residual Diagnostics:**
  - Residual error distribution is centered tightly near zero ($\mu = -47.0\text{ veh/h}$, $\sigma = 530.9\text{ veh/h}$).
  - Residuals display slight heteroscedasticity at high traffic volumes (>5,500 veh/h), where unobserved incident-induced bottlenecks occasionally cause unexpected capacity drops.

### 7. Reusable Model Artifact & Prediction Interface
The fitted Random Forest pipeline is packaged and serialized to [`models/traffic_volume_rf_model.joblib`](file:///r:/UrbanTrafficFlowOptimization/models/traffic_volume_rf_model.joblib) alongside its preprocessing metadata.

Inference can be executed via Python:
```python
from pathlib import Path
from src.model import predict_traffic_volume

model_path = Path("models/traffic_volume_rf_model.joblib")

# Example: Forecast traffic for a Monday evening rush hour
query = {
    "date_time": "2018-10-15 17:00:00",
    "temp_c": 12.5,
    "weather_main": "Clear",
    "rain_1h": 0.0,
    "clouds_all": 20,
    "is_holiday": 0,
}

result = predict_traffic_volume(model_path, query)
print(f"Predicted Volume: {result['predicted_traffic_volume']:.0f} vehicles/hour")
# Output: ~5,781 vehicles/hour
```

---

## 10. Project Deliverables & Academic Reports

All documentation and submission materials have been verified and saved:

* [**Final Project Report (`reports/final_project_report.md`)**](file:///r:/UrbanTrafficFlowOptimization/reports/final_project_report.md): 15-section academic capstone report detailing problem statement, methodology, data cleaning audit, EDA findings, chronological split, metrics, residual diagnostics, and references.
* [**End-to-End Verification Report (`reports/final_verification_report.md`)**](file:///r:/UrbanTrafficFlowOptimization/reports/final_verification_report.md): Comprehensive QA audit report covering cryptographic hash validation, CLI test logs, recalculated metrics, and submission readiness verdict.
* [**Demonstration & Viva Guide (`reports/project_demo_guide.md`)**](file:///r:/UrbanTrafficFlowOptimization/reports/project_demo_guide.md): Step-by-step viva guide with exact CLI commands, figure showcases, Q&A defense points, and live prediction walkthroughs.
* [**Evaluation Metrics (`outputs/model_metrics.csv`)**](file:///r:/UrbanTrafficFlowOptimization/outputs/model_metrics.csv): Exact out-of-sample MAE, RMSE, and $R^2$ values.
* [**EDA Statistical Summary (`outputs/eda_summary.txt`)**](file:///r:/UrbanTrafficFlowOptimization/outputs/eda_summary.txt): Complete textual summary of data cleaning and temporal patterns.

---

## 11. Current Implementation Status

* [x] **Phase 1: Workspace Inspection** — Audited filesystem; identified dataset in `data/`.
* [x] **Phase 2: Project Architecture** — Standard layout created (`data/`, `models/`, `outputs/figures/`, `reports/`, `src/`, `main.py`, `.gitignore`, `requirements.txt`).
* [x] **Phase 3: Python Environment & Dependencies** — `.venv` created with Python 3.11.9; `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib` verified.
* [x] **Phase 4: Data Inspection Implementation** — Modular data loader and schema inspector in `src/data_loader.py` and `src/inspector.py`.
* [x] **Phase 5: Execution & CLI Validation** — Tested standard and custom dataset paths from various working directories.
* [x] **Phase 6: Exploratory Data Analysis (EDA)** — Cleaned data, removed 17 duplicate rows, extracted features, generated 9 figures, 8 summary CSVs, and `outputs/eda_summary.txt`.
* [x] **Phase 7: Regression Modeling & Evaluation** — Chronological 80/20 partition, leakage prevention, Baseline/Linear Regression/Random Forest comparison, 3 diagnostic figures, `outputs/model_metrics.csv`, model serialization to `models/`, and `--train` / `--predict` CLI integration.
* [x] **Phase 8: Submission Quality Audit & Deliverables** — Verification of all claims, scientific qualification of feature importance and sensor anomalies, creation of academic report and demonstration guide.

---

## 12. Known Limitations & Analytical Caveats

1. **Unidirectional Spatial Scope:** The dataset monitors strictly **westbound** traffic on I-94 at ATR Station 301 between Minneapolis and St. Paul. It cannot forecast eastbound traffic or network-wide routing dynamics.
2. **Aggregated Hourly Resolution:** Traffic counts are aggregated into 1-hour intervals. Transient 15-minute shockwaves, sudden signal queues, and immediate incident onset cannot be resolved.
3. **Tracking Discontinuity (2014–2015):** A long data recording hiatus occurs between late 2014 and mid-2015. The chronological train/test split accounts for this by placing all 2017–2018 data in the test set.
4. **Hardware Sensor Artifacts:** 10 records with 0 Kelvin (-273.15°C) and 1 record with 9,831.3 mm rainfall represent sensor faults. These are strictly imputed using training medians and physical bounds, while raw data files remain immutable.
5. **Absence of Real-Time Incident Data:** The dataset lacks traffic incident logs, lane closure notices, construction schedules, and sports/concert attendance. Consequently, model predictions represent **expected regular conditions given schedule and weather**, not guaranteed real-time counts.
6. **Prediction Estimates Disclaimer:** All model predictions are statistical estimates. Downstream traffic management systems should treat predictions as baseline demand forecasts rather than real-time ground truth.

---

## 13. Cloud & Vercel Deployment Architecture

The project provides dual production deployment capabilities: a high-performance **Vercel Serverless Web Application** and a **Streamlit Community Cloud** host for the Python GUI.

### Architecture Overview

```
UrbanTrafficFlowOptimization/
├── api/
│   ├── predict.py            # Vercel serverless Python inference handler (POST /api/predict)
│   └── requirements.txt      # Minimal serverless dependencies (joblib, scikit-learn, etc.)
├── public/                   # Vercel static web assets
│   ├── index.html            # Semantic HTML5 dashboard (Overview, EDA, Live Predictor, Benchmarks)
│   ├── style.css             # Vanilla CSS design system with sleek dark mode and glassmorphism
│   ├── app.js                # Frontend controller with scenario presets and fallback regression
│   └── figures/              # High-resolution publication diagnostic PNG figures (12 plots)
├── app.py                    # Streamlit interactive Python analytics dashboard (local / Streamlit Cloud)
├── vercel.json               # Vercel routing rules (/api/* rewrites and static configuration)
└── models/
    └── traffic_volume_rf_model.joblib  # Trained Random Forest Regressor pipeline (R² = 0.927)
```

### Option A: Deploy to Vercel (Recommended)

1. **One-Command CLI Deployment:**
   From the project directory, run:
   ```bash
   npx vercel
   ```
   Follow the interactive prompts:
   - *Set up and deploy?* `Y`
   - *Which scope?* (Select your Vercel account)
   - *Link to existing project?* `N`
   - *What's your project's name?* `urban-traffic-flow-optimization`
   - *In which directory is your code located?* `./`
   - *Want to modify settings?* `N`

   Vercel will build and assign an instant production URL (e.g. `https://urban-traffic-flow-optimization.vercel.app`).

2. **Deploy via GitHub / GitLab:**
   ```bash
   git add .
   git commit -m "Configure Vercel deployment with serverless API and public dashboard"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/UrbanTrafficFlowOptimization.git
   git push -u origin main
   ```
   In the [Vercel Dashboard](https://vercel.com/new), select **Import Repository**, keep default root settings, and click **Deploy**.

### Option B: Deploy Streamlit GUI to Streamlit Community Cloud

Because Streamlit requires long-lived bidirectional WebSockets, the full Python interface in [`app.py`](file:///r:/UrbanTrafficFlowOptimization/app.py) can be hosted on Streamlit Community Cloud:
1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app**, select your repository, set the **Main file path** to `app.py`, and click **Deploy**.


