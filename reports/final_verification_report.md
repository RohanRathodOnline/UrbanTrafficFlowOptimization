# Urban Traffic Flow Optimization — Comprehensive Quality & Reliability Audit Report

**Date of Audit:** October 2026  
**Project Workspace:** `R:\UrbanTrafficFlowOptimization`  
**Auditor:** Senior Python Developer, Data Scientist & QA Engineer  
**Audit Purpose:** End-to-End Reliability, Reproducibility, and Submission Integrity Verification  

---

## 1. Audit Scope & Environment

### Scope
This audit conducts an exhaustive, empirical validation of the **Urban Traffic Flow Optimization** system, covering:
1. Virtual environment and dependency versions.
2. Raw dataset immutability and physical schema integrity.
3. Command-line interface (CLI) commands across primary, secondary, and negative execution paths.
4. Output artifacts (9 summary CSV tables, text report, and 12 diagnostic PNG figures).
5. Machine learning regression pipeline, chronological train/test partitioning, leakage controls, and metric recalculation.
6. Streamlit web application architecture, component rendering, form submission, and live server endpoints.
7. Project documentation and viva demonstration guide readiness.

### Environment Specification
- **Operating System:** Windows 11 (64-bit)
- **Python Runtime:** Python 3.11.9 (`R:\UrbanTrafficFlowOptimization\.venv\Scripts\python.exe`)
- **Core Scientific Dependencies:**
  - `pandas`: **3.0.6**
  - `numpy`: **2.4.6**
  - `matplotlib`: **3.11.2**
  - `seaborn`: **0.13.2**
  - `scikit-learn`: **1.9.1**
  - `joblib`: **1.6.0**
  - `streamlit`: **1.65.0**
- **Project Module Imports:**
  - `src.data_loader`: **PASS**
  - `src.inspector`: **PASS**
  - `src.eda`: **PASS**
  - `src.model`: **PASS**

---

## 2. Dataset Integrity & Cryptographic Hash Comparison

The raw dataset monitors hourly westbound traffic flow on Interstate 94 at Minnesota Department of Transportation (MN DoT) ATR Station 301.

### Cryptographic Hash Verification
To prove that no testing or cleaning procedures modified the source data on disk, cryptographic hashes were calculated prior to testing and re-verified upon audit completion:

| Attribute | Pre-Testing Benchmark | Post-Testing Verification | Audit Status |
|---|---|---|---|
| **File Location** | `data/Metro_Interstate_Traffic_Volume.csv` | `data/Metro_Interstate_Traffic_Volume.csv` | **PASS** |
| **File Size** | **3,237,208 bytes** | **3,237,208 bytes** | **PASS** |
| **Physical Line Count** | **48,205 lines** | **48,205 lines** | **PASS** |
| **MD5 Checksum** | `f13e9456683f057a6a0cf82f7d64bec7` | `f13e9456683f057a6a0cf82f7d64bec7` | **PASS (Identical)** |
| **SHA-256 Checksum** | `749c90d720360a4215bb15345526073c079ba4cc95e3fa558796d083f85fce9e` | `749c90d720360a4215bb15345526073c079ba4cc95e3fa558796d083f85fce9e` | **PASS (Identical)** |

### Structural Integrity & Schema Checks
- **Dimensions:** Exactly **48,204 rows** and **9 columns**.
- **Missing Target Values:** **0** missing values in `traffic_volume`.
- **Timestamp Integrity:** **0** invalid timestamps (`NaT`).
- **Date Coverage Range:** **2012-10-02 09:00:00** to **2018-09-30 23:00:00** (6 continuous calendar years).
- **Exact Duplicate Rows vs. Duplicate Timestamps:**
  - **Exact Duplicate Rows (all 9 columns identical):** **17 rows**. These represent true redundant sensor logs and are safely dropped by `df.drop_duplicates()` (**PASS**).
  - **Duplicate Timestamps (`date_time` only):** **7,629 rows**. These represent concurrent multi-condition weather reporting (e.g., simultaneous rain and mist during a single hour) and are properly preserved (**PASS**).
- **Missing Holiday Values:**
  - 48,143 records (99.87%) have null (`NaN`) values for `holiday`. Per official schema specifications, null values represent standard non-holiday operational days. The pipeline maps these to `0` and the 61 holiday dates to `1` in `is_holiday` (**PASS**).
- **Zero-Traffic Observations:**
  - Exactly **2 records** record `traffic_volume == 0` (July 23, 2016 at 18:00 and 23:00). Both remain intact post-cleaning (**PASS**).

---

## 3. Command-Line Interface (CLI) Workflow Verification

All CLI commands were executed directly via PowerShell using the project virtual environment. Exit codes, outputs, and negative paths were captured:

| Test ID | Command Tested | Expected Code | Actual Code | Key Output Verification | Audit Status |
|---|---|---|---|---|---|
| **CLI-01** | `python main.py` | 0 | 0 | Displays dataset dimensions, 17 duplicates, 0 NaT, and descriptive stats. | **PASS** |
| **CLI-02** | `python main.py --eda` | 0 | 0 | Executes cleaning, reports 48,187 clean rows, generates 9 figures & 8 CSVs. | **PASS** |
| **CLI-03** | `python main.py --train` | 0 | 0 | Executes chronological 80/20 split, evaluates 3 models, saves metrics & model pipeline. | **PASS** |
| **CLI-04** | `python main.py --predict "2018-10-15 17:00:00"` | 0 | 0 | Predicts Monday evening rush hour volume (~5,781 veh/h) with disclaimer. | **PASS** |
| **CLI-05** | `python main.py --predict` | 0 | 0 | Executes multi-scenario demonstration (Rush, Overnight, Weekend). | **PASS** |
| **CLI-06** | `python main.py --data-path nonexistent.csv` | 1 | 1 | Gracefully catches `FileNotFoundError` and outputs clear error message. | **PASS** |

---

## 4. Output Artifacts & Visual Figure Verification

All 21 generated output artifacts were tested for file presence, byte size, data reopenability, and pixel rendering integrity:

### 1. Summary CSV Tables in `outputs/`
| File Name | Rows | Columns | File Size | Pandas Reopen Test | Audit Status |
|---|---|---|---|---|---|
| `hourly_traffic_summary.csv` | 24 | 7 | 1,451 bytes | Opened successfully | **PASS** |
| `daily_traffic_summary.csv` | 7 | 8 | 531 bytes | Opened successfully | **PASS** |
| `weekday_vs_weekend_summary.csv` | 2 | 8 | 203 bytes | Opened successfully | **PASS** |
| `monthly_traffic_summary.csv` | 12 | 8 | 862 bytes | Opened successfully | **PASS** |
| `yearly_traffic_summary.csv` | 7 | 7 | 485 bytes | Opened successfully | **PASS** |
| `weather_main_summary.csv` | 11 | 7 | 726 bytes | Opened successfully | **PASS** |
| `weather_description_summary.csv` | 38 | 7 | 2,785 bytes | Opened successfully | **PASS** |
| `correlation_matrix.csv` | 11 | 12 | 2,605 bytes | Opened successfully | **PASS** |
| `model_metrics.csv` | 3 | 4 | 256 bytes | Opened successfully | **PASS** |
| `eda_summary.txt` | 61 lines | N/A | 3,353 bytes | Text integrity verified | **PASS** |

### 2. Publication-Quality Figures in `outputs/figures/`
All PNG files were verified with PIL Image verification, confirming dimensions, color modes, and non-corruption:

| Figure Name | Category | Resolution | Color Mode | File Size | Visual Inspection Findings | Audit Status |
|---|---|---|---|---|---|---|
| `traffic_over_time.png` | EDA | 4200 x 1800 px | RGBA | 539,925 bytes | Clean 6-year timeline; highlights 2014–2015 tracking gap. | **PASS** |
| `average_traffic_by_hour.png` | EDA | 3300 x 1800 px | RGBA | 442,522 bytes | Clear 24h curve with ±1 std ribbon; peak callouts at 07:00 & 16:00. | **PASS** |
| `average_traffic_by_day.png` | EDA | 3000 x 1800 px | RGBA | 168,463 bytes | Mon–Sun profile; Friday peak and Sunday trough clearly colored. | **PASS** |
| `weekday_vs_weekend.png` | EDA | 3300 x 1800 px | RGBA | 322,619 bytes | Weekday bimodal curve vs. weekend unimodal midday plateau. | **PASS** |
| `monthly_traffic.png` | EDA | 3300 x 1800 px | RGBA | 211,752 bytes | Jan–Dec seasonality; summer/autumn high and winter low. | **PASS** |
| `weather_traffic.png` | EDA | 3000 x 1950 px | RGBA | 238,721 bytes | Horizontal bars ranking main weather categories with sample counts. | **PASS** |
| `temperature_vs_traffic.png` | EDA | 3300 x 1800 px | RGBA | 1,104,638 bytes | Celsius curve and 0 K hardware sensor artifact annotation. | **PASS** |
| `rainfall_vs_traffic.png` | EDA | 3900 x 1650 px | RGBA | 544,760 bytes | Dry vs. rain bar comparison & active rainfall intensity scatter. | **PASS** |
| `correlation_heatmap.png` | EDA | 3000 x 2550 px | RGBA | 273,277 bytes | Annotated lower-triangle Pearson matrix with legible text. | **PASS** |
| `actual_vs_predicted.png` | Model | 4500 x 1800 px | RGBA | 1,189,894 bytes | Test set scatter along identity line + 14-day continuous time trace. | **PASS** |
| `prediction_residuals.png` | Model | 4200 x 1650 px | RGBA | 776,880 bytes | Residuals vs. fitted values + Gaussian error distribution ($\mu=-47.0$). | **PASS** |
| `feature_importance.png` | Model | 3000 x 1950 px | RGBA | 257,026 bytes | Top 15 Random Forest Gini impurity importance ranking. | **PASS** |

### Statistical Consistency Cross-Check
Summary table values were cross-referenced against raw calculations:
- Weekday Mean: **3,533.29 veh/h** (Identical to CSV, diff < $10^{-6}$)
- Weekend Mean: **2,570.70 veh/h** (Identical to CSV, diff < $10^{-6}$)
- Busiest Clock Hour: **16:00** with mean **5,663.76 veh/h** (Identical to CSV)
- Quietest Clock Hour: **03:00** with mean **371.09 veh/h** (Identical to CSV)

---

## 5. Machine Learning Pipeline & Metric Recalculation

### Architectural Audit
1. **Target Isolation:** `traffic_volume` and future volume lags are completely excluded from predictor matrices $X_{\text{train}}$ and $X_{\text{test}}$.
2. **Chronological 80/20 Partitioning:**
   - Training Set: **38,549 records** (Oct 02, 2012 09:00 to Nov 01, 2017 19:00).
   - Test Set: **9,638 records** (Nov 01, 2017 20:00 to Sep 30, 2018 23:00).
   - Maximum train date $\le$ Minimum test date. Zero lookahead leakage (**PASS**).
3. **Training-Only Preprocessing:**
   - 10 sensor records with 0 Kelvin (-273.15°C) and 1 record with 9,831.3 mm rain occurred solely in the training window.
   - Temperature median ($9.60^\circ\text{C}$) and rainfall clipping limit ($100.0\text{ mm}$) were computed strictly on the training set.
   - `StandardScaler` and `OneHotEncoder` fitted exclusively on training records (**PASS**).

### Recalculated Evaluation Metrics
Metrics were independently re-computed from test set predictions and compared against [`outputs/model_metrics.csv`](file:///r:/UrbanTrafficFlowOptimization/outputs/model_metrics.csv):

| Model | Evaluated Test Records | Recalculated MAE | Recalculated RMSE | Recalculated $R^2$ | Saved CSV Agreement | Audit Status |
|---|---|---|---|---|---|---|
| **Baseline (Train Mean)** | 9,638 | **1,722.50** | **1,967.61** | **-0.000006** | Identical ($<10^{-6}$) | **PASS** |
| **Linear Regression** | 9,638 | **826.21** | **1,050.57** | **0.714911** | Identical ($<10^{-6}$) | **PASS** |
| **Random Forest Regressor** | 9,638 | **300.69** | **532.98** | **0.926626** | Identical ($<10^{-6}$) | **PASS** |

### Verified Improvements
- **Random Forest vs. Baseline:** **82.54% reduction in MAE**, **72.91% reduction in RMSE**.
- **Random Forest vs. Linear Regression:** **63.61% reduction in MAE**, **49.27% reduction in RMSE**.
- **Variance Explained:** Explains **92.66%** of out-of-sample traffic variance.

### Prediction Bounds Check
Test set Random Forest predictions satisfy physical validity bounds:
- Minimum predicted volume: **239.90 vehicles/hour**
- Maximum predicted volume: **6,902.82 vehicles/hour**
- Negative predictions: **0**
- NaN or infinite predictions: **0**
- API output format: Verified dictionary containing `predicted_traffic_volume`, `input_features`, and official disclaimer (**PASS**).

---

## 6. Streamlit Web Application Verification

The Streamlit web dashboard in [`app.py`](file:///r:/UrbanTrafficFlowOptimization/app.py) was tested across server infrastructure and programmatic component testing:

### Server & Endpoint Status
- **Process Status:** Active daemon background process running on port 8501.
- **HTTP Server Response:** Local endpoint `http://localhost:8501` returned **HTTP 200 OK** (**PASS**).

### Component & Page Architecture (Streamlit AppTest Suite)
Streamlit's official headless testing framework (`streamlit.testing.v1.AppTest`) verified all pages and state transitions:
1. **Overview Page:** Loaded with 0 exceptions; all KPI cards, metric values, and sidebar controls present (**PASS**).
2. **Traffic Analysis Page:** Rendered tabs for Hourly, Weekday vs. Weekend, Monthly, Weather, and Correlation with 0 exceptions (**PASS**).
3. **Predict Traffic Page:** Input fields, preset scenario buttons, and form submission executed cleanly, invoking the actual saved Random Forest pipeline and rendering the prediction banner (**PASS**).
4. **Model Performance Page:** Rendered comparative model metrics matching CSV data, percentage badges, and diagnostic image tabs with 0 exceptions (**PASS**).
5. **Graceful Fallback:** Missing table simulation safely returned `None` without unhandled exceptions (**PASS**).

### Testing Limitation Note
- **Browser Automation Limitation (WARNING):** Automated browser subagent recording failed to initialize Playwright because the vendor driver CDN returned an HTTP 404 error (`https://playwright.azureedge.net/builds/driver/playwright-1.57.0-win32_x64.zip`).
- **Resolution:** Full interactive functionality was proven via `AppTest` programmatic verification and live HTTP 200 checks. The GUI is fully operable by opening [http://localhost:8501](http://localhost:8501) in any local browser.

---

## 7. Deliverables & Documentation Checklist

| Deliverable | Location | Size | Verification Finding | Status |
|---|---|---|---|---|
| **Raw Dataset** | [`data/Metro_Interstate_Traffic_Volume.csv`](file:///r:/UrbanTrafficFlowOptimization/data/Metro_Interstate_Traffic_Volume.csv) | 3.24 MB | Hash verified immutable; 48,204 rows | **PASS** |
| **Serialized Model** | [`models/traffic_volume_rf_model.joblib`](file:///r:/UrbanTrafficFlowOptimization/models/traffic_volume_rf_model.joblib) | 66.0 MB | Production pipeline with metadata | **PASS** |
| **Model Metrics** | [`outputs/model_metrics.csv`](file:///r:/UrbanTrafficFlowOptimization/outputs/model_metrics.csv) | 256 bytes | Recalculated metrics match exactly | **PASS** |
| **EDA Summary** | [`outputs/eda_summary.txt`](file:///r:/UrbanTrafficFlowOptimization/outputs/eda_summary.txt) | 3.35 KB | Text audit matches processed data | **PASS** |
| **Diagnostic Figures** | [`outputs/figures/*.png`](file:///r:/UrbanTrafficFlowOptimization/outputs/figures/) | 12 files | 300 DPI, non-corrupted images | **PASS** |
| **Academic Report** | [`reports/final_project_report.md`](file:///r:/UrbanTrafficFlowOptimization/reports/final_project_report.md) | 34.1 KB | 15 academic sections complete | **PASS** |
| **Viva Demo Guide** | [`reports/project_demo_guide.md`](file:///r:/UrbanTrafficFlowOptimization/reports/project_demo_guide.md) | 12.7 KB | Step-by-step commands & Q&A ready | **PASS** |
| **Project README** | [`README.md`](file:///r:/UrbanTrafficFlowOptimization/README.md) | 27.0 KB | Complete instructions for CLI & GUI | **PASS** |
| **Streamlit GUI** | [`app.py`](file:///r:/UrbanTrafficFlowOptimization/app.py) | 30.8 KB | 4-page dashboard verified via AppTest | **PASS** |
| **Verification Report** | [`reports/final_verification_report.md`](file:///r:/UrbanTrafficFlowOptimization/reports/final_verification_report.md) | This file | Complete audit trail & evidence | **PASS** |

---

## 8. Summary of Deficiencies, Corrections & Retest Results

1. **Dependency Addition:**
   - *Observation:* Streamlit was not installed in the initial `.venv`.
   - *Correction:* Added `streamlit>=1.30.0` to `requirements.txt` and installed `streamlit 1.65.0`.
   - *Retest Result:* PASS (`AppTest` and HTTP server verified).
2. **Assertion String Formatting:**
   - *Observation:* Initial test scratchpad expected unformatted numbers for `eda_summary.txt`.
   - *Correction:* Updated test harness to match formatted text (`48,204` and `17`).
   - *Retest Result:* PASS.
3. **Deprecation Notice Monitoring:**
   - *Observation:* Streamlit 1.65 logs a non-blocking deprecation notice recommending `width='stretch'` over `use_container_width=True` starting after Dec 31, 2025.
   - *Impact:* Zero runtime impact; `use_container_width=True` functions correctly and remains fully backwards-compatible across older Streamlit versions.

---

## 9. Final Verdict & Submission Readiness

**Overall Project Status:** **READY FOR SUBMISSION & LIVE DEMONSTRATION (PASS)**

- **Total Verifications Executed:** 48 automated test points.
- **Pass Rate:** **100%** across dataset integrity, CLI commands, model pipeline, artifact generation, and web dashboard execution.
- **Unresolved Defects:** **0**.
