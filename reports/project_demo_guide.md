# Urban Traffic Flow Optimization: Project Demonstration & Viva Guide

This guide outlines the exact, step-by-step commands to run a complete live demonstration of the **Urban Traffic Flow Optimization** project during an evaluation, viva, or college presentation.

All commands are executed from the project root: `R:\UrbanTrafficFlowOptimization`.

---

## 0. Prerequisites & Environment Check

Before beginning the demonstration, confirm the virtual environment is intact and active:

### PowerShell (Windows)
```powershell
cd R:\UrbanTrafficFlowOptimization
.\.venv\Scripts\Activate.ps1
python --version
```
*Expected Python Version:* `Python 3.11.x`

---

## 1. Step 1: Initial Dataset Inspection

Demonstrate that the dataset loads correctly, schema validation passes, and descriptive statistics are computed without errors.

### Command
```powershell
python main.py
```

### What to Explain to Evaluators
1. **Dataset Location:** Loaded from [`data/Metro_Interstate_Traffic_Volume.csv`](file:///r:/UrbanTrafficFlowOptimization/data/Metro_Interstate_Traffic_Volume.csv) (3.24 MB).
2. **Benchmark Integrity:** Exactly **48,204 rows** and **9 columns**, matching the official UCI repository benchmark.
3. **Target Variable:** `traffic_volume` spans from 0 to 7,280 vehicles/hour with a mean of ~3,260 vehicles/hour.
4. **Data Cleanliness:** 0 invalid timestamps (`NaT`), 0 missing target values, and original raw CSV remains completely unmodified.

---

## 2. Step 2: Exploratory Data Analysis (EDA) Pipeline

Demonstrate data cleaning, feature extraction, duplicate audit, and generation of all diagnostic charts.

### Command
```powershell
python main.py --eda
```

### Key Audit Points to Highlight
* **Deduplication Audit:** Exactly **17 exact duplicate rows** were detected and safely removed (reducing count to **48,187 unique records**). Point out that 7,629 records share timestamps due to multi-condition weather reporting (e.g., simultaneous rain and mist), which are preserved.
* **Holiday Mapping:** 48,143 null values in `holiday` represent normal operational days and were correctly mapped to non-holidays (`is_holiday = 0`), while 61 designated holiday events were mapped to `is_holiday = 1`.
* **Zero-Traffic Points:** 2 valid zero-traffic observations on July 23, 2016 were confirmed intact.

### Figures to Showcase to the Evaluators
Open and show the publication-quality figures located in [`outputs/figures/`](file:///r:/UrbanTrafficFlowOptimization/outputs/figures/):
1. **[`average_traffic_by_hour.png`](file:///r:/UrbanTrafficFlowOptimization/outputs/figures/average_traffic_by_hour.png):** Shows the diurnal 24-hour cycle. Point out the bimodal morning rush (07:00, ~6,048 veh/h) and evening rush (16:00, ~6,204 veh/h).
2. **[`weekday_vs_weekend.png`](file:///r:/UrbanTrafficFlowOptimization/outputs/figures/weekday_vs_weekend.png):** Compares the weekday bimodal rush curve against the weekend unimodal midday plateau (~27.2% overall volume reduction on weekends).
3. **[`average_traffic_by_day.png`](file:///r:/UrbanTrafficFlowOptimization/outputs/figures/average_traffic_by_day.png):** Displays Friday as the busiest day of the week and Sunday as the quietest.
4. **[`temperature_vs_traffic.png`](file:///r:/UrbanTrafficFlowOptimization/outputs/figures/temperature_vs_traffic.png):** Explains the 10 zero-Kelvin sensor anomalies annotated in the plot and the positive correlation between warm temperatures and traffic volume.
5. **[`correlation_heatmap.png`](file:///r:/UrbanTrafficFlowOptimization/outputs/figures/correlation_heatmap.png):** Demonstrates Pearson correlations, showing `hour` ($r = +0.352$) and `is_weekend` ($r = -0.219$) as the dominant linear drivers.

---

## 3. Step 3: Regression Model Training and Comparison

Demonstrate model training, strict chronological splitting, leak-free preprocessing, and out-of-sample evaluation.

### Command
```powershell
python main.py --train
```

### Key Technical Aspects to Explain
1. **Chronological 80/20 Split (No Random Shuffling):**
   - **Training Set (Oldest 80%):** 38,549 observations (October 2012 to November 2017).
   - **Test Set (Latest 20%):** 9,638 observations (November 2017 to September 2018).
   - *Why not random split?* Random splitting causes lookahead data leakage in time-series data. The chronological split tests real-world forecasting into the future.
2. **Leak-Free Preprocessing:**
   - Standard scalers, one-hot encoders, and outlier thresholds (median temperature of $9.60^\circ\text{C}$ and rain clipping at $100\text{ mm}$) were fitted strictly on training data.
3. **Target Leakage Safeguard:**
   - Traffic volume was not used as an input feature (no autoregressive lags), ensuring the model is a true schedule- and weather-driven forecaster.

### Actual Evaluation Results to Present
Direct the evaluators to [`outputs/model_metrics.csv`](file:///r:/UrbanTrafficFlowOptimization/outputs/model_metrics.csv):

| Model | MAE (veh/h) | RMSE (veh/h) | $R^2$ Score | Assessment |
|---|---|---|---|---|
| **Baseline (Train Mean)** | 1,722.50 | 1,967.61 | -0.000006 | Zero explanatory power ($R^2 \approx 0$). |
| **Linear Regression** | 826.21 | 1,050.57 | 0.7149 | Explains 71.5% of variance with cyclical Fourier terms. |
| **Random Forest Regressor** | **300.69** | **532.98** | **0.9266** | **Explains 92.7% of variance, reducing MAE by 82.5% vs baseline and 63.6% vs linear regression.** |

### Model Diagnostic Plots to Showcase
1. **[`actual_vs_predicted.png`](file:///r:/UrbanTrafficFlowOptimization/outputs/figures/actual_vs_predicted.png):**
   - Shows test set scatter along the diagonal identity line.
   - Shows the 14-day continuous test trace matching actual traffic peaks and valleys.
2. **[`prediction_residuals.png`](file:///r:/UrbanTrafficFlowOptimization/outputs/figures/prediction_residuals.png):**
   - Residual scatter evenly distributed around 0.
   - Histogram showing Gaussian error distribution centered at $\mu = -47.0\text{ veh/h}$.
3. **[`feature_importance.png`](file:///r:/UrbanTrafficFlowOptimization/outputs/figures/feature_importance.png):**
   - Explains that `hour_cos`, `hour`, and `hour_sin` combine for >85% of tree splits.
   - Clarifies that tree impurity importance reflects split utility and collinear feature division rather than physical causality.

---

## 4. Step 4: Live Traffic Volume Prediction (Inference)

Demonstrate real-time inference using the trained and saved Random Forest pipeline.

### Command A: Specific Timestamp Inference
Predict traffic volume for an upcoming Monday evening rush hour:
```powershell
python main.py --predict "2018-10-15 17:00:00"
```

### Command B: Multi-Scenario Batch Demonstration
Execute three pre-configured operational scenarios (evening rush, overnight, and weekend afternoon):
```powershell
python main.py --predict
```

### Expected Live Output
```text
Scenario: Evening Rush Hour (Weekday)
  Input Timestamp         : 2018-10-15 17:00:00
  Weather Conditions      : Clear, 15.0°C, Rain: 0.0mm
  Predicted Traffic Volume: 5,781 vehicles/hour

Scenario: Overnight Low Flow (Weekday 03:00 AM)
  Input Timestamp         : 2018-10-16 03:00:00
  Weather Conditions      : Clouds, 8.0°C, Rain: 0.0mm
  Predicted Traffic Volume: 343 vehicles/hour

Scenario: Weekend Afternoon Leisure Flow (Saturday 14:00 PM)
  Input Timestamp         : 2018-10-20 14:00:00
  Weather Conditions      : Clear, 18.0°C, Rain: 0.0mm
  Predicted Traffic Volume: 4,642 vehicles/hour
```

### Programmatic Python Inference Demo
You can also demonstrate loading the model in an interactive Python session:
```python
from pathlib import Path
from src.model import predict_traffic_volume

model_path = Path("models/traffic_volume_rf_model.joblib")
result = predict_traffic_volume(
    model_path,
    {
        "date_time": "2018-10-15 17:00:00",
        "temp_c": 12.0,
        "weather_main": "Clear",
        "rain_1h": 0.0,
        "clouds_all": 20,
        "is_holiday": 0,
    }
)
print("Predicted:", result["predicted_traffic_volume"], "vehicles/hour")
```

---

## 5. Step 5: Interactive Web GUI Dashboard Demonstration

Demonstrate the Streamlit web dashboard to provide a graphical, interactive presentation.

### Launch Command
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
*Open in web browser:* `http://localhost:8501`

### What to Showcase on Each Page
1. **📊 Overview Page:**
   - Walk through the KPI cards: Total Historical Observations (48,204), Average Volume (~3,260 veh/h), Peak Flow (7,280 veh/h), and 6-Year Date Span.
   - Show the interactive longitudinal trend graph and 24-hour diurnal profile.
   - Highlight the peak traffic summary cards (Morning Rush at 07:00, Evening Peak at 16:00, Overnight lull at 03:00).
2. **📈 Traffic Analysis Page:**
   - Click through the tabs: Hourly & Daily rhythms, Weekday vs. Weekend overlay, Seasonality, Weather categories, and Pearson Correlation Heatmap.
   - Expand the data summary tables to demonstrate underlying data.
3. **🚦 Predict Traffic Page:**
   - Click quick scenario preset buttons: **🌅 Morning Rush**, **🌇 Evening Rush**, **🌙 Overnight**, or **🚗 Weekend**.
   - Adjust sliders (Temperature, Cloud Cover) or weather condition.
   - Click **⚡ Predict Traffic Volume** to execute real-time inference with the saved model.
   - Point out the predicted volume banner, input feature interpretation table, contextual traffic regime classification, and analytical disclaimer.
4. **🎯 Model Performance Page:**
   - Showcase the side-by-side metric comparison cards (Baseline, Linear Regression, Random Forest).
   - Show the 82.54% MAE reduction badge.
   - Click through the diagnostic tabs: **Actual vs. Predicted**, **Residual Diagnostics**, and **Feature Importance**.

---

## 6. Step 6: Saved Model Artifact & File Verification

Show the evaluators where artifacts are stored on disk:

* **Production Model Artifact:** [`models/traffic_volume_rf_model.joblib`](file:///r:/UrbanTrafficFlowOptimization/models/traffic_volume_rf_model.joblib) (~66 MB).
  - Contains the fitted `ColumnTransformer` (StandardScaler + OneHotEncoder), the fitted `RandomForestRegressor`, and training metadata.
* **Evaluation Metrics:** [`outputs/model_metrics.csv`](file:///r:/UrbanTrafficFlowOptimization/outputs/model_metrics.csv).
* **EDA Summary Report:** [`outputs/eda_summary.txt`](file:///r:/UrbanTrafficFlowOptimization/outputs/eda_summary.txt).
* **Comprehensive Project Report:** [`reports/final_project_report.md`](file:///r:/UrbanTrafficFlowOptimization/reports/final_project_report.md).
* **Interactive Streamlit App:** [`app.py`](file:///r:/UrbanTrafficFlowOptimization/app.py).

---

## 7. Viva / Defense Questions & Answers

### Q1: Why did you use a chronological split instead of K-fold cross-validation or random train_test_split?
> **Answer:** In traffic flow and schedule data, randomly shuffling observations leaks temporal patterns from future records into past records. A chronological split reserves the latest 20% of consecutive observations (November 2017 to September 2018) exclusively for testing, reflecting true real-world deployment where future conditions must be forecasted from historical records.

### Q2: Why did Random Forest achieve an $R^2$ of 0.9266 while Linear Regression only reached 0.7149?
> **Answer:** Freeway traffic exhibits sharp non-linear threshold behaviors. For example, 07:00 AM generates heavy traffic on a weekday due to morning commutes, but low traffic on a Sunday. Linear regression assumes additive linear effects and cannot easily capture these higher-order feature interactions without extensive manual interaction terms. Random Forest decision trees naturally partition feature space along interaction boundaries (`hour` $\times$ `is_weekend`).

### Q3: Did you include previous hours' traffic counts as input features?
> **Answer:** No. Autoregressive traffic lags (e.g., $t-1$ volume) were intentionally excluded to prevent target leakage and to ensure the system functions as a robust forecaster capable of predicting volume based solely on schedule and weather forecasts, even if upstream traffic sensors experience transmission outages or data hiatuses (such as the 2014–2015 tracking gap).

### Q4: Does feature importance prove that weather or hour causes traffic?
> **Answer:** No. Random Forest feature importance measures the Mean Decrease in Impurity (Gini importance) across tree splits. It evaluates predictive association, not physical causality. Furthermore, collinear features like `hour`, `hour_cos`, and `hour_sin` divide importance among themselves rather than concentrating on a single attribute.

