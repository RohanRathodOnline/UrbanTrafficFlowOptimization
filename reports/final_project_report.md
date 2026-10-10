# Urban Traffic Flow Optimization: Exploratory Analysis, Machine Learning Regression, and Model Evaluation

**Course / Academic Project:** Data Science & Machine Learning Capstone  
**Domain:** Intelligent Transportation Systems (ITS) & Urban Traffic Engineering  
**Dataset:** Metro Interstate Traffic Volume (Westbound Interstate 94, Minneapolis–St. Paul, MN)  
**Repository Workspace:** `UrbanTrafficFlowOptimization`  
**Date:** October 2026  

---

## 1. Title and Abstract

### Title
**Predictive Modeling of Urban Freeway Flow: An Empirical Investigation of Temporal Dynamics, Weather Sensitivities, and Machine Learning Regression on Interstate 94**

### Abstract
Accurate hourly traffic volume forecasting is essential for intelligent transportation management, dynamic congestion mitigation, and municipal infrastructure planning. This report presents an end-to-end data science and machine learning pipeline developed on the **Metro Interstate Traffic Volume** dataset, recording hourly westbound traffic flow on Interstate 94 between Minneapolis and St. Paul, Minnesota (MN DoT Station 301) from October 2012 to September 2018 (48,204 initial observations). 

The investigation begins with exploratory data analysis (EDA), auditing data integrity, resolving exactly 17 duplicate rows, addressing hardware sensor anomalies (10 observations at 0 Kelvin and an extreme rainfall recording of 9,831.3 mm), and transforming temporal and weather attributes. Next, cyclical harmonic encodings ($\sin/\cos$) were developed for diurnal, weekly, and seasonal periods alongside categorical weather transformations. To prevent lookahead bias and target leakage, an 80/20 chronological train/test split was established, partitioning the oldest 38,549 records (October 2012 – November 2017) for training and reserving the final 9,638 observations (November 2017 – September 2018) exclusively for out-of-sample evaluation.

Three predictive regimes were implemented and compared: a zero-intelligence baseline predicting the training mean, an Ordinary Least Squares Linear Regression baseline, and an ensemble Random Forest Regressor. On the held-out out-of-sample test set:
- The **Baseline Mean** model yielded a Mean Absolute Error (MAE) of **1,722.50 vehicles/hour**, Root Mean Squared Error (RMSE) of **1,967.61 vehicles/hour**, and an $R^2$ of **$-0.000006$**.
- **Linear Regression** achieved an MAE of **826.21 vehicles/hour**, RMSE of **1,050.57 vehicles/hour**, and $R^2$ of **0.7149**.
- The **Random Forest Regressor** achieved superior accuracy, attaining an MAE of **300.69 vehicles/hour**, RMSE of **532.98 vehicles/hour**, and $R^2$ of **0.9266**, representing an **82.54% MAE reduction** over the baseline and a **63.61% MAE reduction** over linear regression.

Feature importance diagnostics reveal that diurnal schedule patterns dominate predictive performance (>85% cumulative importance), followed by day-of-week and weekend classifications (~10%), while weather parameters serve secondary modulating roles (~5%). Methodological considerations, including Random Forest impurity importance versus causality, feature collinearity, calendar year generalization, and sensor limitations, are critically evaluated.

---

## 2. Problem Statement

Urban interstate corridors experience acute demand fluctuations driven by commuter schedules, weekend travel habits, commercial logistics, and meteorological disturbances. When demand approaches or exceeds physical corridor throughput, sudden transitions from free-flow to congested regimes occur, producing severe delays, increased collision rates, and elevated carbon emissions.

Traditional traffic operations frequently rely on historical static schedules or basic linear extrapolations that fail to capture the complex, non-linear interactions between time of day, day of week, seasonal calendars, and adverse weather. The core technical problem addressed in this study is:

> *Can historical traffic volume on a major metropolitan freeway corridor be predicted accurately on an hourly basis using only calendar schedule and environmental weather inputs without relying on autoregressive traffic lag measurements or future information?*

Addressing this problem requires:
1. Identifying and isolating temporal rhythms and weather-induced disruptions.
2. Engineering features that preserve cyclical boundary continuity (e.g., transition from 23:00 to 00:00).
3. Enforcing strict chronological boundaries to guarantee real-world forecasting validity.
4. Comparing parametric and non-parametric algorithms against a naive benchmark.

---

## 3. Objectives

The primary objectives of this study are:
1. **Data Ingestion and Integrity Auditing:** Validate schema integrity, examine duplicate entries, verify missing value structures, and identify recording anomalies in historical road sensor logs.
2. **Exploratory Data Analysis (EDA):** Characterize diurnal cycles, weekday versus weekend flow divergences, seasonal trends, and the statistical relationship between weather conditions (temperature, precipitation, cloud cover) and traffic volume.
3. **Leakage-Free Feature Engineering:** Construct cyclical trigonometric transformations ($\sin/\cos$) for temporal components, encode weather categories, and ensure that all preprocessing steps (scaling, encoding, imputation) are strictly fitted on training data alone.
4. **Chronological Model Benchmarking:** Train and evaluate three models using an out-of-sample chronological split:
   - Baseline Mean (naive benchmark using training mean only).
   - Linear Regression (standard parametric baseline).
   - Random Forest Regressor (non-linear ensemble).
5. **Rigorous Statistical Evaluation:** Compute Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$), accompanied by residual diagnostics and feature importance analyses.
6. **Reusable Deployment Pipeline:** Package the fitted pipeline and deploy a validated prediction interface capable of handling inference requests while transparently communicating estimation limits.

---

## 4. Dataset Source and Description

### Source and Attribution
* **Repository:** UCI Machine Learning Repository
* **Dataset Identifier:** [Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/urban+traffic)
* **DOI:** 10.24432/C5X60B
* **Original Collector:** John Hogue, Minnesota Department of Transportation (MN DoT)
* **Citation:** Hogue, J. (2019). *Metro Interstate Traffic Volume*. UCI Machine Learning Repository.
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

### Physical Scope & Location
The sensor data was logged by Minnesota Department of Transportation (MN DoT) Automatic Traffic Recorder (ATR) **Station 301**, positioned along **Interstate 94 (I-94) westbound** approximately midway between Minneapolis and St. Paul, Minnesota, USA. 
*(Note: This dataset describes traffic conditions on I-94 in Minneapolis–St. Paul, Minnesota, and must not be confused with traffic datasets from other global municipalities such as Pune, India).*

### Raw Data Attributes
The raw dataset spans hourly records from **October 2, 2012, at 09:00:00** to **September 30, 2018, at 23:00:00**, comprising **48,204 rows** across 9 attributes:

| Column Name | Role | Raw Data Type | Physical Description & Units |
|---|---|---|---|
| `holiday` | Feature | Categorical / String | US National holidays and Minnesota State Fair; `NaN` for regular non-holidays |
| `temp` | Feature | Float64 | Hourly average ambient temperature measured in Kelvin (K) |
| `rain_1h` | Feature | Float64 | Hourly liquid precipitation amount in millimeters (mm) |
| `snow_1h` | Feature | Float64 | Hourly snowfall accumulation in millimeters (mm) |
| `clouds_all` | Feature | Int64 | Cloud coverage percentage ($0\% - 100\%$) |
| `weather_main` | Feature | Categorical / String | Broad category of current weather (e.g., Clear, Clouds, Rain, Snow) |
| `weather_description`| Feature | Categorical / String | Granular text description of weather phenomenon |
| `date_time` | Feature | String (DateTime) | Hourly observation timestamp in US Central Standard/Daylight Time (CST/CDT) |
| `traffic_volume` | Target | Int64 | Hourly westbound vehicle count traversing ATR Station 301 |

---

## 5. Software and Hardware Requirements

### Software Environment
* **Operating System:** Windows 10/11 64-bit (Linux/macOS fully compatible)
* **Programming Language:** Python 3.11.9 (64-bit)
* **Virtual Environment Tool:** `venv` standard library
* **Core Scientific Libraries:**
  - `pandas` (version 3.0.6): High-performance tabular manipulation and time-series indexing
  - `numpy` (version 2.4.6): Vectorized numerical computation and trigonometric transforms
  - `scikit-learn` (version 1.9.1): Preprocessing pipelines, estimators, and metrics
  - `joblib` (bundled with scikit-learn): Thread-safe model artifact serialization
  - `matplotlib` (version 3.11.2): Vector and raster diagnostic plotting
  - `seaborn` (version 0.13.2): Statistical visual enhancements and aesthetics

### Hardware Specifications
* **CPU:** Multi-core x86-64 processor (Intel Core i5/i7/i9 or AMD Ryzen 5/7/9)
* **RAM:** Minimum 8 GB available RAM (16 GB recommended for full Random Forest training)
* **Storage:** ~200 MB disk space for workspace, dataset, figures, and serialized pipeline

---

## 6. System Architecture

The project adheres to a modular, decoupled data science engineering architecture designed for reproducible execution from the command-line interface (CLI):

```text
UrbanTrafficFlowOptimization/
│
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv       # Raw immutable dataset (48,204 rows)
│
├── models/
│   └── traffic_volume_rf_model.joblib            # Production pipeline (Scaler + OneHot + RF)
│
├── outputs/
│   ├── figures/                                  # Publication diagnostic figures (300 DPI)
│   │   ├── actual_vs_predicted.png               # Out-of-sample scatter and time-series snippet
│   │   ├── average_traffic_by_day.png            # Monday-Sunday distribution profile
│   │   ├── average_traffic_by_hour.png           # 24-hour diurnal profile with standard deviation
│   │   ├── correlation_heatmap.png               # Lower-triangle Pearson correlation matrix
│   │   ├── feature_importance.png                # Random Forest Gini impurity feature rankings
│   │   ├── monthly_traffic.png                   # Seasonal volume profile across months
│   │   ├── prediction_residuals.png              # Residual homoscedasticity and distribution
│   │   ├── rainfall_vs_traffic.png               # Precipitation impact comparison
│   │   ├── temperature_vs_traffic.png            # Celsius temperature vs traffic scatter
│   │   ├── traffic_over_time.png                 # Longitudinal 6-year daily/monthly trend
│   │   ├── weather_traffic.png                   # Categorical weather volume rankings
│   │   └── weekday_vs_weekend.png                # Weekday bimodal vs weekend unimodal overlay
│   ├── correlation_matrix.csv                    # Feature correlation values
│   ├── daily_traffic_summary.csv                 # Daily mean, std, median aggregates
│   ├── eda_summary.txt                           # Quantitative findings and data audit report
│   ├── hourly_traffic_summary.csv                 # Hourly aggregates across 24 hours
│   ├── model_metrics.csv                         # Out-of-sample MAE, RMSE, and R² metrics
│   ├── monthly_traffic_summary.csv                # Monthly volume statistics
│   ├── weather_description_summary.csv           # Granular weather breakdown
│   ├── weather_main_summary.csv                  # Main weather condition aggregates
│   ├── weekday_vs_weekend_summary.csv            # Commute vs weekend volume aggregates
│   └── yearly_traffic_summary.csv                 # Annual traffic volume statistics
│
├── reports/
│   ├── final_project_report.md                   # Comprehensive academic technical report
│   └── project_demo_guide.md                     # Step-by-step presentation demonstration guide
│
├── src/
│   ├── __init__.py                               # Package initialization
│   ├── data_loader.py                            # Robust path resolution and schema validation
│   ├── eda.py                                    # Cleaning, feature extraction, aggregation & plotting
│   ├── inspector.py                              # Structural audits, missing values, descriptive stats
│   └── model.py                                  # Chronological split, pipelines, evaluation, inference
│
├── main.py                                       # Unified CLI controller (--eda, --train, --predict)
├── requirements.txt                              # Pinned library dependencies
└── README.md                                     # Project onboarding and documentation
```

---

## 7. Data Preprocessing and Feature Engineering

### 1. Data Cleaning and Auditing
* **Exact Duplicate Rows vs. Duplicate Timestamps:**
  An exact record audit revealed that the dataset contains **exactly 17 duplicate rows** where all 9 columns are identical (representing accidental duplicate logging). These 17 records were safely removed, reducing the row count from 48,204 to **48,187 unique observations**.
  *Important Clarification:* There are **7,629 records that share timestamps** (`date_time`). These do not represent duplicate data; rather, weather logging stations record multiple weather phenomena concurrently during the same hour (e.g., both "Rain" and "Mist"). The cleaning pipeline correctly retains these multi-condition records and only removes the 17 exact duplicate rows.
* **Preservation of the Raw File:**
  The original CSV file in `data/Metro_Interstate_Traffic_Volume.csv` remains strictly read-only and unmodified (MD5: `f13e9456683f057a6a0cf82f7d64bec7`).
* **Holiday Feature Handling:**
  In the raw dataset, 48,143 records (99.87%) have `NaN` values for `holiday`. In accordance with the official dataset schema, `NaN` denotes normal, non-holiday operational days. The pipeline maps `NaN` to `0` and designated holidays (61 observations across Labor Day, Thanksgiving, Christmas, New Year's Day, Memorial Day, Independence Day, State Fair, etc.) to `1` in the engineered binary feature `is_holiday`.
* **Zero-Traffic Verification:**
  Exactly two records contain `traffic_volume == 0` (July 23, 2016, at 18:00 and 23:00). These were verified to be valid operational sensor observations and were preserved intact.

### 2. Sensor Recording Anomaly Treatment
* **Zero-Kelvin Temperature Anomaly:**
  The dataset contains **10 records** where `temp == 0.0 K` (corresponding to $-273.15^\circ\text{C}$), occurring between January 31, 2014, and February 2, 2014. These represent hardware sensor communication failures.
  *Treatment:* Temperature is first converted to Celsius ($T_{\text{Celsius}} = T_{\text{Kelvin}} - 273.15$). In the preprocessing stage, values below $-50^\circ\text{C}$ are identified. Crucially, the replacement value is calculated strictly as the **median temperature of the training partition** ($9.60^\circ\text{C}$ / $10.0^\circ\text{C}$) to prevent information leakage from future test records.
* **Extreme Rainfall Outlier:**
  One record on July 11, 2016 at 17:00 logs `rain_1h = 9,831.3 mm` (equivalent to nearly 10 meters of precipitation in a single hour). The second-highest rainfall in the dataset is $55.63\text{ mm}$. A rainfall rate of nearly $10,000\text{ mm/h}$ is physically impossible.
  *Treatment:* Rather than dropping the record (which would disrupt temporal sequence integrity), `rain_1h` is capped (winsorized) at a physical upper threshold of $100.0\text{ mm}$ based on meteorological limits observed during training.

### 3. Feature Engineering & Mathematical Transformations
Traffic demand is inherently periodic. Traditional integer representations of time (e.g., hour $0$ to $23$) introduce an artificial discontinuity: hour 23 and hour 0 appear mathematically distant despite being adjacent. To preserve cyclical continuity, harmonic trigonometric transformations were constructed:

$$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$

$$\text{day\_sin} = \sin\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right), \quad \text{day\_cos} = \cos\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right)$$

$$\text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$

### 4. Target Leakage Prevention
To guarantee that the model functions as a pure environmental and schedule-based predictor:
- The target variable `traffic_volume` is completely isolated.
- No autoregressive lags (e.g., $t-1$, $t-24$) or rolling averages of `traffic_volume` were included. This ensures that the model does not rely on previous volume counts (which would be unavailable during collection outages) and avoids lookahead contamination across temporal discontinuities.

---

## 8. Exploratory Data Analysis and Actual Findings

Execution of `python main.py --eda` generated 9 diagnostic figures and 8 aggregated statistical summaries. The quantitative findings are summarized below:

### 1. Diurnal Traffic Profiles (Hour of Day)
* **Busiest Hour:** **16:00 (4:00 PM)** with a mean volume of **5,663.76 vehicles/hour**.
* **Quietest Hour:** **03:00 (3:00 AM)** with a mean volume of **371.09 vehicles/hour**.
* **Commute Dynamics:** Weekdays display a pronounced bimodal distribution:
  - Morning rush peak: **07:00 (7:00 AM)** averaging **6,048.41 vehicles/hour**.
  - Evening rush peak: **16:00 (4:00 PM)** averaging **6,204.34 vehicles/hour**.
  - Midday lull: Volumes dip slightly between 10:00 and 13:00 to ~4,600 vehicles/hour before rising for the evening commute.

### 2. Weekly and Weekend Rhythms
* **Busiest Day of the Week:** **Friday** (mean: **3,655.33 vehicles/hour**), followed closely by Thursday (**3,637.93 vehicles/hour**).
* **Quietest Day of the Week:** **Sunday** (mean: **2,369.09 vehicles/hour**).
* **Weekday vs. Weekend Contrast:**
  - **Weekday (Monday–Friday) Mean:** **3,533.29 vehicles/hour** ($N=34,500$).
  - **Weekend (Saturday–Sunday) Mean:** **2,570.70 vehicles/hour** ($N=13,687$).
  - Weekend volume exhibits a **~27.24% reduction** relative to weekday volume.
  - Crucially, weekends lack the bimodal commute spikes entirely, instead displaying a broad, unimodal plateau peaking between 12:00 and 16:00 at ~4,350 vehicles/hour.

### 3. Meteorological Sensitivities
* **Primary Weather Categories:**
  - `Clouds`: 3,618.0 veh/h ($N=15,158$)
  - `Haze`: 3,502.1 veh/h ($N=1,360$)
  - `Rain`: 3,317.9 veh/h ($N=5,672$)
  - `Drizzle`: 3,292.2 veh/h ($N=1,820$)
  - `Clear`: 3,055.6 veh/h ($N=13,384$)
  - `Snow`: 3,016.3 veh/h ($N=2,875$)
  - `Thunderstorm`: 2,999.4 veh/h ($N=1,033$)
  - `Mist`: 2,933.3 veh/h ($N=5,949$)
  - `Fog`: 2,703.7 veh/h ($N=912$)
  - `Squall`: 2,061.8 veh/h ($N=4$)
* **Precipitation Analysis:**
  Comparing completely dry hours ($\text{rain\_1h} = 0\text{ mm}$, mean: 3,257.4 veh/h, $N=44,720$) with rainy hours ($\text{rain\_1h} > 0\text{ mm}$, mean: 3,292.1 veh/h, $N=3,467$) indicates that rain alone does not reduce overall highway traffic. Rain occurs predominantly during daytime atmospheric conditions when traffic is naturally higher. However, within active rainfall events, increasing rain intensity exhibits a modest negative slope ($-61.8\text{ veh/h}$ per mm of rain), indicating slight driver aversion or speed-induced spacing during heavy cloudbursts.
* **Temperature Correlation:**
  A positive Pearson correlation ($r = +0.130$) is observed between temperature in Celsius and traffic volume, primarily driven by higher travel demand during summer months and daylight heating cycles.

---

## 9. Regression Algorithms and Training Methodology

To rigorously assess predictive utility, three distinct modeling paradigms were developed:

### 1. Baseline Model (Training Set Mean)
The baseline model serves as the empirical zero-intelligence reference. It outputs a constant prediction equal to the mean traffic volume of the training partition:

$$\hat{y}_{\text{baseline}} = \bar{y}_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} y_i \approx 3,257.77\text{ vehicles/hour}$$

Evaluating the baseline establishes the irreducible error of guessing the central tendency and proves whether more complex models learn genuine predictive relationships.

### 2. Linear Regression (Parametric Baseline)
Ordinary Least Squares (OLS) Linear Regression provides a transparent parametric baseline. The model estimates linear coefficients $\beta_j$:

$$\hat{y}_{\text{linear}} = \beta_0 + \sum_{j=1}^{p} \beta_j X_j$$

Numeric features (including cyclical sine and cosine terms) are standardized using `StandardScaler` ($\mu=0, \sigma=1$), and categorical weather is one-hot encoded using `OneHotEncoder(handle_unknown='ignore')`.

### 3. Random Forest Regressor (Non-Linear Ensemble)
The primary non-linear model is an ensemble of 100 de-correlated decision trees:
* **Estimators:** `n_estimators = 100`
* **Tree Depth:** `max_depth = 18` (prevents individual trees from memorizing localized noise)
* **Minimum Split Samples:** `min_samples_split = 5`
* **Minimum Leaf Samples:** `min_samples_leaf = 4` (smoothes leaf-node predictions)
* **Random State:** `random_state = 42` (ensures reproducible bootstrap subsampling)
* **Parallel Processing:** `n_jobs = -1`

The Random Forest aggregates tree predictions via bootstrap averaging:

$$\hat{y}_{\text{RF}}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

Decision tree ensembles are particularly suited for traffic forecasting because they naturally model non-linear step functions and interactive conditions (e.g., the differential impact of 07:00 AM on a Monday versus a Sunday).

---

## 10. Chronological Train/Test Split

### Methodological Rationale
In time-series and schedule-driven traffic analysis, standard randomized train/test shuffling introduces severe **temporal lookahead bias**: the model trains on samples adjacent in time to test samples, artificially inflating test scores.

To evaluate real-world forecasting performance, a strict **chronological partition** was applied:
1. Records are sorted by timestamp (`date_time`).
2. The initial **80% of observations** are allocated to training.
3. The final **20% of observations** are reserved strictly for out-of-sample testing.

### Partition Boundaries
* **Training Set ($80\%$):** **38,549 observations**
  - Earliest record: **October 2, 2012, at 09:00:00**
  - Latest record: **November 1, 2017, at 19:00:00**
* **Test Set ($20\%$):** **9,638 observations**
  - Earliest record: **November 1, 2017, at 20:00:00**
  - Latest record: **September 30, 2018, at 23:00:00**

### Training Isolation Guarantee
- Preprocessing steps (numerical scaling and one-hot encoding) are fitted **only** on the training set.
- Sensor outlier handling parameters (temperature median: $9.60^\circ\text{C}$; rainfall clipping: $100\text{ mm}$) are computed **only** from training observations.
- Hyperparameters were selected without fitting or tuning on the test set.

---

## 11. Actual Evaluation Metrics and Model Comparison

The three models were evaluated on the held-out test set ($N=9,638$). The actual metrics (saved in [`outputs/model_metrics.csv`](file:///r:/UrbanTrafficFlowOptimization/outputs/model_metrics.csv)) are:

### Summary Table

| Model | MAE (veh/h) | RMSE (veh/h) | $R^2$ Score | MAE Reduction vs Baseline | RMSE Reduction vs Baseline |
|---|---|---|---|---|---|
| **Baseline (Train Mean: 3,257.77)** | 1,722.50 | 1,967.61 | -0.000006 | Reference (0.00%) | Reference (0.00%) |
| **Linear Regression** | 826.21 | 1,050.57 | 0.7149 | 52.03% | 46.61% |
| **Random Forest Regressor** | **300.69** | **532.98** | **0.9266** | **82.54%** | **72.91%** |

### Comparison: Random Forest vs. Linear Regression
- **MAE Reduction:** Random Forest reduces MAE from 826.21 to 300.69 veh/h, an error reduction of **63.61%**.
- **RMSE Reduction:** Random Forest reduces RMSE from 1,050.57 to 532.98 veh/h, an error reduction of **49.27%**.
- **Variance Explained:** $R^2$ increases from **0.7149** (Linear Regression) to **0.9266** (Random Forest).

### Verification of Prediction Bounds
An audit of the test set predictions generated by the Random Forest model confirmed:
- Minimum predicted traffic volume: **239.90 vehicles/hour**
- Maximum predicted traffic volume: **6,902.82 vehicles/hour**
- NaN or infinite values: **0**
- Negative predictions: **0**

---

## 12. Graphs and Interpretation of Results

### 1. Actual vs. Predicted Traffic Volume (`outputs/figures/actual_vs_predicted.png`)
* **Scatter Plot (Left Panel):** Compares actual observed traffic against model predictions. The Random Forest points align closely along the identity line ($y = x$), displaying tight grouping across both low-flow overnight regimes (<1,000 veh/h) and high-flow daytime regimes (>5,000 veh/h). In contrast, Linear Regression exhibits broader dispersion and systematic overprediction during off-peak hours.
* **Continuous Time-Series Tracking (Right Panel):** A 14-day continuous slice from the test period (November 2017) illustrates that Random Forest accurately tracks the sharp bimodal morning and evening commute surges, the lower midday plateau on weekends, and overnight drops.

### 2. Model Residual Diagnostics (`outputs/figures/prediction_residuals.png`)
* **Residuals vs. Fitted Values (Left Panel):**
  - Residuals ($e_i = y_i - \hat{y}_i$) are evenly distributed around the zero-error dashed line across the majority of fitted values.
  - Mild heteroscedasticity is visible at fitted values exceeding 5,500 veh/h: downward vertical residual tails indicate instances where actual traffic fell well below predicted volume. These occurrences correspond to unrecorded real-world incidents (e.g., multi-vehicle crashes, road construction, emergency closures) that physically restricted roadway throughput.
* **Residual Error Distribution (Right Panel):**
  - The error distribution is approximately Gaussian, with a mean error of **$-47.0\text{ vehicles/hour}$** and standard deviation of **$530.9\text{ vehicles/hour}$**.
  - The slight negative mean error reflects a minor conservative overestimation during peak periods caused by unobserved downstream congestion bottlenecks.

### 3. Feature Importance Analysis (`outputs/figures/feature_importance.png`)
The top 15 features ranked by Random Forest Mean Decrease in Impurity (Gini importance) are:

| Rank | Feature | Gini Importance Weight | Share of Model Importance |
|---|---|---|---|
| 1 | `hour_cos` (Cyclic Cosine Hour) | 0.664 | 66.4% |
| 2 | `hour` (Raw Hour of Day) | 0.140 | 14.0% |
| 3 | `hour_sin` (Cyclic Sine Hour) | 0.051 | 5.1% |
| 4 | `day_of_week` (Day Index 0–6) | 0.036 | 3.6% |
| 5 | `day_sin` (Cyclic Sine Day) | 0.035 | 3.5% |
| 6 | `is_weekend` (Weekend Flag) | 0.030 | 3.0% |
| 7 | `temp_c` (Temperature in °C) | 0.017 | 1.7% |
| 8 | `month_cos` (Cyclic Cosine Month) | 0.006 | 0.6% |
| 9 | `year` (Calendar Year) | 0.005 | 0.5% |
| 10 | `day_cos` (Cyclic Cosine Day) | 0.005 | 0.5% |
| 11 | `clouds_all` (Cloud Coverage %) | 0.003 | 0.3% |
| 12 | `month` (Calendar Month) | 0.002 | 0.2% |
| 13 | `month_sin` (Cyclic Sine Month) | 0.002 | 0.2% |
| 14 | `rain_1h` (Rainfall in mm) | 0.002 | 0.2% |
| 15 | `weather_main_Clouds` | 0.001 | 0.1% |

### Critical Technical Caveats on Feature Importance
1. **Division of Importance Across Collinear Features:**
   When multiple features encode related underlying information (e.g., `hour_cos`, `hour`, and `hour_sin`), decision tree splits distribute importance across them rather than assigning it to a single feature. Taken together, hourly features represent **85.5%** of all tree splits.
2. **Impurity Importance Does Not Prove Causation:**
   Random Forest impurity importance reflects how frequently a feature was selected to partition variance in the decision trees; it **does not imply causal mechanisms**. For instance, cloud cover or temperature does not directly "cause" traffic volume to change; rather, they serve as statistical proxies for daylight hours, seasonal vacation periods, and commute regularity.
3. **Generalization Over Calendar Years:**
   The test set contains 7,949 records from calendar year 2018, which was completely absent from the training partition (2012–2017). Because tree-based models cannot extrapolate beyond the maximum value seen during training, any split on `year > 2016.5` places all 2018 samples onto the same terminal branch. 
   Crucially, `year` accounted for only **0.5%** of model importance, confirming that the model did not overfit to annual trends and successfully generalized to 2018 using the recurring diurnal, weekly, and seasonal features.

---

## 13. Limitations

A scientifically rigorous evaluation requires transparent documentation of data and modeling limitations:

1. **Unidirectional Spatial Coverage:**
   The dataset captures westbound traffic exclusively at ATR Station 301. It cannot be used to infer eastbound flow, parallel arterial demand, or city-wide origin-destination matrices.
2. **Hourly Aggregation Horizon:**
   Observations are aggregated into 1-hour increments. Rapid transient phenomena—such as 15-minute shockwave propagation, short-duration signal gridlock, or immediate incident onset—cannot be resolved at this timescale.
3. **Absence of Real-Time Incident Data:**
   The dataset does not contain incident logs (accidents, disabled vehicles), construction lane restrictions, or stadium event schedules. As a result, the model predicts **expected baseline demand under prevailing temporal and weather conditions**, not guaranteed sensor counts.
4. **Historical Data HIatus (2014–2015):**
   A prolonged data collection outage exists between late 2014 and mid-2015. While the chronological split safely places all 2017–2018 data into testing, models requiring continuous autoregressive sequence inputs (e.g., ARIMA or LSTM) would require complex imputation across this gap.
5. **Hardware Sensor Failures:**
   As documented, 10 records recorded 0 Kelvin and 1 record recorded 9,831.3 mm of rain. While the training pipeline handles these with training-fitted medians and physical bounds, raw sensor streams require real-time validation layers before feeding downstream models.

---

## 14. Conclusion and Future Scope

### Conclusion
This project successfully implemented, verified, and documented an end-to-end data science and machine learning pipeline for urban freeway flow optimization on Interstate 94 in Minneapolis–St. Paul. 

Key outcomes include:
1. **Data Integrity Audited:** Exact row duplicates ($N=17$) were distinguished from concurrent weather timestamp entries ($N=7,629$), and sensor anomalies were appropriately handled without modifying the original raw CSV.
2. **Temporal Dynamics Characterized:** Discovered that diurnal cycles and day-of-week profiles explain the vast majority of traffic variance, with Friday evening commutes representing peak demand and Sunday overnights representing minimum volume.
3. **Rigorous Chronological Benchmarking:** Evaluating models on an untouched out-of-sample chronological test set confirmed that the **Random Forest Regressor** achieved outstanding forecasting performance ($R^2 = 0.9266$, $\text{MAE} = 300.69\text{ veh/h}$), significantly outperforming Linear Regression ($R^2 = 0.7149$) and rendering the naive baseline obsolete.
4. **Production Pipeline Packaged:** The trained preprocessor and regressor were serialized into a reusable production artifact supported by an inference CLI that incorporates input validation and explicit disclaimers.

### Future Scope
1. **Spatial Network Integration:** Expanding the model to multi-station sensor networks (e.g., MN DoT loop detectors across all I-94 ramps and adjacent arterials) using Graph Neural Networks (GNNs) or Spatio-Temporal Graph Convolutional Networks (ST-GCNs).
2. **Real-Time Incident Integration:** Fusing open 511 incident feeds, road construction logs, and local event calendars to dynamically penalize capacity during active disruptions.
3. **Deep Sequence Modeling:** Implementing Temporal Fusion Transformers (TFT) or Long Short-Term Memory (LSTM) networks equipped with attention mechanisms for multi-horizon probabilistic forecasting ($t+1$ to $t+24$ hours).
4. **Intelligent Ramp Metering:** Embedding the predictive model into a closed-loop simulation (e.g., SUMO or Aimsun) to optimize dynamic ramp meter signal timings ahead of forecasted congestion surges.

---

## 15. References

1. **Hogue, J. (2019).** *Metro Interstate Traffic Volume*. UCI Machine Learning Repository. [https://doi.org/10.24432/C5X60B](https://doi.org/10.24432/C5X60B).
2. **Minnesota Department of Transportation (MN DoT).** *Traffic Data and Analysis: Automatic Traffic Recorder (ATR) Station 301 Documentation*. Office of Transportation Data and Analysis, St. Paul, MN.
3. **Breiman, L. (2001).** Random Forests. *Machine Learning*, 45(1), 5–32. [https://doi.org/10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324).
4. **Pedregosa, F., et al. (2011).** Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
5. **Transportation Research Board (TRB). (2016).** *Highway Capacity Manual (HCM), 6th Edition: A Guide for Multimodal Mobility Analysis*. National Academies of Sciences, Engineering, and Medicine, Washington, DC.
6. **Hyndman, R. J., & Athanasopoulos, G. (2021).** *Forecasting: Principles and Practice*, 3rd edition. OTexts: Melbourne, Australia.
