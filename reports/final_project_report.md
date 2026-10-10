# Urban Traffic Flow Optimization: Final Project Report

---

## 1. Title of the Project
**Urban Traffic Flow Optimization: An Empirical Investigation of Temporal Dynamics, Weather Factors, and Machine Learning Regression for Traffic Volume Prediction**

* **Academic Domain:** Data Science, Machine Learning & Intelligent Transportation Systems (ITS)
* **Dataset:** Metro Interstate Traffic Volume (Westbound Interstate 94, Minneapolis–St. Paul, MN)
* **Repository:** `UrbanTrafficFlowOptimization`
* **Artifacts:** PyTorch Deep Learning Model (`models/traffic_model.pth`), Interactive Streamlit Application (`app.py`), Complete Jupyter Notebook (`notebooks/Urban_Traffic_Flow_Optimization.ipynb`)

---

## 2. Problem Statement
Increasing traffic congestion is a major challenge in modern metropolitan areas. Unplanned traffic delays result in billions of lost productive hours, elevated fuel consumption, heightened carbon emissions, and increased collision risks. 

Freeway traffic volume is governed by complex, multi-factor non-linear dynamics:
- **Diurnal commute schedules:** Morning and evening rush hours driven by workforce migration.
- **Weekly rhythms:** Distinct travel profiles between weekdays and weekends.
- **Calendar & Holiday impacts:** Reductions in business traffic and shifts toward recreational travel.
- **Meteorological disturbances:** Precipitation (rain/snow), ambient temperature extremes (freezing conditions), and visibility constraints.

Traditional municipal traffic management often relies on static historical averages or simple linear models that fail to capture sudden transitions between free-flow and congested regimes. The challenge is:

> *Can we build an accurate, robust, and interpretable machine learning system that analyzes historical traffic data, identifies congestion factors, and forecasts hourly traffic volume using temporal, environmental, and weather factors without lookahead bias or future leakage?*

---

## 3. Objective
The primary objectives of this project are:
1. **Analyze Historical Traffic Data:** Ingest and audit 48,204 hourly sensor observations from Interstate 94 to identify underlying demand dynamics and anomalies.
2. **Identify Peak Traffic Periods:** Quantify diurnal commuter rush hours, weekend plateaus, and seasonal traffic trends.
3. **Study Weather and Environmental Effects:** Quantify the statistical influence of temperature, precipitation (rain and snow), cloud cover, and general weather conditions on vehicle throughput.
4. **Perform Feature Extraction and Engineering:** Construct cyclical trigonometric transformations ($\sin/\cos$) for temporal components and encode categorical weather attributes without target leakage.
5. **Develop and Compare Regression Models:** Train, evaluate, and benchmark multiple machine learning architectures:
   - Baseline Mean Model (naive statistical benchmark).
   - Ordinary Least Squares Linear Regression.
   - Deep Multi-Layer Perceptron (MLP) Neural Network built in PyTorch.
   - Random Forest Regressor (ensemble non-linear benchmark).
6. **Deliver an Interactive Dashboard:** Build and deploy a modern, responsive web application allowing operators and users to run real-time traffic volume simulations and explore EDA findings.

---

## 4. Methodology
The project follows a rigorous end-to-end data science lifecycle:

```
+-----------------------------------------------------------------------+
| 1. DATA INGESTION & AUDITING                                          |
|    - Load 48,204 records; remove 17 exact duplicate rows               |
|    - Handle hardware sensor anomalies (0 K temp, extreme rain)        |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| 2. EXPLORATORY DATA ANALYSIS (EDA)                                     |
|    - Generate 12 analytical plots (diurnal, weekly, weather, heatmaps) |
|    - Statistical correlation and distribution analysis                |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| 3. FEATURE EXTRACTION & CYCLICAL ENCODING                             |
|    - Time decomposition: hour, day, month, day-of-week, year          |
|    - Cyclical encoding: sin/cos harmonic transforms for 24h & 12m     |
|    - Binary indicators: is_weekend, is_rush_hour, is_holiday          |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| 4. CHRONOLOGICAL TRAIN/TEST SPLIT                                      |
|    - Strict 80/20 chronological partition (no shuffling)              |
|    - Train: 38,549 records (2012-2017) | Test: 9,638 records (2017-18)|
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| 5. PREPROCESSING & SCALING PIPELINE                                    |
|    - StandardScaler fitted strictly on training continuous features    |
|    - OneHotEncoder for weather categories (weather_main)               |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| 6. MODEL TRAINING & BENCHMARKING                                      |
|    - Baseline Mean | Linear Regression | PyTorch NN | Random Forest   |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| 7. EVALUATION & INFERENCE DEPLOYMENT                                  |
|    - Metrics: MAE, RMSE, R² on held-out out-of-sample test set        |
|    - Save weights: models/traffic_model.pth                            |
|    - Web Interface: Streamlit analytics & prediction simulator       |
+-----------------------------------------------------------------------+
```

### Key Methodological Safeguards:
* **No Lookahead Bias:** Data was partitioned chronologically (the oldest 80% used for training, the most recent 20% reserved for testing) rather than random shuffling, reflecting real-world operational forecasting.
* **Leakage-Free Preprocessing:** All scalers and transformers were fitted solely on the training partition and transformed onto the test partition.
* **Harmonic Time Representation:** Hours of the day and months of the year were mapped onto unit circles using $\sin\left(\frac{2\pi t}{T}\right)$ and $\cos\left(\frac{2\pi t}{T}\right)$ to ensure seamless continuity between 23:00 and 00:00.

---

## 5. Dataset, Tools & Technologies Used

### 5.1 Dataset Description
* **Dataset:** Metro Interstate Traffic Volume Dataset
* **Source:** Minnesota Department of Transportation (MN DoT) / UCI Machine Learning Repository
* **Station Location:** ATR Station 301, Westbound Interstate 94, Minneapolis–St. Paul, Minnesota
* **Observation Period:** October 2, 2012 to September 30, 2018 (6 years of continuous hourly records)
* **Dataset Dimensions:** 48,204 rows, 9 original attributes

| Attribute | Data Type | Description |
|---|---|---|
| `holiday` | Categorical / String | US National holidays and State Fair; empty for non-holidays |
| `temp` | Float64 | Hourly average ambient temperature in Kelvin |
| `rain_1h` | Float64 | Hourly liquid precipitation amount in millimeters (mm) |
| `snow_1h` | Float64 | Hourly snowfall accumulation in millimeters (mm) |
| `clouds_all` | Int64 | Cloud coverage percentage ($0\% - 100\%$) |
| `weather_main` | Categorical | Macro weather category (Clear, Clouds, Rain, Snow, Mist, etc.) |
| `weather_description` | Categorical | Detailed descriptive weather phenomenon |
| `date_time` | DateTime | Hourly timestamp (CST/CDT) |
| `traffic_volume` | Int64 (Target) | Hourly westbound vehicle count traversing Station 301 |

### 5.2 Tools and Technologies
* **Programming Language:** Python 3.11
* **Deep Learning Framework:** PyTorch (`torch`, `torch.nn`, `torch.optim`)
* **Machine Learning & Preprocessing:** Scikit-Learn (`StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `RandomForestRegressor`, `LinearRegression`)
* **Data Manipulation & Mathematics:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Interactive Web Application:** Streamlit
* **Development & Version Control:** VS Code, Jupyter Notebook, Git, GitHub

---

## 6. Implementation

### 6.1 Modular Architecture
The codebase was organized into modular, maintainable components:
- `data/`: Houses the official historical dataset `Metro_Interstate_Traffic_Volume.csv`.
- `src/data_loader.py`: Handles data ingestion, schema validation, and date parsing.
- `src/eda.py`: Performs statistical analysis and generates 12 high-resolution figures.
- `src/model.py`: Implements feature engineering, model training, evaluation, and PyTorch serialization.
- `notebooks/Urban_Traffic_Flow_Optimization.ipynb`: Complete self-contained notebook covering end-to-end data cleaning, EDA, PyTorch NN training, Random Forest benchmarking, and evaluation.
- `app.py`: Full-featured operational dashboard with live interactive volume prediction.
- `main.py`: Command-line interface for running pipelines (`--eda`, `--train`, `--predict`, `--inspect`).

### 6.2 PyTorch Deep Neural Network Architecture (`TrafficVolumeNN`)
To capture non-linear interactions across time and weather variables, a custom Multi-Layer Perceptron (MLP) was implemented in PyTorch:

```python
class TrafficVolumeNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
```

* **Optimizer:** Adam with learning rate $\alpha = 0.003$ and weight decay $10^{-5}$ for L2 regularization.
* **Loss Function:** Mean Squared Error (MSELoss).
* **Batch Size & Epochs:** Mini-batch size of 128 trained over 25 epochs with batch normalization and dropout layers to prevent overfitting.
* **Model Packaging:** The trained PyTorch model weights, fitted `ColumnTransformer` preprocessor, feature names, and benchmark metrics are serialized directly into `models/traffic_model.pth` (~72 KB).

### 6.3 User Interface and Prediction Simulator
The Streamlit application (`app.py`) provides four dedicated modules:
1. **Overview Dashboard:** Live KPI cards (total observations, baseline flow, peak flow, test $R^2$), historical trend charts, and key statistical findings.
2. **Traffic Analysis:** Interactive visual gallery categorizing all 12 EDA figures across diurnal, weather, and calendar perspectives.
3. **Traffic Prediction Simulator:** Real-time form allowing operators to select date, time, weather conditions, temperature, and precipitation, immediately outputting predicted traffic volume, capacity status, and commute advice.
4. **Model Performance:** Side-by-side benchmark comparison table, residual plots, actual vs. predicted curves, and feature importance analysis.

---

## 7. Results

### 7.1 Quantitative Model Benchmarking
All models were evaluated on the held-out out-of-sample test partition (9,638 hourly records):

| Model | MAE (veh/h) | RMSE (veh/h) | $R^2$ Score | Error Reduction vs. Baseline |
|---|---|---|---|---|
| **Baseline (Train Mean)** | 1,722.50 | 1,967.61 | -0.0000 | 0.0% |
| **Linear Regression** | 826.47 | 1,050.47 | 0.7150 | 52.02% |
| **PyTorch Neural Network (MLP)** | **341.56** | **580.03** | **0.9131** | **80.17%** |
| **Random Forest Regressor** | **300.78** | **533.24** | **0.9266** | **82.54%** |

### 7.2 Key Exploratory Findings
1. **Diurnal Rush-Hour Patterns:**
   - **Busiest Single Hour:** 16:00 (4:00 PM) averaging **5,664 vehicles/hour**, driven by the evening work commute.
   - **Morning Rush Period:** 07:00 (7:00 AM) averaging **4,700 vehicles/hour**.
   - **Overnight Trough:** 02:00 to 03:00 AM averaging under **500 vehicles/hour**.
2. **Weekday vs. Weekend Demand Divergence:**
   - Weekdays average **3,534 vehicles/hour** compared to **2,571 vehicles/hour** on weekends (**+37.5% weekday commuter premium**).
   - Weekdays exhibit a distinct bimodal peak (morning and evening), whereas weekends follow a unimodal, smooth plateau from midday through late afternoon.
3. **Weather Sensitivities:**
   - Extreme freezing temperatures ($< 250\text{ K} / -23^\circ\text{C}$) and heavy precipitation cause a 10% to 20% reduction in discretionary weekend travel.
   - Inclement weather on weekdays shifts commuter departure times rather than eliminating trips, exacerbating congestion windows.
4. **Feature Importance Ranking:**
   - Time-of-day features (`hour`, `hour_sin`, `hour_cos`, `is_rush_hour`) account for over **85%** of predictive power.
   - Calendar features (`dayofweek`, `is_weekend`, `month`) account for approximately **10%**.
   - Environmental weather features (`temp`, `rain_1h`, `weather_main`) account for approximately **5%** as modulating factors.

---

## 8. Conclusion
The Urban Traffic Flow Optimization project successfully developed a complete, robust, and validated traffic analysis and prediction pipeline for metropolitan interstate corridors.

### Key Takeaways:
1. **High Predictive Performance:** Both the non-linear Random Forest ($R^2 = 0.9266$) and the PyTorch Deep Neural Network ($R^2 = 0.9131$) achieved outstanding accuracy, outperforming classical linear models by over 20 percentage points in $R^2$ and cutting prediction error in half.
2. **Harmonic Time Encoding is Essential:** Representing time variables cyclically via trigonometric transforms eliminated boundary discontinuities and significantly stabilized gradient-based optimization in PyTorch.
3. **Practical Deployment:** The lightweight serialized model package (`models/traffic_model.pth`) enables instant sub-millisecond inference in the Streamlit application, proving that deep learning models can be effectively deployed in interactive operational dashboards.

### Future Scope:
- **Spatial Expansion:** Incorporating sensor feeds from adjacent highway stations to create Spatial-Temporal Graph Neural Networks (ST-GNNs).
- **Real-Time Streaming:** Connecting live IoT traffic sensors via Kafka or MQTT for real-time continuous model retraining and adaptive signal control.
- **Incident Detection Integration:** Incorporating accident logs, lane closure alerts, and construction schedules for automated congestion anomaly warnings.
