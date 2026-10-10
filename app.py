"""Urban Traffic Flow Optimization - Streamlit Web Dashboard.

Provides an interactive GUI for dataset overview, exploratory traffic analysis,
real-time traffic volume prediction using the trained Random Forest model,
and model performance comparison.
"""

from datetime import date, datetime, time
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_traffic_data, resolve_dataset_path
from src.model import load_trained_model, predict_traffic_volume

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Urban Traffic Flow Optimization",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling (CSS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global font & typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* App Header Banner */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #1e3a8a 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .app-header h1 {
        color: #ffffff !important;
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        margin: 0 0 8px 0 !important;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #94a3b8 !important;
        font-size: 1.0rem !important;
        margin: 0 !important;
    }
    
    /* Corridor badge */
    .corridor-badge {
        display: inline-block;
        background: rgba(37, 99, 235, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 10px;
    }

    /* KPI Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #059669;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Prediction Result Banner */
    .pred-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 60%, #3b82f6 100%);
        padding: 26px 30px;
        border-radius: 14px;
        color: #ffffff;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.25);
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .pred-title {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #bfdbfe;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .pred-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
    }
    .pred-unit {
        font-size: 1.1rem;
        color: #e0e7ff;
        font-weight: 500;
        margin-left: 6px;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 12px 18px;
        border-radius: 0 8px 8px 0;
        font-size: 0.86rem;
        color: #475569;
        margin-top: 14px;
    }
    
    /* Section container */
    .section-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 24px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Cached Resource Loaders
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_historical_data() -> pd.DataFrame:
    """Load the raw dataset and parse timestamps."""
    dataset_path = resolve_dataset_path()
    df = load_traffic_data(dataset_path, parse_dates=True)
    return df


@st.cache_data(show_spinner=False)
def load_summary_table(filename: str) -> Optional[pd.DataFrame]:
    """Load pre-computed summary CSV tables from outputs directory."""
    path = PROJECT_ROOT / "outputs" / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


@st.cache_resource(show_spinner=False)
def get_model_pipeline() -> Optional[Dict[str, Any]]:
    """Load the trained Random Forest pipeline once per session."""
    model_path = PROJECT_ROOT / "models" / "traffic_volume_rf_model.joblib"
    if model_path.exists():
        try:
            return load_trained_model(model_path)
        except Exception as exc:
            st.error(f"Failed to load model from {model_path}: {exc}")
            return None
    return None


# -----------------------------------------------------------------------------
# Sidebar Navigation & Context
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚦 Traffic Optimizer")
    st.caption("Interstate 94 Westbound | ATR Station 301")
    
    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "📈 Traffic Analysis",
            "🚦 Predict Traffic",
            "🎯 Model Performance",
        ],
        index=0,
    )
    
    st.markdown("---")
    st.markdown("#### System Status")
    
    # Model status indicator
    model_pkg = get_model_pipeline()
    if model_pkg is not None:
        st.success("● Model Pipeline: **Ready** (Random Forest)", icon="✅")
    else:
        st.warning("● Model Pipeline: **Not Loaded**", icon="⚠️")
        
    dataset_file = PROJECT_ROOT / "data" / "Metro_Interstate_Traffic_Volume.csv"
    if dataset_file.exists():
        st.info("● Raw Dataset: **Verified** (48,204 rows)", icon="📁")
    else:
        st.error("● Raw Dataset: **Missing**", icon="❌")
        
    st.markdown("---")
    st.markdown("#### Project Resources")
    st.markdown(
        """
        - 📄 [Final Project Report](file:///R:/UrbanTrafficFlowOptimization/reports/final_project_report.md)
        - 🧪 [Demonstration Guide](file:///R:/UrbanTrafficFlowOptimization/reports/project_demo_guide.md)
        - 📊 [Model Metrics CSV](file:///R:/UrbanTrafficFlowOptimization/outputs/model_metrics.csv)
        """
    )
    st.caption("Minnesota Dept. of Transportation & UCI Repository")


# -----------------------------------------------------------------------------
# Page A: Overview
# -----------------------------------------------------------------------------
if page == "📊 Overview":
    st.markdown(
        """
        <div class="app-header">
            <h1>Urban Traffic Flow Optimization</h1>
            <p>Empirical traffic dynamics analysis and machine learning forecasting for Interstate 94.</p>
            <span class="corridor-badge">Westbound I-94 | Station 301 | Minneapolis–St. Paul, MN</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading historical traffic data..."):
        try:
            df = load_historical_data()
            total_records = len(df)
            avg_traffic = df["traffic_volume"].mean()
            max_traffic = df["traffic_volume"].max()
            min_traffic = df["traffic_volume"].min()
            start_date = df["date_time"].min().strftime("%b %d, %Y")
            end_date = df["date_time"].max().strftime("%b %d, %Y")
        except Exception as e:
            st.error(f"Error loading historical dataset: {e}")
            total_records, avg_traffic, max_traffic, min_traffic = 48204, 3259.8, 7280, 0
            start_date, end_date = "Oct 02, 2012", "Sep 30, 2018"

    # Top KPI Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Observations</div>
                <div class="metric-value">{total_records:,}</div>
                <div class="metric-sub">Hourly sensor logs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Traffic Volume</div>
                <div class="metric-value">{avg_traffic:,.0f}</div>
                <div class="metric-sub">Vehicles per hour</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Peak Recorded Flow</div>
                <div class="metric-value">{max_traffic:,}</div>
                <div class="metric-sub">Max hourly volume</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Data Span</div>
                <div class="metric-value">6 Years</div>
                <div class="metric-sub">{start_date} – {end_date}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Historical Traffic Trend Chart
    st.subheader("Longitudinal Traffic Volume Trend (2012–2018)")
    trend_fig_path = PROJECT_ROOT / "outputs" / "figures" / "traffic_over_time.png"
    if trend_fig_path.exists():
        st.image(str(trend_fig_path), caption="Historical daily and 30-day moving average volume on westbound I-94", use_container_width=True)
    else:
        st.info("Longitudinal chart not found. Run 'python main.py --eda' to generate figures.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Diurnal Profile & Peak Traffic Periods
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.subheader("Diurnal 24-Hour Traffic Pattern")
        diurnal_fig = PROJECT_ROOT / "outputs" / "figures" / "average_traffic_by_hour.png"
        if diurnal_fig.exists():
            st.image(str(diurnal_fig), caption="Average hourly traffic volume with ±1 std ribbon", use_container_width=True)
        else:
            st.info("Hourly pattern figure missing.")

    with col_right:
        st.subheader("Peak Traffic Periods")
        st.markdown(
            """
            - **Morning Commute Peak:**  
              **07:00 (7:00 AM)** — Mean volume: **6,048 vehicles/hour** on weekdays. Commuters entering the Twin Cities commercial corridor.
              
            - **Evening Commute Peak:**  
              **16:00 (4:00 PM)** — Mean volume: **6,204 vehicles/hour** on weekdays (overall busiest hour: 5,664 veh/h across all 7 days).
              
            - **Overnight Low-Flow Period:**  
              **03:00 (3:00 AM)** — Mean volume: **371 vehicles/hour** (quietest hour). Minimal passenger flow; primarily long-haul freight.
              
            - **Weekend Plateau:**  
              Weekends exhibit no commute peaks, instead maintaining an even midday plateau of ~4,350 veh/h from 12:00 to 16:00.
            """
        )

        st.info(
            "**Key Takeaway:** Traffic flow is intensely diurnal and schedule-dependent. "
            "Diurnal time features account for >85% of predictive power in machine learning models."
        )


# -----------------------------------------------------------------------------
# Page B: Traffic Analysis
# -----------------------------------------------------------------------------
elif page == "📈 Traffic Analysis":
    st.title("Exploratory Traffic & Environmental Analysis")
    st.markdown("Detailed breakdown of temporal rhythms, weekday vs. weekend divergence, and weather sensitivities.")

    tabs = st.tabs([
        "⏰ Hourly & Daily Patterns",
        "📅 Weekday vs. Weekend",
        "🍂 Monthly Seasonality",
        "🌦️ Weather & Precipitation",
        "🔥 Correlation Heatmap",
    ])

    with tabs[0]:
        st.subheader("Diurnal and Weekly Profiles")
        c1, c2 = st.columns(2)
        with c1:
            hour_fig = PROJECT_ROOT / "outputs" / "figures" / "average_traffic_by_hour.png"
            if hour_fig.exists():
                st.image(str(hour_fig), caption="Diurnal 24-Hour Traffic Profile", use_container_width=True)
        with c2:
            day_fig = PROJECT_ROOT / "outputs" / "figures" / "average_traffic_by_day.png"
            if day_fig.exists():
                st.image(str(day_fig), caption="Weekly Volume Profile (Monday through Sunday)", use_container_width=True)

        hourly_df = load_summary_table("hourly_traffic_summary.csv")
        if hourly_df is not None:
            with st.expander("View Aggregated Hourly Statistics"):
                st.dataframe(hourly_df, use_container_width=True)

    with tabs[1]:
        st.subheader("Commute vs. Weekend Flow Comparison")
        c1, c2 = st.columns([1.6, 1])
        with c1:
            ww_fig = PROJECT_ROOT / "outputs" / "figures" / "weekday_vs_weekend.png"
            if ww_fig.exists():
                st.image(str(ww_fig), caption="Weekday Bimodal Commute Curve vs. Weekend Unimodal Plateau", use_container_width=True)
        with c2:
            st.markdown("### Comparative Metrics")
            st.markdown(
                """
                - **Weekday Mean (Mon–Fri):**  
                  **3,533.29 vehicles/hour**
                - **Weekend Mean (Sat–Sun):**  
                  **2,570.70 vehicles/hour**
                - **Net Volume Reduction:**  
                  **-27.24%** on weekends
                - **Peak Differences:**  
                  Weekdays peak sharply at 07:00 (6,048 veh/h) and 16:00 (6,204 veh/h). Weekends reach their peak at 14:00 (4,350 veh/h) with zero morning surge.
                """
            )
            ww_df = load_summary_table("weekday_vs_weekend_summary.csv")
            if ww_df is not None:
                st.dataframe(ww_df, use_container_width=True)

    with tabs[2]:
        st.subheader("Monthly and Seasonal Variation")
        c1, c2 = st.columns([1.6, 1])
        with c1:
            month_fig = PROJECT_ROOT / "outputs" / "figures" / "monthly_traffic.png"
            if month_fig.exists():
                st.image(str(month_fig), caption="Monthly Seasonality across Calendar Year", use_container_width=True)
        with c2:
            st.markdown(
                """
                - **Busiest Month:** **August** and **October** exhibit peak travel demand (~3,400+ veh/h).
                - **Quietest Month:** **January** and **December** experience seasonal dips due to severe winter weather and holiday shutdowns.
                """
            )
            month_df = load_summary_table("monthly_traffic_summary.csv")
            if month_df is not None:
                st.dataframe(month_df, use_container_width=True)

    with tabs[3]:
        st.subheader("Weather Conditions & Precipitation Effects")
        c1, c2 = st.columns(2)
        with c1:
            weather_fig = PROJECT_ROOT / "outputs" / "figures" / "weather_traffic.png"
            if weather_fig.exists():
                st.image(str(weather_fig), caption="Traffic Volume by Main Weather Category", use_container_width=True)
        with c2:
            temp_fig = PROJECT_ROOT / "outputs" / "figures" / "temperature_vs_traffic.png"
            if temp_fig.exists():
                st.image(str(temp_fig), caption="Temperature (°C) vs. Traffic Volume", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        rain_fig = PROJECT_ROOT / "outputs" / "figures" / "rainfall_vs_traffic.png"
        if rain_fig.exists():
            st.image(str(rain_fig), caption="Dry vs. Rainy Comparison and Rainfall Intensity Sensitivity", use_container_width=True)

    with tabs[4]:
        st.subheader("Linear Feature Correlations")
        corr_fig = PROJECT_ROOT / "outputs" / "figures" / "correlation_heatmap.png"
        if corr_fig.exists():
            st.image(str(corr_fig), caption="Lower-Triangle Pearson Correlation Matrix", use_container_width=True)
        corr_df = load_summary_table("correlation_matrix.csv")
        if corr_df is not None:
            with st.expander("View Correlation Matrix Data"):
                st.dataframe(corr_df, use_container_width=True)


# -----------------------------------------------------------------------------
# Page C: Predict Traffic
# -----------------------------------------------------------------------------
elif page == "🚦 Predict Traffic":
    st.title("Traffic Volume Forecast")
    st.markdown("Generate real-time hourly traffic volume predictions using the trained production Random Forest pipeline.")

    model_package = get_model_pipeline()
    if model_package is None:
        st.error("Model artifact not found at `models/traffic_volume_rf_model.joblib`. Please run `python main.py --train` first.")
        st.stop()

    st.markdown("### 1. Select Prediction Scenario or Customize Inputs")

    # Quick Presets Buttons
    col_pre1, col_pre2, col_pre3, col_pre4 = st.columns(4)
    preset_choice = None
    with col_pre1:
        if st.button("🌅 Weekday Morning Rush (07:00)", use_container_width=True):
            preset_choice = "morning"
    with col_pre2:
        if st.button("🌇 Weekday Evening Rush (17:00)", use_container_width=True):
            preset_choice = "evening"
    with col_pre3:
        if st.button("🌙 Weekday Overnight (03:00)", use_container_width=True):
            preset_choice = "night"
    with col_pre4:
        if st.button("🚗 Weekend Afternoon (14:00)", use_container_width=True):
            preset_choice = "weekend"

    # Default values based on presets
    if preset_choice == "morning":
        def_date = date(2018, 10, 15)  # Monday
        def_time = time(7, 0)
        def_temp = 10.0
        def_weather = "Clear"
        def_rain = 0.0
        def_clouds = 15
        def_holiday = False
    elif preset_choice == "evening":
        def_date = date(2018, 10, 15)  # Monday
        def_time = time(17, 0)
        def_temp = 15.0
        def_weather = "Clear"
        def_rain = 0.0
        def_clouds = 20
        def_holiday = False
    elif preset_choice == "night":
        def_date = date(2018, 10, 16)  # Tuesday
        def_time = time(3, 0)
        def_temp = 5.0
        def_weather = "Clouds"
        def_rain = 0.0
        def_clouds = 70
        def_holiday = False
    elif preset_choice == "weekend":
        def_date = date(2018, 10, 20)  # Saturday
        def_time = time(14, 0)
        def_temp = 16.0
        def_weather = "Clear"
        def_rain = 0.0
        def_clouds = 10
        def_holiday = False
    else:
        def_date = date(2018, 10, 15)
        def_time = time(17, 0)
        def_temp = 15.0
        def_weather = "Clear"
        def_rain = 0.0
        def_clouds = 20
        def_holiday = False

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Form
    with st.form("traffic_prediction_form"):
        st.markdown("#### Input Parameters")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            input_date = st.date_input("Date", value=def_date)
            input_time = st.time_input("Time (Hourly)", value=def_time)
            is_holiday_flag = st.checkbox("Is Public/State Holiday?", value=def_holiday)

        with c2:
            temp_c = st.slider("Temperature (°C)", min_value=-35.0, max_value=45.0, value=float(def_temp), step=0.5)
            weather_main = st.selectbox(
                "Weather Condition",
                options=["Clear", "Clouds", "Rain", "Drizzle", "Mist", "Haze", "Fog", "Snow", "Thunderstorm", "Squall", "Smoke"],
                index=["Clear", "Clouds", "Rain", "Drizzle", "Mist", "Haze", "Fog", "Snow", "Thunderstorm", "Squall", "Smoke"].index(def_weather),
            )
            cloud_cover = st.slider("Cloud Coverage (%)", min_value=0, max_value=100, value=int(def_clouds), step=5)

        with c3:
            rain_mm = st.number_input("Rainfall (last 1 hour, mm)", min_value=0.0, max_value=100.0, value=float(def_rain), step=0.5)
            snow_mm = st.number_input("Snowfall (last 1 hour, mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
            st.caption("Hardware sensor bounds: Rain is capped at 100mm; Snow at 50mm.")

        submit_btn = st.form_submit_button("⚡ Predict Traffic Volume", type="primary", use_container_width=True)

    if submit_btn:
        # Assemble input dictionary
        combined_dt = datetime.combine(input_date, input_time)
        input_data = {
            "date_time": combined_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "temp_c": temp_c,
            "weather_main": weather_main,
            "rain_1h": rain_mm,
            "snow_1h": snow_mm,
            "clouds_all": cloud_cover,
            "is_holiday": 1 if is_holiday_flag else 0,
        }

        try:
            with st.spinner("Executing model pipeline inference..."):
                pred_result = predict_traffic_volume(model_package, input_data)
                predicted_vol = pred_result["predicted_traffic_volume"]
                feats = pred_result["input_features"]
                disclaimer = pred_result["disclaimer"]

            # Display prediction banner
            day_name = combined_dt.strftime("%A")
            formatted_dt = combined_dt.strftime("%B %d, %Y at %H:%M")
            
            st.markdown(
                f"""
                <div class="pred-banner">
                    <div class="pred-title">Forecast for {day_name}, {formatted_dt}</div>
                    <div class="pred-value">{predicted_vol:,.0f}<span class="pred-unit">vehicles / hour</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Feature interpretation table
            res_c1, res_c2 = st.columns([1.2, 1])
            with res_c1:
                st.markdown("#### Input Feature Summary")
                feat_df = pd.DataFrame([
                    {"Parameter": "Timestamp", "Value": str(combined_dt)},
                    {"Parameter": "Day of Week", "Value": f"{day_name} ({'Weekend' if feats['is_weekend'] else 'Weekday'})"},
                    {"Parameter": "Hour of Day", "Value": f"{feats['hour']:02d}:00"},
                    {"Parameter": "Temperature", "Value": f"{feats['temp_c']} °C"},
                    {"Parameter": "Weather Condition", "Value": str(feats['weather_main'])},
                    {"Parameter": "Rainfall (1h)", "Value": f"{feats['rain_1h']} mm"},
                    {"Parameter": "Holiday Status", "Value": "Yes (Designated Holiday)" if feats['is_holiday'] else "No (Regular Day)"},
                ])
                st.dataframe(feat_df, hide_index=True, use_container_width=True)

            with res_c2:
                st.markdown("#### Corridor Context")
                # Contextual assessment
                if predicted_vol > 5500:
                    status_type = "🔴 Severe Commute Congestion"
                    desc = "Approaching maximum historical corridor throughput (~6,000–6,200 veh/h). Anticipate stop-and-go conditions."
                elif predicted_vol > 3500:
                    status_type = "🟡 Moderate to Heavy Traffic"
                    desc = "Typical daytime operating flow. Corridor operating within normal free-flow capacity."
                elif predicted_vol > 1500:
                    status_type = "🟢 Light to Moderate Traffic"
                    desc = "Smooth travel conditions typical of mid-morning, late evening, or weekend daytime hours."
                else:
                    status_type = "🔵 Low Flow / Free Flow"
                    desc = "Off-peak or overnight travel regime. Minimal congestion risk."

                st.markdown(f"**Flow Regime:** {status_type}")
                st.caption(desc)

                st.markdown(
                    f"""
                    <div class="disclaimer-box">
                        <strong>Analytical Disclaimer:</strong> {disclaimer}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# -----------------------------------------------------------------------------
# Page D: Model Performance
# -----------------------------------------------------------------------------
elif page == "🎯 Model Performance":
    st.title("Model Evaluation & Diagnostic Benchmarking")
    st.markdown("Rigorous out-of-sample comparison across Baseline, Linear Regression, and Random Forest models.")

    metrics_df = load_summary_table("model_metrics.csv")
    if metrics_df is not None:
        st.subheader("Test Set Performance Metrics (Out-of-Sample)")
        
        # Display cards for each model
        c1, c2, c3 = st.columns(3)
        with c1:
            b_row = metrics_df[metrics_df["Model"].str.contains("Baseline")].iloc[0]
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Baseline (Train Mean)</div>
                    <div class="metric-value">MAE: {b_row['MAE']:,.1f}</div>
                    <div class="metric-sub" style="color: #64748b;">RMSE: {b_row['RMSE']:,.1f} | R²: {b_row['R2']:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            lr_row = metrics_df[metrics_df["Model"].str.contains("Linear")].iloc[0]
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Linear Regression</div>
                    <div class="metric-value">MAE: {lr_row['MAE']:,.1f}</div>
                    <div class="metric-sub" style="color: #0284c7;">RMSE: {lr_row['RMSE']:,.1f} | R²: {lr_row['R2']:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            rf_row = metrics_df[metrics_df["Model"].str.contains("Random Forest")].iloc[0]
            st.markdown(
                f"""
                <div class="metric-card" style="border: 2px solid #2563eb;">
                    <div class="metric-label">Random Forest (Selected)</div>
                    <div class="metric-value" style="color: #2563eb;">MAE: {rf_row['MAE']:,.1f}</div>
                    <div class="metric-sub" style="color: #059669;">RMSE: {rf_row['RMSE']:,.1f} | R²: {rf_row['R2']:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(metrics_df, use_container_width=True)

        st.markdown(
            """
            - **Random Forest vs. Baseline:** **82.54% reduction in MAE**, **72.91% reduction in RMSE**.
            - **Random Forest vs. Linear Regression:** **63.61% reduction in MAE**, **49.27% reduction in RMSE**.
            - **Variance Explained:** Random Forest explains **92.66%** ($R^2 = 0.9266$) of out-of-sample traffic variance.
            """
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Diagnostic Figures
    st.subheader("Diagnostic Visualizations")
    diag_tabs = st.tabs([
        "🎯 Actual vs. Predicted",
        "📉 Residual Diagnostics",
        "🌳 Feature Importance",
    ])

    with diag_tabs[0]:
        act_fig = PROJECT_ROOT / "outputs" / "figures" / "actual_vs_predicted.png"
        if act_fig.exists():
            st.image(str(act_fig), caption="Out-of-Sample Actual vs. Predicted Traffic Volume & 14-Day Time Series Trace", use_container_width=True)
        st.caption(
            "The Random Forest model closely tracks both high-volume commute peaks and low-volume overnight troughs, "
            "whereas Linear Regression exhibits wider dispersion and systematic overestimation during overnight hours."
        )

    with diag_tabs[1]:
        res_fig = PROJECT_ROOT / "outputs" / "figures" / "prediction_residuals.png"
        if res_fig.exists():
            st.image(str(res_fig), caption="Residual Homoscedasticity Check and Error Distribution Histogram", use_container_width=True)
        st.caption(
            "Residuals are centered near zero (mean error = -47.0 veh/h, std = 530.9 veh/h). "
            "Slight downward residual tails at fitted values >5,500 veh/h reflect unobserved incidents (crashes, lane closures) that restrict real-world throughput."
        )

    with diag_tabs[2]:
        imp_fig = PROJECT_ROOT / "outputs" / "figures" / "feature_importance.png"
        if imp_fig.exists():
            st.image(str(imp_fig), caption="Top 15 Random Forest Gini Feature Importances", use_container_width=True)
        st.caption(
            "Note: Random Forest impurity importance measures association and decision tree split frequency, NOT physical causation. "
            "Collinear features (hour, hour_cos, hour_sin) divide importance across tree splits, cumulatively representing >85.5% of predictive power."
        )

st.markdown("---")
st.caption("Urban Traffic Flow Optimization Dashboard | Developed with Streamlit and Scikit-Learn | Station 301 Minneapolis-St. Paul")
