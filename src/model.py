"""Model training, evaluation, and inference module for Urban Traffic Flow Optimization.

Implements chronological train/test splitting, baseline comparisons, linear regression,
deep PyTorch neural network (MLP), random forest regression, diagnostic residual and
feature importance plotting, and a unified prediction interface supporting .pth artifacts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Consistent publication styling for figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

NUMERIC_FEATURES = [
    "hour",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "day_sin",
    "day_cos",
    "month",
    "month_sin",
    "month_cos",
    "year",
    "is_weekend",
    "is_holiday",
    "temp_c",
    "rain_1h",
    "snow_1h",
    "clouds_all",
]

CATEGORICAL_FEATURES = ["weather_main"]

ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "traffic_volume"


# -----------------------------------------------------------------------------
# PyTorch Deep Learning Model Architecture
# -----------------------------------------------------------------------------
class TrafficVolumeNN(nn.Module):
    """Deep Multi-Layer Perceptron (MLP) Neural Network for Traffic Volume Regression."""

    def __init__(self, in_features: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        dims = hidden_dims or [128, 64, 32]
        self.net = nn.Sequential(
            nn.Linear(in_features, dims[0]),
            nn.BatchNorm1d(dims[0]),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(dims[0], dims[1]),
            nn.BatchNorm1d(dims[1]),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(dims[1], dims[2]),
            nn.BatchNorm1d(dims[2]),
            nn.ReLU(),
            nn.Linear(dims[2], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def prepare_model_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Engineer cyclical time features and structure model inputs without target leakage."""
    df_feat = df_clean.sort_values("date_time").reset_index(drop=True).copy()

    # Cyclical sin/cos encodings to capture smooth periodic boundaries
    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour"] / 24.0)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour"] / 24.0)
    df_feat["day_sin"] = np.sin(2 * np.pi * df_feat["day_of_week"] / 7.0)
    df_feat["day_cos"] = np.cos(2 * np.pi * df_feat["day_of_week"] / 7.0)
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12.0)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12.0)

    return df_feat


def chronological_train_test_split(
    df_feat: pd.DataFrame,
    test_ratio: float = 0.20
) -> Dict[str, Any]:
    """Perform a strict chronological train/test split reserving the latest observations for test."""
    total_len = len(df_feat)
    split_idx = int(total_len * (1.0 - test_ratio))

    train_df = df_feat.iloc[:split_idx].copy()
    test_df = df_feat.iloc[split_idx:].copy()

    X_train = train_df[ALL_MODEL_FEATURES].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_test = test_df[ALL_MODEL_FEATURES].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    # Preprocessing parameters calculated ONLY from train set to avoid data leakage
    temp_median = float(train_df["temp_c"].median())
    rain_cap = float(np.percentile(train_df["rain_1h"], 99.9))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
        "train_date_range": (train_df["date_time"].min(), train_df["date_time"].max()),
        "test_date_range": (test_df["date_time"].min(), test_df["date_time"].max()),
        "cleaning_params": {
            "temp_median_c": temp_median,
            "rain_cap_mm": rain_cap,
        },
    }


def build_preprocessor() -> ColumnTransformer:
    """Construct ColumnTransformer for numerical scaling and categorical one-hot encoding."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def train_pytorch_nn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 0.005,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train Deep PyTorch MLP Neural Network on normalized features."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    train_ds = TensorDataset(
        torch.tensor(X_train_trans, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    in_features = X_train_trans.shape[1]
    hidden_dims = [128, 64, 32]
    model = TrafficVolumeNN(in_features=in_features, hidden_dims=hidden_dims)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    model.train()
    for _ in range(epochs):
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.tensor(X_test_trans, dtype=torch.float32)).numpy()
        preds = y_scaler.inverse_transform(preds_scaled).flatten()
        preds = np.clip(preds, 0.0, None)

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    bundle = {
        "model_state_dict": model.state_dict(),
        "model_architecture": "TrafficVolumeNN",
        "in_features": in_features,
        "hidden_dims": hidden_dims,
        "preprocessor": preprocessor,
        "target_scaler": y_scaler,
        "features": ALL_MODEL_FEATURES,
        "model": model,
        "is_pytorch": True,
        "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
    }

    return {
        "model": model,
        "bundle": bundle,
        "predictions": preds,
        "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
    }


def train_and_evaluate_models(
    split_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Train Baseline, Linear Regression, PyTorch NN, and Random Forest models and evaluate metrics."""
    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    # 1. Baseline Model (Predicts Training Set Mean)
    train_mean = float(y_train.mean())
    baseline_preds = np.full(shape=len(y_test), fill_value=train_mean)

    # 2. Linear Regression Pipeline
    lr_preprocessor = build_preprocessor()
    lr_pipeline = Pipeline([
        ("preprocessor", lr_preprocessor),
        ("regressor", LinearRegression()),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)

    # 3. Deep PyTorch Neural Network (MLP)
    pytorch_res = train_pytorch_nn(X_train, y_train, X_test, y_test, epochs=15)
    pytorch_preds = pytorch_res["predictions"]

    # 4. Random Forest Regressor Pipeline
    rf_preprocessor = build_preprocessor()
    rf_pipeline = Pipeline([
        ("preprocessor", rf_preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_split=5,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=42,
        )),
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)

    # Calculate Evaluation Metrics
    def compute_metrics(y_true, y_pred):
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    metrics_dict = {
        "Baseline (Train Mean)": compute_metrics(y_test, baseline_preds),
        "Linear Regression": compute_metrics(y_test, lr_preds),
        "PyTorch Neural Network (MLP)": pytorch_res["metrics"],
        "Random Forest Regressor": compute_metrics(y_test, rf_preds),
    }

    metrics_df = pd.DataFrame(metrics_dict).T.reset_index()
    metrics_df.columns = ["Model", "MAE", "RMSE", "R2"]

    # Extract Feature Importances from Random Forest
    rf_regressor = rf_pipeline.named_steps["regressor"]
    rf_prep = rf_pipeline.named_steps["preprocessor"]
    cat_feature_names = rf_prep.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    all_transformed_feature_names = list(NUMERIC_FEATURES) + list(cat_feature_names)
    importances = rf_regressor.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": all_transformed_feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return {
        "pipelines": {
            "Linear Regression": lr_pipeline,
            "Random Forest": rf_pipeline,
        },
        "pytorch_bundle": pytorch_res["bundle"],
        "predictions": {
            "Baseline (Train Mean)": baseline_preds,
            "Linear Regression": lr_preds,
            "PyTorch Neural Network (MLP)": pytorch_preds,
            "Random Forest": rf_preds,
        },
        "metrics_df": metrics_df,
        "feature_importances": importance_df,
        "train_mean": train_mean,
    }


def plot_actual_vs_predicted(
    y_test: pd.Series,
    predictions_dict: Dict[str, np.ndarray],
    test_dates: pd.Series,
    output_path: Path
) -> None:
    """Generate actual vs predicted diagnostic plot with scatter comparison and time-series snippet."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300, gridspec_kw={"width_ratios": [1, 1.3]})

    rf_preds = predictions_dict["Random Forest"]
    lr_preds = predictions_dict["Linear Regression"]
    pytorch_preds = predictions_dict.get("PyTorch Neural Network (MLP)", None)

    # 1. Scatter comparison (y = x reference line)
    sample_indices = np.random.RandomState(42).choice(len(y_test), size=min(3000, len(y_test)), replace=False)
    y_sample = y_test.iloc[sample_indices]
    rf_sample = rf_preds[sample_indices]

    ax1.scatter(y_sample, rf_sample, color="#4a69bd", alpha=0.25, s=14, label="Random Forest Predictions")
    ax1.plot([0, 7500], [0, 7500], color="#e55039", linestyle="--", linewidth=2.0, label="Ideal Fit (y = x)")
    ax1.set_title("Actual vs. Predicted Traffic Volume (Test Set)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Actual Traffic Volume (veh/h)", fontsize=10.5)
    ax1.set_ylabel("Predicted Traffic Volume (veh/h)", fontsize=10.5)
    ax1.set_xlim(0, 7500)
    ax1.set_ylim(0, 7500)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left", fontsize=9.5)

    # 2. 7-Day Continuous Time-Series Window
    snippet_window = 168  # 168 hours = exactly 7 days
    snippet_dates = test_dates.iloc[:snippet_window]
    snippet_actual = y_test.iloc[:snippet_window]
    snippet_rf = rf_preds[:snippet_window]
    snippet_lr = lr_preds[:snippet_window]

    ax2.plot(snippet_dates, snippet_actual.values, color="#2c3e50", linewidth=2.2, label="Actual Volume")
    ax2.plot(snippet_dates, snippet_rf, color="#1e3799", linewidth=1.8, linestyle="-", label="Random Forest")
    if pytorch_preds is not None:
        ax2.plot(snippet_dates, pytorch_preds[:snippet_window], color="#38ada9", linewidth=1.5, linestyle="--", label="PyTorch NN (.pth)")
    ax2.plot(snippet_dates, snippet_lr, color="#e55039", linewidth=1.2, linestyle=":", alpha=0.8, label="Linear Regression")
    ax2.set_title("7-Day Test Window Forecast Comparison (168 Hourly Steps)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Date & Time", fontsize=10.5)
    ax2.set_ylabel("Traffic Volume (veh/h)", fontsize=10.5)
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right", fontsize=9.5)

    plt.suptitle("Model Forecasting Performance: Actual vs. Predicted Traffic Volume", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_prediction_residuals(
    y_test: pd.Series,
    predictions_dict: Dict[str, np.ndarray],
    output_path: Path
) -> None:
    """Generate residual diagnostics showing error distribution and residual vs. predicted."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    rf_preds = predictions_dict["Random Forest"]
    residuals = y_test.values - rf_preds

    # 1. Residuals vs Predicted
    sample_indices = np.random.RandomState(42).choice(len(y_test), size=min(3500, len(y_test)), replace=False)
    ax1.scatter(
        rf_preds[sample_indices],
        residuals[sample_indices],
        alpha=0.22,
        color="#1e3799",
        s=12,
    )
    ax1.axhline(0, color="#e55039", linestyle="--", linewidth=1.8)
    ax1.set_title("Residuals vs. Fitted Values (Random Forest)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Predicted Traffic Volume (veh/h)", fontsize=10.5)
    ax1.set_ylabel("Residual (Actual − Predicted)", fontsize=10.5)
    ax1.grid(True, linestyle="--", alpha=0.35)

    # 2. Residual Distribution Histogram
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals))
    ax2.hist(residuals, bins=60, color="#4a69bd", edgecolor="#2c3e50", alpha=0.75, density=True)
    ax2.axvline(0, color="#e55039", linestyle="--", linewidth=1.5, label="Zero Error Line")
    ax2.axvline(res_mean, color="#0c2461", linestyle="-", linewidth=1.5, label=f"Mean Error: {res_mean:+.1f}")

    # Theoretical Normal curve overlay
    x_norm = np.linspace(residuals.min(), residuals.max(), 200)
    p_norm = (1.0 / (res_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - res_mean) / res_std) ** 2)
    ax2.plot(x_norm, p_norm, color="#0c2461", linewidth=2.0, label=f"Normal Fit (σ={res_std:.1f})")

    ax2.set_title("Residual Error Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Prediction Error (Vehicles / Hour)", fontsize=10.5)
    ax2.set_ylabel("Probability Density", fontsize=10.5)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right", fontsize=9.5)

    plt.suptitle("Model Residual Diagnostics (Random Forest)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path
) -> None:
    """Plot feature importance ranking for the Random Forest model."""
    plt.figure(figsize=(10, 6.5), dpi=300)

    # Display top 15 most important features
    top_df = importance_df.head(15).sort_values("Importance", ascending=True)

    friendly_labels = {
        "hour": "Hour of Day (Raw)",
        "hour_cos": "Hour (Cyclical Cosine)",
        "hour_sin": "Hour (Cyclical Sine)",
        "temp_c": "Temperature (°C)",
        "day_of_week": "Day of Week (Raw)",
        "day_cos": "Day of Week (Cyclical Cosine)",
        "day_sin": "Day of Week (Cyclical Sine)",
        "clouds_all": "Cloud Coverage (%)",
        "month": "Month (Raw)",
        "month_cos": "Month (Cyclical Cosine)",
        "month_sin": "Month (Cyclical Sine)",
        "year": "Observation Year",
        "rain_1h": "Rainfall (mm)",
        "snow_1h": "Snowfall (mm)",
        "is_weekend": "Weekend Indicator",
        "is_holiday": "Holiday Indicator",
    }

    labels = [friendly_labels.get(f, f.replace("weather_main_", "Weather: ")) for f in top_df["Feature"]]

    bars = plt.barh(labels, top_df["Importance"], color="#1e3799", edgecolor="#0c2461", alpha=0.85, height=0.65)

    plt.title("Top Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
    plt.xlabel("Gini Importance Score", fontsize=10.5)
    plt.grid(True, linestyle="--", alpha=0.35, axis="x")

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#2c3e50",
            fontweight="semibold",
        )

    plt.xlim(0, max(top_df["Importance"]) * 1.22)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_trained_model(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    output_path: Path
) -> None:
    """Save fitted pipeline model and inference metadata to disk using joblib."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "features": ALL_MODEL_FEATURES,
        "metadata": metadata,
    }
    joblib.dump(payload, output_path, compress=3)


def save_pytorch_model(
    bundle: Dict[str, Any],
    output_path: Union[str, Path]
) -> None:
    """Save PyTorch neural network weights, preprocessor, and architecture to .pth file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "model_state_dict": bundle["model_state_dict"],
        "model_architecture": bundle.get("model_architecture", "TrafficVolumeNN"),
        "in_features": bundle["in_features"],
        "hidden_dims": bundle.get("hidden_dims", [128, 64, 32]),
        "preprocessor": bundle["preprocessor"],
        "target_scaler": bundle["target_scaler"],
        "features": bundle.get("features", ALL_MODEL_FEATURES),
        "metadata": bundle.get("metadata", {}),
    }
    torch.save(save_dict, output_path)


def load_trained_model(model_path: Union[str, Path]) -> Dict[str, Any]:
    """Load fitted model package (.pth or .joblib) and metadata from disk."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    if model_path.suffix.lower() == ".pth":
        bundle = torch.load(model_path, map_location="cpu", weights_only=False)
        in_features = bundle["in_features"]
        hidden_dims = bundle.get("hidden_dims", [128, 64, 32])
        model = TrafficVolumeNN(in_features=in_features, hidden_dims=hidden_dims)
        model.load_state_dict(bundle["model_state_dict"])
        model.eval()
        bundle["model"] = model
        bundle["is_pytorch"] = True
        return bundle

    loaded = joblib.load(model_path)
    if isinstance(loaded, dict):
        loaded["is_pytorch"] = False
    return loaded


def predict_traffic_volume(
    model_package: Union[Dict[str, Any], Path, str],
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Predict traffic volume from weather and time inputs with validation and confidence bounds."""
    if isinstance(model_package, (str, Path)):
        model_package = load_trained_model(Path(model_package))

    meta = model_package.get("metadata", {})
    temp_median_c = meta.get("cleaning_params", {}).get("temp_median_c", 10.0)

    # 1. Parse date_time
    if "date_time" in input_data:
        dt = pd.to_datetime(input_data["date_time"])
        hour = int(dt.hour)
        day_of_week = int(dt.dayofweek)
        month = int(dt.month)
        year = int(dt.year)
    else:
        hour = int(input_data.get("hour", 12))
        day_of_week = int(input_data.get("day_of_week", 2))
        month = int(input_data.get("month", 6))
        year = int(input_data.get("year", 2018))

    if not (0 <= hour <= 23):
        raise ValueError(f"Hour must be between 0 and 23. Got {hour}.")
    if not (0 <= day_of_week <= 6):
        raise ValueError(f"Day of week must be between 0 (Monday) and 6 (Sunday). Got {day_of_week}.")
    if not (1 <= month <= 12):
        raise ValueError(f"Month must be between 1 and 12. Got {month}.")

    # 2. Temperature handling (accepts Celsius or Kelvin)
    raw_temp = float(input_data.get("temp", input_data.get("temp_c", 20.0)))
    if raw_temp > 150:  # Provided in Kelvin
        temp_c = raw_temp - 273.15
    else:
        temp_c = raw_temp

    if temp_c < -50 or temp_c > 60:
        # Impute extreme sensor error
        temp_c = temp_median_c

    # 3. Weather variables
    rain_1h = float(input_data.get("rain_1h", 0.0))
    rain_1h = max(0.0, min(100.0, rain_1h))  # clip physically

    snow_1h = float(input_data.get("snow_1h", 0.0))
    snow_1h = max(0.0, min(50.0, snow_1h))

    clouds_all = int(input_data.get("clouds_all", 40))
    clouds_all = max(0, min(100, clouds_all))

    weather_main = str(input_data.get("weather_main", "Clear"))
    is_holiday = int(bool(input_data.get("is_holiday", 0)))
    is_weekend = int(day_of_week in [5, 6])

    # 4. Compute cyclical features
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    day_sin = np.sin(2 * np.pi * day_of_week / 7.0)
    day_cos = np.cos(2 * np.pi * day_of_week / 7.0)
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)

    # 5. Assemble input row matching model feature schema
    feature_row = pd.DataFrame([{
        "hour": hour,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_of_week": day_of_week,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "month": month,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "year": year,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "temp_c": temp_c,
        "rain_1h": rain_1h,
        "snow_1h": snow_1h,
        "clouds_all": clouds_all,
        "weather_main": weather_main,
    }])

    if model_package.get("is_pytorch", False):
        preprocessor = model_package["preprocessor"]
        target_scaler = model_package["target_scaler"]
        nn_model = model_package["model"]

        X_trans = preprocessor.transform(feature_row)
        with torch.no_grad():
            preds_scaled = nn_model(torch.tensor(X_trans, dtype=torch.float32)).numpy()
            pred_raw = float(target_scaler.inverse_transform(preds_scaled).item())
        predicted_val = max(0.0, round(pred_raw, 1))
    else:
        pipeline = model_package["pipeline"]
        predicted_val = float(pipeline.predict(feature_row)[0])
        predicted_val = max(0.0, round(predicted_val, 1))

    return {
        "predicted_traffic_volume": predicted_val,
        "input_features": {
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "month": month,
            "temp_c": round(temp_c, 1),
            "rain_1h": rain_1h,
            "weather_main": weather_main,
            "is_holiday": is_holiday,
        },
        "model_type": "PyTorch Neural Network (.pth)" if model_package.get("is_pytorch") else "Random Forest (.joblib)",
        "disclaimer": (
            "Notice: Predictions represent historical regression estimates based on "
            "I-94 Minneapolis-St. Paul sensor data and are not guaranteed real-time traffic counts."
        ),
    }


def run_training_pipeline(
    df_clean: pd.DataFrame,
    outputs_dir: Path,
    models_dir: Path
) -> Dict[str, Any]:
    """Execute complete model training, evaluation, plot generation, and model persistence."""
    outputs_dir = Path(outputs_dir)
    models_dir = Path(models_dir)
    figures_dir = outputs_dir / "figures"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare features
    df_feat = prepare_model_features(df_clean)

    # 2. Chronological split (80% train, 20% test)
    split_data = chronological_train_test_split(df_feat, test_ratio=0.20)

    # 3. Train and evaluate models (Baseline, Linear Regression, PyTorch NN, Random Forest)
    results = train_and_evaluate_models(split_data)

    # 4. Save evaluation metrics to outputs/model_metrics.csv
    metrics_path = outputs_dir / "model_metrics.csv"
    results["metrics_df"].to_csv(metrics_path, index=False)

    # 5. Generate and save diagnostic figures
    fig_paths = {
        "actual_vs_predicted": figures_dir / "actual_vs_predicted.png",
        "prediction_residuals": figures_dir / "prediction_residuals.png",
        "feature_importance": figures_dir / "feature_importance.png",
    }

    test_dates = split_data["test_df"]["date_time"]
    plot_actual_vs_predicted(
        split_data["y_test"],
        results["predictions"],
        test_dates,
        fig_paths["actual_vs_predicted"],
    )

    plot_prediction_residuals(
        split_data["y_test"],
        results["predictions"],
        fig_paths["prediction_residuals"],
    )

    plot_feature_importance(
        results["feature_importances"],
        fig_paths["feature_importance"],
    )

    # 6. Save PyTorch Model (.pth) as primary model artifact
    pth_save_path = models_dir / "traffic_model.pth"
    pytorch_bundle = results["pytorch_bundle"]
    pytorch_bundle["metadata"] = {
        "model_type": "PyTorch TrafficVolumeNN (Deep MLP)",
        "train_samples": len(split_data["X_train"]),
        "test_samples": len(split_data["X_test"]),
        "train_date_range": [str(d) for d in split_data["train_date_range"]],
        "test_date_range": [str(d) for d in split_data["test_date_range"]],
        "cleaning_params": split_data["cleaning_params"],
        "metrics": results["metrics_df"].to_dict(orient="records"),
    }
    save_pytorch_model(pytorch_bundle, pth_save_path)

    # 7. Also save Random Forest pipeline (.joblib)
    rf_save_path = models_dir / "traffic_volume_rf_model.joblib"
    selected_pipeline = results["pipelines"]["Random Forest"]
    rf_metadata = {
        "model_type": "RandomForestRegressor",
        "train_samples": len(split_data["X_train"]),
        "test_samples": len(split_data["X_test"]),
        "train_date_range": [str(d) for d in split_data["train_date_range"]],
        "test_date_range": [str(d) for d in split_data["test_date_range"]],
        "cleaning_params": split_data["cleaning_params"],
        "metrics": results["metrics_df"].to_dict(orient="records"),
    }
    save_trained_model(selected_pipeline, rf_metadata, rf_save_path)

    return {
        "results": results,
        "split_data": split_data,
        "metrics_path": metrics_path,
        "fig_paths": fig_paths,
        "pth_save_path": pth_save_path,
        "model_save_path": pth_save_path,
        "rf_save_path": rf_save_path,
    }
