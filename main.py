"""Main entry point for Urban Traffic Flow Optimization project.

Runs initial dataset inspection, validates data types, reports missing
values, duplicates, and verifies target traffic volume statistics.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path to guarantee imports regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_traffic_data, resolve_dataset_path
from src.inspector import inspect_traffic_dataset, print_inspection_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Urban Traffic Flow Optimization - Dataset Inspection & EDA Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Custom path to the Metro Interstate Traffic Volume CSV file. "
             "If omitted, standard locations (e.g. data/Metro_Interstate_Traffic_Volume.csv) are searched.",
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Run Exploratory Data Analysis (EDA) pipeline: clean data, extract features, "
             "compute temporal/weather aggregations, generate 9 diagnostic figures and summary tables.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train and evaluate regression models (Baseline Mean, Linear Regression, Random Forest) "
             "using chronological 80/20 train/test split. Generates evaluation metrics, diagnostic plots, and saves model.",
    )
    parser.add_argument(
        "--predict",
        type=str,
        nargs="?",
        const="2018-10-15 17:00:00",
        default=None,
        help="Run inference on a given timestamp (e.g. '2018-10-15 17:00:00') using the saved Random Forest model. "
             "If omitted, sample predictions for morning peak and evening peak are demonstrated.",
    )
    return parser.parse_args()


def main():
    """Execute the dataset inspection, EDA, or model training/inference workflow."""
    args = parse_args()

    print("\nStarting Urban Traffic Flow Optimization system...")
    try:
        resolved_path = resolve_dataset_path(args.data_path)
        print(f"Loading dataset from: {resolved_path}")
        df = load_traffic_data(resolved_path, parse_dates=True)
    except FileNotFoundError as fnf_err:
        print(f"\n[ERROR] File Not Found:\n{fnf_err}", file=sys.stderr)
        return 1
    except ValueError as val_err:
        print(f"\n[ERROR] Schema Validation Failed:\n{val_err}", file=sys.stderr)
        return 1
    except RuntimeError as run_err:
        print(f"\n[ERROR] Data Loading Failed:\n{run_err}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[UNEXPECTED ERROR] {exc}", file=sys.stderr)
        return 1

    try:
        inspection_results = inspect_traffic_dataset(df)
        print_inspection_report(inspection_results, file_path=str(resolved_path))
    except Exception as exc:
        print(f"\n[ERROR] Inspection failed during analysis: {exc}", file=sys.stderr)
        return 1

    outputs_dir = PROJECT_ROOT / "outputs"
    models_dir = PROJECT_ROOT / "models"

    # If --eda flag is supplied, execute Phase 2 EDA pipeline
    if args.eda:
        from src.eda import generate_all_eda_outputs

        print("\n" + "=" * 70)
        print(" EXECUTING EXPLORATORY DATA ANALYSIS (EDA) PIPELINE")
        print("=" * 70)

        try:
            eda_results = generate_all_eda_outputs(df, outputs_dir)
            audit = eda_results["audit"]
            patterns = eda_results["patterns"]

            print("\n[EDA Audit Results]")
            print(f"  - Initial Raw Records     : {audit['initial_rows']:,}")
            print(f"  - Duplicate Records Found : {audit['duplicates_removed']:,} (Removed safely)")
            print(f"  - Cleaned Records Intact  : {audit['final_rows']:,}")
            print(f"  - Zero-Traffic Records    : {audit['zero_traffic_before']} before, {audit['zero_traffic_after']} after (intact)")
            print(f"  - Holiday Distribution    : {audit['holiday_count']:,} holidays, {audit['non_holiday_count']:,} regular days")

            print("\n[EDA Key Temporal Highlights]")
            print(f"  - Busiest Hour            : {patterns['busiest_hour']:02d}:00 (Mean: {patterns['busiest_hour_mean']:,.2f} veh/h)")
            print(f"  - Quietest Hour           : {patterns['quietest_hour']:02d}:00 (Mean: {patterns['quietest_hour_mean']:,.2f} veh/h)")
            print(f"  - Busiest Day             : {patterns['busiest_day']} (Mean: {patterns['busiest_day_mean']:,.2f} veh/h)")
            print(f"  - Quietest Day            : {patterns['quietest_day']} (Mean: {patterns['quietest_day_mean']:,.2f} veh/h)")

            ww = patterns["weekday_vs_weekend_summary"]
            wd_mean = ww.loc[ww['is_weekend'] == 0, 'mean'].values[0]
            we_mean = ww.loc[ww['is_weekend'] == 1, 'mean'].values[0]
            print(f"  - Weekday vs. Weekend     : Weekday Mean = {wd_mean:,.2f} vs Weekend Mean = {we_mean:,.2f} veh/h")

            print("\n[Generated Figures in outputs/figures/]")
            for name, p in eda_results["figure_paths"].items():
                print(f"  - {name:<25}: {p.name}")

            print("\n[Generated Data Summaries in outputs/]")
            for name, p in eda_results["csv_paths"].items():
                print(f"  - {name:<25}: {p.name}")
            print(f"  - eda_summary_text         : {eda_results['summary_txt_path'].name}")
            print("=" * 70)

        except Exception as exc:
            print(f"\n[ERROR] EDA Pipeline failed: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    # If --train flag is supplied, execute Model Training & Evaluation pipeline
    if args.train:
        from src.eda import clean_and_engineer_features
        from src.model import run_training_pipeline

        print("\n" + "=" * 70)
        print(" EXECUTING REGRESSION MODEL TRAINING & EVALUATION PIPELINE")
        print("=" * 70)

        try:
            df_clean, audit = clean_and_engineer_features(df)
            train_output = run_training_pipeline(df_clean, outputs_dir, models_dir)

            metrics_df = train_output["results"]["metrics_df"]
            split_data = train_output["split_data"]

            print("\n[Chronological Train/Test Partition]")
            print(f"  - Training Set            : {len(split_data['X_train']):,} samples ({split_data['train_date_range'][0]} to {split_data['train_date_range'][1]})")
            print(f"  - Test Set (Latest 20%)   : {len(split_data['X_test']):,} samples ({split_data['test_date_range'][0]} to {split_data['test_date_range'][1]})")

            print("\n[Model Performance Evaluation Metrics (Out-of-Sample Test Set)]")
            print(metrics_df.to_string(index=False))

            print("\n[Generated Model Diagnostic Figures in outputs/figures/]")
            for name, p in train_output["fig_paths"].items():
                print(f"  - {name:<25}: {p.name}")

            print(f"\n[Model Saved]             : {train_output['model_save_path'].name}")
            print("=" * 70)

        except Exception as exc:
            print(f"\n[ERROR] Model training failed: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    # If --predict flag is supplied, demonstrate inference
    if args.predict is not None:
        from src.model import predict_traffic_volume

        model_path = models_dir / "traffic_model.pth"
        if not model_path.exists():
            print(f"\n[ERROR] Model file not found at '{model_path}'. Please run with --train first.", file=sys.stderr)
            return 1

        print("\n" + "=" * 70)
        print(" TRAFFIC VOLUME PREDICTION INFERENCE")
        print("=" * 70)

        sample_scenarios = [
            {
                "label": "Evening Rush Hour (Weekday)",
                "data_time": args.predict if args.predict else "2018-10-15 17:00:00",
                "temp_c": 15.0,
                "weather_main": "Clear",
                "rain_1h": 0.0,
                "clouds_all": 10,
                "is_holiday": 0,
            },
            {
                "label": "Overnight Low Flow (Weekday 03:00 AM)",
                "data_time": "2018-10-16 03:00:00",
                "temp_c": 8.0,
                "weather_main": "Clouds",
                "rain_1h": 0.0,
                "clouds_all": 75,
                "is_holiday": 0,
            },
            {
                "label": "Weekend Afternoon Leisure Flow (Saturday 14:00 PM)",
                "data_time": "2018-10-20 14:00:00",
                "temp_c": 18.0,
                "weather_main": "Clear",
                "rain_1h": 0.0,
                "clouds_all": 20,
                "is_holiday": 0,
            },
        ]

        for sc in sample_scenarios:
            input_spec = {
                "date_time": sc["data_time"],
                "temp_c": sc["temp_c"],
                "weather_main": sc["weather_main"],
                "rain_1h": sc["rain_1h"],
                "clouds_all": sc["clouds_all"],
                "is_holiday": sc["is_holiday"],
            }
            pred_res = predict_traffic_volume(model_path, input_spec)
            print(f"\nScenario: {sc['label']}")
            print(f"  Input Timestamp         : {sc['data_time']}")
            print(f"  Weather Conditions      : {sc['weather_main']}, {sc['temp_c']}°C, Rain: {sc['rain_1h']}mm")
            print(f"  Predicted Traffic Volume: {pred_res['predicted_traffic_volume']:,.0f} vehicles/hour")
            print(f"  {pred_res['disclaimer']}")

        print("=" * 70)

    print("\nWorkflow completed successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
