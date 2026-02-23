"""
Production-oriented training and forecasting workflow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lightgbm import LGBMRegressor

from training import train_demand_models, train_rrp_models


def _default_param_grid() -> dict:
    return {
        "num_leaves": [15, 31, 63],
        "max_depth": [4, 6, 8, 10, -1],
        "learning_rate": [0.01, 0.03, 0.05, 0.08],
        "min_child_samples": [20, 40, 60, 100],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "lambda_l1": [0.0, 0.1, 0.5, 1.0],
        "lambda_l2": [0.0, 0.1, 0.5, 1.0],
    }


def _print_training_summary(training_results: dict, label_prefix: str) -> None:
    print(f"{label_prefix} model performance on the test set:")
    metrics_table = training_results["metrics_table"]
    if metrics_table.empty:
        raise RuntimeError(f"{label_prefix} training completed but metrics_table is empty.")
    print(metrics_table.to_string())


def _save_training_results(
    training_results: dict,
    output_path: Path,
    first_model_key: str,
    second_model_key: str,
    first_sheet: str,
    second_sheet: str,
) -> None:
    metrics_table = training_results["metrics_table"]
    with pd.ExcelWriter(output_path) as writer:
        metrics_table.to_excel(writer, sheet_name="metrics")
        pd.DataFrame([training_results[first_model_key]["best_params"]]).to_excel(
            writer, sheet_name=first_sheet, index=False
        )
        pd.DataFrame([training_results[second_model_key]["best_params"]]).to_excel(
            writer, sheet_name=second_sheet, index=False
        )
    print("Training results saved.")


def run_training_and_evaluation(
    df_features: pd.DataFrame,
    param_grid: dict | None = None,
    n_splits: int = 5,
    search_strategy: str = "randomized",
    n_trials: int = 100,
    early_stopping_rounds: int = 50,
    output_path: Path = Path("training_results.xlsx"),
) -> dict:
    """
    Train and evaluate demand models, then export evaluation artifacts.
    """
    if param_grid is None:
        param_grid = _default_param_grid()

    training_results = train_demand_models(
        df_features,
        param_grid=param_grid,
        n_splits=n_splits,
        search_strategy=search_strategy,
        n_trials=n_trials,
        early_stopping_rounds=early_stopping_rounds,
    )

    _print_training_summary(training_results, label_prefix="Demand")
    print()
    print("Best params (demand_log):")
    print(training_results["demand_log_model"]["best_params"])
    print()
    print("Best params (demand):")
    print(training_results["demand_model"]["best_params"])
    print()

    _save_training_results(
        training_results=training_results,
        output_path=output_path,
        first_model_key="demand_log_model",
        second_model_key="demand_model",
        first_sheet="best_params_demand_log",
        second_sheet="best_params_demand",
    )
    return training_results


def run_rrp_training_and_evaluation(
    df_features: pd.DataFrame,
    param_grid: dict | None = None,
    n_splits: int = 5,
    search_strategy: str = "randomized",
    n_trials: int = 100,
    early_stopping_rounds: int = 50,
    output_path: Path = Path("training_results_rrp.xlsx"),
) -> dict:
    """
    Train and evaluate RRP models, then export evaluation artifacts.
    """
    if param_grid is None:
        param_grid = _default_param_grid()

    training_results = train_rrp_models(
        df_features,
        param_grid=param_grid,
        n_splits=n_splits,
        search_strategy=search_strategy,
        n_trials=n_trials,
        early_stopping_rounds=early_stopping_rounds,
    )

    _print_training_summary(training_results, label_prefix="RRP")
    print()
    print("Best params (RRP_log):")
    print(training_results["rrp_log_model"]["best_params"])
    print()
    print("Best params (RRP):")
    print(training_results["rrp_model"]["best_params"])
    print()

    _save_training_results(
        training_results=training_results,
        output_path=output_path,
        first_model_key="rrp_log_model",
        second_model_key="rrp_model",
        first_sheet="best_params_rrp_log",
        second_sheet="best_params_rrp",
    )
    return training_results


def _build_quantile_forecast(
    df_features: pd.DataFrame,
    data_production_features: pd.DataFrame,
    training_results: dict,
    target_to_forecast: str,
    random_state: int,
    pessimistic_alpha: float,
    optimistic_alpha: float,
    future_feature_overrides: dict[str, pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.Index]:
    if not (0.0 < pessimistic_alpha < 0.5):
        raise ValueError("pessimistic_alpha must be > 0 and < 0.5.")
    if not (0.5 < optimistic_alpha < 1.0):
        raise ValueError("optimistic_alpha must be > 0.5 and < 1.")

    candidate_model_keys = [
        f"{target_to_forecast}_model",
        f"{target_to_forecast.lower()}_model",
    ]
    model_key = next((k for k in candidate_model_keys if k in training_results), None)
    if model_key is None:
        raise KeyError(
            f"{candidate_model_keys[0]} not found in training_results. Available keys: "
            f"{sorted(training_results.keys())}"
        )

    best_params = training_results[model_key]["best_params"]
    feature_columns = training_results[model_key]["feature_columns"]
    missing_feature_columns = [
        col for col in feature_columns if col not in data_production_features.columns
    ]
    if missing_feature_columns:
        raise KeyError(
            "Missing required production feature columns: "
            f"{missing_feature_columns}"
        )

    full_train_df = df_features[feature_columns + [target_to_forecast]].dropna().copy()
    X_full = full_train_df[feature_columns]
    y_full = full_train_df[target_to_forecast]

    future_index = data_production_features.index[
        data_production_features.index > df_features.index.max()
    ]
    if future_index.empty:
        raise ValueError(
            "No future dates found in data_production_features index. "
            "Expected dates strictly greater than the max date in df_features."
        )

    X_future = data_production_features.loc[future_index, feature_columns].copy()
    if future_feature_overrides:
        for col, override in future_feature_overrides.items():
            if col not in X_future.columns:
                continue
            X_future[col] = pd.Series(override).reindex(future_index).values

    quantile_alphas = {
        "pessimistic": pessimistic_alpha,
        "p50": 0.50,
        "optimistic": optimistic_alpha,
    }
    quantile_preds: dict[str, np.ndarray] = {}

    for q_name, alpha in quantile_alphas.items():
        q_model = LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=5000,
            n_jobs=-1,
            random_state=random_state,
            verbosity=-1,
            verbose=-1,
            **best_params,
        )
        q_model.fit(X_full, y_full)
        quantile_preds[q_name] = q_model.predict(X_future)

    is_log_target = target_to_forecast.lower().endswith("_log")
    forecast_quantiles = pd.DataFrame(index=future_index)
    if is_log_target:
        forecast_quantiles["forecast_pessimistic"] = np.expm1(quantile_preds["pessimistic"])
        forecast_quantiles["forecast_p50"] = np.expm1(quantile_preds["p50"])
        forecast_quantiles["forecast_optimistic"] = np.expm1(quantile_preds["optimistic"])
    else:
        forecast_quantiles["forecast_pessimistic"] = quantile_preds["pessimistic"]
        forecast_quantiles["forecast_p50"] = quantile_preds["p50"]
        forecast_quantiles["forecast_optimistic"] = quantile_preds["optimistic"]

    q_sorted = np.sort(
        forecast_quantiles[
            ["forecast_pessimistic", "forecast_p50", "forecast_optimistic"]
        ].values,
        axis=1,
    )
    forecast_quantiles["forecast_pessimistic"] = q_sorted[:, 0]
    forecast_quantiles["forecast_p50"] = q_sorted[:, 1]
    forecast_quantiles["forecast_optimistic"] = q_sorted[:, 2]

    return forecast_quantiles, future_index


def _merge_forecast_into_features(
    df_features: pd.DataFrame,
    forecast_quantiles: pd.DataFrame,
    actual_col: str,
    full_col: str,
) -> pd.DataFrame:
    out = df_features.join(forecast_quantiles, how="outer").sort_index()
    out[full_col] = out[actual_col]
    out.loc[forecast_quantiles.index, full_col] = forecast_quantiles["forecast_p50"]
    return out


def _create_forecast_figure(
    historical_series: pd.Series,
    forecast_quantiles: pd.DataFrame,
    title_prefix: str,
    yaxis_title: str,
    target_to_forecast: str,
    pessimistic_alpha: float,
    optimistic_alpha: float,
) -> go.Figure:
    fig = go.Figure()

    historical_index = historical_series.index
    future_index = forecast_quantiles.index
    horizon_days = len(future_index)
    forecast_end = future_index.max().date()

    fig.add_trace(
        go.Scatter(
            x=historical_index,
            y=historical_series.values,
            mode="lines",
            name=f"Historical {title_prefix}",
            line=dict(color="blue", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_index,
            y=forecast_quantiles["forecast_p50"],
            mode="lines",
            name="Forecast P50 (main)",
            line=dict(color="orange", width=2),
        )
    )

    pessimistic_label = int(round(pessimistic_alpha * 100))
    optimistic_label = int(round(optimistic_alpha * 100))

    fig.add_trace(
        go.Scatter(
            x=future_index,
            y=forecast_quantiles["forecast_pessimistic"],
            mode="lines",
            name=f"Forecast P{pessimistic_label} (pessimistic)",
            line=dict(color="red", width=1.5, dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_index,
            y=forecast_quantiles["forecast_optimistic"],
            mode="lines",
            name=f"Forecast P{optimistic_label} (optimistic)",
            line=dict(color="green", width=1.5, dash="dot"),
        )
    )

    fig.update_layout(
        title=(
            f"{title_prefix} Forecast ({horizon_days} days, through {forecast_end}) "
            f"using target={target_to_forecast}"
        ),
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        template="plotly_white",
        height=550,
        hovermode="x unified",
    )
    return fig


def run_full_retraining_and_forecast(
    df_features: pd.DataFrame,
    data_production_features: pd.DataFrame,
    training_results: dict,
    target_to_forecast: str = "demand_log",
    random_state: int = 42,
    pessimistic_alpha: float = 0.10,
    optimistic_alpha: float = 0.90,
    show_figure: bool = True,
) -> dict:
    """
    Retrain demand quantile models on full historical data and forecast.
    """
    forecast_quantiles, future_index = _build_quantile_forecast(
        df_features=df_features,
        data_production_features=data_production_features,
        training_results=training_results,
        target_to_forecast=target_to_forecast,
        random_state=random_state,
        pessimistic_alpha=pessimistic_alpha,
        optimistic_alpha=optimistic_alpha,
    )
    df_features_with_forecast = _merge_forecast_into_features(
        df_features=df_features,
        forecast_quantiles=forecast_quantiles,
        actual_col="demand",
        full_col="demand_full",
    )

    print(
        f"Demand forecast preview (P{int(round(pessimistic_alpha * 100))}/P50/"
        f"P{int(round(optimistic_alpha * 100))}):"
    )
    print(forecast_quantiles.head())

    fig = _create_forecast_figure(
        historical_series=df_features["demand"],
        forecast_quantiles=forecast_quantiles,
        title_prefix="Demand",
        yaxis_title="Demand",
        target_to_forecast=target_to_forecast,
        pessimistic_alpha=pessimistic_alpha,
        optimistic_alpha=optimistic_alpha,
    )
    if show_figure:
        fig.show()

    return {
        "forecast_quantiles": forecast_quantiles,
        "future_index": future_index,
        "df_features_with_forecast": df_features_with_forecast,
        "figure": fig,
    }


def run_full_rrp_retraining_and_forecast(
    df_features: pd.DataFrame,
    data_production_features: pd.DataFrame,
    rrp_training_results: dict,
    demand_forecast_quantiles: pd.DataFrame,
    target_to_forecast: str = "RRP_log",
    random_state: int = 42,
    pessimistic_alpha: float = 0.10,
    optimistic_alpha: float = 0.90,
    show_figure: bool = True,
) -> dict:
    """
    Retrain RRP quantile models on full historical data and forecast.
    Uses demand P50 forecast as future demand feature.
    """
    forecast_quantiles, future_index = _build_quantile_forecast(
        df_features=df_features,
        data_production_features=data_production_features,
        training_results=rrp_training_results,
        target_to_forecast=target_to_forecast,
        random_state=random_state,
        pessimistic_alpha=pessimistic_alpha,
        optimistic_alpha=optimistic_alpha,
        future_feature_overrides={
            "demand": demand_forecast_quantiles["forecast_p50"],
        },
    )
    df_features_with_forecast = _merge_forecast_into_features(
        df_features=df_features,
        forecast_quantiles=forecast_quantiles,
        actual_col="RRP",
        full_col="RRP_full",
    )

    print(
        f"RRP forecast preview (P{int(round(pessimistic_alpha * 100))}/P50/"
        f"P{int(round(optimistic_alpha * 100))}):"
    )
    print(forecast_quantiles.head())

    fig = _create_forecast_figure(
        historical_series=df_features["RRP"],
        forecast_quantiles=forecast_quantiles,
        title_prefix="RRP",
        yaxis_title="RRP",
        target_to_forecast=target_to_forecast,
        pessimistic_alpha=pessimistic_alpha,
        optimistic_alpha=optimistic_alpha,
    )
    if show_figure:
        fig.show()

    return {
        "forecast_quantiles": forecast_quantiles,
        "future_index": future_index,
        "df_features_with_forecast": df_features_with_forecast,
        "figure": fig,
    }


def build_revenue_dataframe(
    df_features: pd.DataFrame,
    demand_forecast_quantiles: pd.DataFrame,
    rrp_forecast_quantiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge historical and forecasted demand/RRP quantiles and compute revenues.
    """
    out_index = df_features.index.union(demand_forecast_quantiles.index).union(
        rrp_forecast_quantiles.index
    )
    out = pd.DataFrame(index=out_index).sort_index()

    out["demand_p50"] = df_features["demand"].reindex(out.index)
    out["demand_p10"] = out["demand_p50"]
    out["demand_p90"] = out["demand_p50"]
    out.loc[demand_forecast_quantiles.index, "demand_p10"] = demand_forecast_quantiles[
        "forecast_pessimistic"
    ]
    out.loc[demand_forecast_quantiles.index, "demand_p50"] = demand_forecast_quantiles[
        "forecast_p50"
    ]
    out.loc[demand_forecast_quantiles.index, "demand_p90"] = demand_forecast_quantiles[
        "forecast_optimistic"
    ]

    out["rrp_p50"] = df_features["RRP"].reindex(out.index)
    out["rrp_p10"] = out["rrp_p50"]
    out["rrp_p90"] = out["rrp_p50"]
    out.loc[rrp_forecast_quantiles.index, "rrp_p10"] = rrp_forecast_quantiles[
        "forecast_pessimistic"
    ]
    out.loc[rrp_forecast_quantiles.index, "rrp_p50"] = rrp_forecast_quantiles[
        "forecast_p50"
    ]
    out.loc[rrp_forecast_quantiles.index, "rrp_p90"] = rrp_forecast_quantiles[
        "forecast_optimistic"
    ]

    out["revenue"] = out["rrp_p50"] * out["demand_p50"]
    out["revenue_optimistic"] = out["rrp_p90"] * out["demand_p90"]
    out["revenue_pessimistic"] = out["rrp_p10"] * out["demand_p10"]
    forecast_index = demand_forecast_quantiles.index.union(rrp_forecast_quantiles.index)
    out["is_forecast"] = out.index.isin(forecast_index)

    return out


def create_full_series_figure(
    df: pd.DataFrame,
    p50_col: str,
    p10_col: str,
    p90_col: str,
    title: str,
    yaxis_title: str,
) -> go.Figure:
    forecast_mask = (
        df["is_forecast"].fillna(False).astype(bool)
        if "is_forecast" in df.columns
        else pd.Series(False, index=df.index)
    )
    historical_mask = ~forecast_mask

    historical_series = df[p50_col].where(historical_mask)
    forecast_p50_series = df[p50_col].where(forecast_mask)
    forecast_p10_series = df[p10_col].where(forecast_mask)
    forecast_p90_series = df[p90_col].where(forecast_mask)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=historical_series,
            mode="lines",
            name="Historical",
            line=dict(color="blue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=forecast_p50_series,
            mode="lines",
            name="P50 forecast",
            line=dict(color="orange", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=forecast_p10_series,
            mode="lines",
            name="P10 (pessimistic)",
            line=dict(color="red", width=1.5, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=forecast_p90_series,
            mode="lines",
            name="P90 (optimistic)",
            line=dict(color="green", width=1.5, dash="dot"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        template="plotly_white",
        height=550,
        hovermode="x unified",
    )
    return fig


def build_quarterly_scenario_table(
    revenue_df: pd.DataFrame,
    periods: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a quarterly summary by scenario with demand totals, average RRP, and revenues.
    """
    if periods is None:
        periods = ["2020Q4", "2021Q1", "2021Q2", "2021Q3", "2021Q4"]

    quarter_labels = revenue_df.index.to_period("Q").astype(str)
    mask = quarter_labels.isin(periods)
    if not mask.any():
        raise ValueError(f"No rows found for requested periods: {periods}")

    window_df = revenue_df.loc[mask].copy()
    window_df["quarter"] = window_df.index.to_period("Q").astype(str)

    scenario_definitions = [
        ("Pessimistic", "demand_p10", "rrp_p10", "revenue_pessimistic"),
        ("Expected", "demand_p50", "rrp_p50", "revenue"),
        ("Optimistic", "demand_p90", "rrp_p90", "revenue_optimistic"),
    ]

    rows: list[dict[str, float | str]] = []
    for quarter, quarter_df in window_df.groupby("quarter", sort=False):
        for scenario_name, demand_col, rrp_col, revenue_col in scenario_definitions:
            rows.append(
                {
                    "quarter": quarter,
                    "scenario": scenario_name,
                    "total_demand": float(quarter_df[demand_col].sum()),
                    "average_rrp": float(quarter_df[rrp_col].mean()),
                    "expected_revenue": float(quarter_df[revenue_col].sum()),
                }
            )

    return pd.DataFrame(rows)


def run_forecasting_product(
    df_features: pd.DataFrame,
    data_production_features: pd.DataFrame,
    target_to_forecast: str = "demand_log",
    param_grid: dict | None = None,
    n_splits: int = 5,
    search_strategy: str = "randomized",
    n_trials: int = 100,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
    pessimistic_alpha: float = 0.10,
    optimistic_alpha: float = 0.90,
    output_path: Path = Path("training_results.xlsx"),
    show_figure: bool = True,
) -> dict:
    """
    Backward-compatible wrapper that executes demand training/evaluation then forecast.
    """
    training_results = run_training_and_evaluation(
        df_features=df_features,
        param_grid=param_grid,
        n_splits=n_splits,
        search_strategy=search_strategy,
        n_trials=n_trials,
        early_stopping_rounds=early_stopping_rounds,
        output_path=output_path,
    )

    forecast_output = run_full_retraining_and_forecast(
        df_features=df_features,
        data_production_features=data_production_features,
        training_results=training_results,
        target_to_forecast=target_to_forecast,
        random_state=random_state,
        pessimistic_alpha=pessimistic_alpha,
        optimistic_alpha=optimistic_alpha,
        show_figure=show_figure,
    )

    return {
        "training_results": training_results,
        **forecast_output,
    }

