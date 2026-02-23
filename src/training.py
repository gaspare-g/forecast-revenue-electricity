"""
Training utilities for demand forecasting with LightGBM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit


def get_lgbm_param_grid() -> dict:
    """
    Default grid used to derive randomized-search intervals.
    """
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


def get_demand_training_features(df: pd.DataFrame) -> list[str]:
    """
    Build the feature list requested for demand forecasting.
    """
    base_features = [
        "min_temperature",
        "max_temperature",
        "solar_exposure",
        "rainfall",
        "school_day",
        "holiday",
        "day",
        "month",
        "year",
        "day_of_week",
        "fourier_year_sin_1",
        "fourier_year_cos_1",
        "fourier_year_sin_2",
        "fourier_year_cos_2",
    ]

    temp_sq_features = sorted(
        [
            col
            for col in df.columns
            if "temperature" in col.lower() and col.endswith("_sq")
        ]
    )
    temp_month_interactions = sorted(
        [
            col
            for col in df.columns
            if "temperature" in col.lower() and col.endswith("_x_month")
        ]
    )

    requested = (
        base_features
        + temp_sq_features
        + temp_month_interactions
    )
    return [col for col in requested if col in df.columns]


def get_rrp_training_features(df: pd.DataFrame) -> list[str]:
    """
    Build the feature list for RRP forecasting.
    Uses demand feature set plus demand as an additional predictor.
    """
    requested = get_demand_training_features(df) + ["demand"]
    return [col for col in requested if col in df.columns]


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    ape = np.abs(y_true - y_pred) / denom
    return float(np.nanmean(ape) * 100)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": _safe_mape(np.asarray(y_true), np.asarray(y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def _time_series_train_test_split(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_idx, test_idx = list(tscv.split(X))[-1]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def _build_model(params: dict, random_state: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        random_state=random_state,
        n_estimators=5000,
        n_jobs=-1,
        verbosity=-1,
        verbose=-1,
        **params,
    )


def _build_interval_param_distributions(param_grid: dict) -> dict:
    """
    Convert a coarse grid into interval-based distributions for RandomizedSearchCV.
    """
    max_depth_candidates = sorted({int(v) for v in param_grid["max_depth"] if int(v) != -1})
    if not max_depth_candidates:
        raise ValueError("param_grid['max_depth'] must contain at least one value different from -1.")

    return {
        "num_leaves": randint(
            int(min(param_grid["num_leaves"])),
            int(max(param_grid["num_leaves"])) + 1,
        ),
        "max_depth": [-1, *range(max_depth_candidates[0], max_depth_candidates[-1] + 1)],
        "learning_rate": uniform(
            float(min(param_grid["learning_rate"])),
            float(max(param_grid["learning_rate"])) - float(min(param_grid["learning_rate"])),
        ),
        "min_child_samples": randint(
            int(min(param_grid["min_child_samples"])),
            int(max(param_grid["min_child_samples"])) + 1,
        ),
        "subsample": uniform(
            float(min(param_grid["subsample"])),
            float(max(param_grid["subsample"])) - float(min(param_grid["subsample"])),
        ),
        "colsample_bytree": uniform(
            float(min(param_grid["colsample_bytree"])),
            float(max(param_grid["colsample_bytree"])) - float(min(param_grid["colsample_bytree"])),
        ),
        "lambda_l1": uniform(
            float(min(param_grid["lambda_l1"])),
            float(max(param_grid["lambda_l1"])) - float(min(param_grid["lambda_l1"])),
        ),
        "lambda_l2": uniform(
            float(min(param_grid["lambda_l2"])),
            float(max(param_grid["lambda_l2"])) - float(min(param_grid["lambda_l2"])),
        ),
    }


def _cv_rmse_with_early_stopping(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_splits: int,
    random_state: int,
    early_stopping_rounds: int,
) -> float:
    fold_scores = []
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for tr_idx, va_idx in tscv.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = _build_model(params=params, random_state=random_state)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[early_stopping(early_stopping_rounds, verbose=False)],
        )
        pred = model.predict(X_va)
        fold_scores.append(np.sqrt(mean_squared_error(y_va, pred)))

    return float(np.mean(fold_scores))


def _search_randomized(
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict,
    n_splits: int,
    n_trials: int,
    random_state: int,
    early_stopping_rounds: int,
) -> tuple[dict, float]:
    param_distributions = _build_interval_param_distributions(param_grid)
    sampled_configs = list(
        ParameterSampler(
            param_distributions=param_distributions,
            n_iter=n_trials,
            random_state=random_state,
        )
    )

    best_params = None
    best_rmse = np.inf

    for trial_idx, sampled_params in enumerate(sampled_configs, start=1):
        params = {k: v.item() if hasattr(v, "item") else v for k, v in sampled_params.items()}
        cv_rmse = _cv_rmse_with_early_stopping(
            X=X,
            y=y,
            params=params,
            n_splits=n_splits,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
        )

        print(f"[randomized][trial {trial_idx}/{n_trials}] cv_rmse={cv_rmse:.5f}")
        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best_params = params
            print(
                f"[randomized][trial {trial_idx}/{n_trials}] new_best_rmse={best_rmse:.5f}"
            )

    return best_params, float(best_rmse)


def _fit_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    random_state: int,
    early_stopping_rounds: int,
) -> LGBMRegressor:
    split_at = int(len(X_train) * 0.85)
    X_tr, X_va = X_train.iloc[:split_at], X_train.iloc[split_at:]
    y_tr, y_va = y_train.iloc[:split_at], y_train.iloc[split_at:]

    model = _build_model(params=params, random_state=random_state)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        callbacks=[early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model


def _train_single_target(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    param_grid: dict | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    search_strategy: str = "randomized",
    n_trials: int = 300,
    early_stopping_rounds: int = 100,
) -> dict:
    if param_grid is None:
        param_grid = get_lgbm_param_grid()

    if not feature_cols:
        raise ValueError(
            f"No feature columns available for target '{target_col}'. "
            "Check feature engineering and selected training features."
        )

    # LightGBM can handle NaNs, but we explicitly drop them as requested.
    model_df = df[feature_cols + [target_col]].dropna().copy()
    if model_df.empty:
        raise ValueError(
            f"No rows left after dropna for target '{target_col}'. "
            "Training was skipped to avoid returning empty results."
        )
    if len(model_df) <= n_splits + 1:
        raise ValueError(
            f"Not enough rows after dropna for target '{target_col}': "
            f"{len(model_df)} rows with n_splits={n_splits}."
        )

    X = model_df[feature_cols]
    y = model_df[target_col]

    X_train, X_test, y_train, y_test = _time_series_train_test_split(
        X, y, n_splits=n_splits
    )

    if search_strategy == "randomized":
        print(
            f"Starting randomized search for target='{target_col}' "
            f"with {n_trials} trials and {n_splits}-fold TimeSeriesSplit."
        )
        best_params, best_cv_rmse = _search_randomized(
            X=X_train,
            y=y_train,
            param_grid=param_grid,
            n_splits=n_splits,
            n_trials=n_trials,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
        )
    else:
        raise ValueError(
            "search_strategy must be 'randomized'."
        )

    best_model = _fit_final_model(
        X_train=X_train,
        y_train=y_train,
        params=best_params,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
    )

    y_pred = best_model.predict(X_test)
    metrics = evaluate_predictions(y_test.values, y_pred)

    return {
        "target": target_col,
        "feature_columns": feature_cols,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "best_params": best_params,
        "best_cv_rmse": best_cv_rmse,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": pd.Series(y_pred, index=y_test.index, name=f"{target_col}_pred"),
        "model": best_model,
        "search_strategy": search_strategy,
        "n_trials": n_trials,
    }


def train_demand_models(
    df_features: pd.DataFrame,
    param_grid: dict | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    search_strategy: str = "randomized",
    n_trials: int = 300,
    early_stopping_rounds: int = 100,
) -> dict:
    """
    Train two demand models:
    1) demand_log
    2) demand
    """
    feature_cols = get_demand_training_features(df_features)

    log_result = _train_single_target(
        df=df_features,
        target_col="demand_log",
        feature_cols=feature_cols,
        param_grid=param_grid,
        n_splits=n_splits,
        random_state=random_state,
        search_strategy=search_strategy,
        n_trials=n_trials,
        early_stopping_rounds=early_stopping_rounds,
    )

    demand_result = _train_single_target(
        df=df_features,
        target_col="demand",
        feature_cols=feature_cols,
        param_grid=param_grid,
        n_splits=n_splits,
        random_state=random_state,
        search_strategy=search_strategy,
        n_trials=n_trials,
        early_stopping_rounds=early_stopping_rounds,
    )

    metrics_table = pd.DataFrame(
        {
            "demand_log": log_result["metrics"],
            "demand": demand_result["metrics"],
        }
    ).T

    return {
        "feature_columns": feature_cols,
        "demand_log_model": log_result,
        "demand_model": demand_result,
        "metrics_table": metrics_table,
    }


def train_rrp_models(
    df_features: pd.DataFrame,
    param_grid: dict | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    search_strategy: str = "randomized",
    n_trials: int = 300,
    early_stopping_rounds: int = 100,
) -> dict:
    """
    Train two RRP models:
    1) RRP_log
    2) RRP
    """
    feature_cols = get_rrp_training_features(df_features)

    rrp_log_result = _train_single_target(
        df=df_features,
        target_col="RRP_log",
        feature_cols=feature_cols,
        param_grid=param_grid,
        n_splits=n_splits,
        random_state=random_state,
        search_strategy=search_strategy,
        n_trials=n_trials,
        early_stopping_rounds=early_stopping_rounds,
    )

    rrp_result = _train_single_target(
        df=df_features,
        target_col="RRP",
        feature_cols=feature_cols,
        param_grid=param_grid,
        n_splits=n_splits,
        random_state=random_state,
        search_strategy=search_strategy,
        n_trials=n_trials,
        early_stopping_rounds=early_stopping_rounds,
    )

    metrics_table = pd.DataFrame(
        {
            "RRP_log": rrp_log_result["metrics"],
            "RRP": rrp_result["metrics"],
        }
    ).T

    return {
        "feature_columns": feature_cols,
        "rrp_log_model": rrp_log_result,
        "rrp_model": rrp_result,
        "metrics_table": metrics_table,
    }
