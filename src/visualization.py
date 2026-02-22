"""
Visualization functions for exploratory data analysis.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf


def _extract_series(
    df: pd.DataFrame, column: str, date_col: str = None, dropna: bool = True
) -> pd.Series:
    """
    Extract and optionally clean a series for lag-based analyses.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe.")

    if date_col and date_col in df.columns:
        series = pd.Series(df[column].values, index=pd.to_datetime(df[date_col]))
    else:
        series = df[column]

    if dropna:
        series = series.dropna()

    if len(series) < 3:
        raise ValueError(
            "Series is too short after cleaning. Provide more data points for ACF/PACF."
        )

    return series


def _get_significant_lags_table(
    values, confint, alpha: float, column_name: str, series_name: str
) -> pd.DataFrame:
    """
    Build and print a table of statistically significant lags.
    A lag is significant when zero lies outside its confidence interval.
    """
    ci_lower = confint[:, 0]
    ci_upper = confint[:, 1]

    rows = []
    for lag in range(len(values)):
        lower = float(ci_lower[lag])
        upper = float(ci_upper[lag])
        value = float(values[lag])

        if lower > 0:
            significance = "Positive (above zero)"
        elif upper < 0:
            significance = "Negative (below zero)"
        else:
            continue

        rows.append(
            {
                "Lag": lag,
                "Value": round(value, 6),
                "CI_Lower": round(lower, 6),
                "CI_Upper": round(upper, 6),
                "Significance": significance,
            }
        )

    significant_df = pd.DataFrame(rows)
    ci_pct = int((1 - alpha) * 100)
    print(f"\nSignificant {series_name} lags for '{column_name}' at CI {ci_pct}%:")
    if significant_df.empty:
        print("No significant lags found.")
    else:
        print(significant_df.to_string(index=False))

    return significant_df


def create_acf_plot(
    df: pd.DataFrame,
    column: str,
    date_col: str = None,
    nlags: int = 40,
    alpha: float = 0.05,
    extra_lags: list = None,
    title: str = None,
    dropna: bool = True,
) -> go.Figure:
    """
    Create an Autocorrelation Function (ACF) plot with confidence intervals.

    Parameters:
    -----------
    extra_lags : list, optional
        Specific lag values to highlight on top of standard lag output.
        Useful for checking domain-relevant lags (e.g., [7, 14, 21, 28, 365]).
    """
    series = _extract_series(df, column, date_col=date_col, dropna=dropna)

    max_lags = len(series) - 1
    if max_lags < 1:
        raise ValueError("Not enough observations to compute ACF.")

    base_nlags = int(max(1, nlags))
    valid_extra_lags = []
    if extra_lags:
        valid_extra_lags = sorted(
            {
                int(lag)
                for lag in extra_lags
                if isinstance(lag, (int, float)) and int(lag) >= 0
            }
        )

    requested_max_lag = (
        max([base_nlags] + valid_extra_lags) if valid_extra_lags else base_nlags
    )
    compute_nlags = min(requested_max_lag, max_lags)
    display_cutoff = min(base_nlags, compute_nlags)

    acf_vals, confint = acf(series, nlags=compute_nlags, alpha=alpha, fft=True)
    lags = list(range(len(acf_vals)))
    lags_to_plot = sorted(
        {lag for lag in lags if lag <= display_cutoff}
        | {lag for lag in valid_extra_lags if lag <= compute_nlags}
    )
    lag_to_pos = {lag: pos for pos, lag in enumerate(lags_to_plot)}

    fig = go.Figure()

    for lag in lags_to_plot:
        val = acf_vals[lag]
        x_pos = lag_to_pos[lag]
        fig.add_shape(
            type="line",
            x0=x_pos,
            x1=x_pos,
            y0=0,
            y1=float(val),
            line=dict(color="#1f77b4", width=2),
        )

    acf_custom_data = [[lag] for lag in lags_to_plot]
    fig.add_trace(
        go.Scatter(
            x=[lag_to_pos[lag] for lag in lags_to_plot],
            y=[acf_vals[lag] for lag in lags_to_plot],
            mode="markers",
            marker=dict(color="#1f77b4", size=8),
            name="ACF",
            customdata=acf_custom_data,
            hovertemplate="Lag %{customdata[0]}<br>ACF %{y:.4f}<extra></extra>",
        )
    )

    upper = confint[:, 1] - acf_vals
    lower = confint[:, 0] - acf_vals
    ci_custom_data = [[lag] for lag in lags_to_plot]
    fig.add_trace(
        go.Scatter(
            x=[lag_to_pos[lag] for lag in lags_to_plot],
            y=[upper[lag] for lag in lags_to_plot],
            mode="lines",
            line=dict(color="rgba(214, 39, 40, 0.7)", dash="dash"),
            name=f"Upper CI ({int((1-alpha)*100)}%)",
            customdata=ci_custom_data,
            hovertemplate="Lag %{customdata[0]}<br>Upper CI %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lag_to_pos[lag] for lag in lags_to_plot],
            y=[lower[lag] for lag in lags_to_plot],
            mode="lines",
            line=dict(color="rgba(214, 39, 40, 0.7)", dash="dash"),
            name=f"Lower CI ({int((1-alpha)*100)}%)",
            customdata=ci_custom_data,
            hovertemplate="Lag %{customdata[0]}<br>Lower CI %{y:.4f}<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.update_layout(
        title=title or f"Autocorrelation Function (ACF) - {column}",
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
        template="plotly_white",
        height=450,
        hovermode="x",
    )
    fig.update_xaxes(showticklabels=False, range=[-0.5, len(lags_to_plot) - 0.5])

    if valid_extra_lags:
        highlighted_lags = [lag for lag in valid_extra_lags if lag <= compute_nlags]
        if highlighted_lags:
            highlighted_values = [float(acf_vals[lag]) for lag in highlighted_lags]
            for lag in highlighted_lags:
                x_pos = lag_to_pos[lag]
                fig.add_vline(
                    x=x_pos,
                    line_width=1,
                    line_dash="dot",
                    line_color="rgba(255, 127, 14, 0.8)",
                )

            fig.add_trace(
                go.Scatter(
                    x=[lag_to_pos[lag] for lag in highlighted_lags],
                    y=highlighted_values,
                    mode="markers+text",
                    text=[f"Lag {lag}" for lag in highlighted_lags],
                    textposition="top center",
                    marker=dict(color="#ff7f0e", size=10, symbol="diamond"),
                    name="Extra lags",
                    customdata=[[lag] for lag in highlighted_lags],
                    hovertemplate="Lag %{customdata[0]}<br>ACF %{y:.4f}<extra></extra>",
                )
            )

    _get_significant_lags_table(
        values=acf_vals,
        confint=confint,
        alpha=alpha,
        column_name=column,
        series_name="ACF",
    )

    return fig


def create_pacf_plot(
    df: pd.DataFrame,
    column: str,
    date_col: str = None,
    nlags: int = 40,
    alpha: float = 0.05,
    extra_lags: list = None,
    method: str = "ywm",
    title: str = None,
    dropna: bool = True,
) -> go.Figure:
    """
    Create a Partial Autocorrelation Function (PACF) plot with confidence intervals.

    Parameters:
    -----------
    extra_lags : list, optional
        Specific lag values to highlight on top of standard lag output.
        Useful for checking domain-relevant lags (e.g., [7, 14, 21, 28, 365]).
    """
    series = _extract_series(df, column, date_col=date_col, dropna=dropna)

    max_lags = max(1, (len(series) // 2) - 1)
    base_nlags = int(max(1, nlags))
    valid_extra_lags = []
    if extra_lags:
        valid_extra_lags = sorted(
            {
                int(lag)
                for lag in extra_lags
                if isinstance(lag, (int, float)) and int(lag) >= 0
            }
        )

    requested_max_lag = (
        max([base_nlags] + valid_extra_lags) if valid_extra_lags else base_nlags
    )
    compute_nlags = min(requested_max_lag, max_lags)
    display_cutoff = min(base_nlags, compute_nlags)

    if compute_nlags < 1:
        raise ValueError("Not enough observations to compute PACF.")

    pacf_vals, confint = pacf(series, nlags=compute_nlags, alpha=alpha, method=method)
    lags = list(range(len(pacf_vals)))
    lags_to_plot = sorted(
        {lag for lag in lags if lag <= display_cutoff}
        | {lag for lag in valid_extra_lags if lag <= compute_nlags}
    )
    lag_to_pos = {lag: pos for pos, lag in enumerate(lags_to_plot)}

    fig = go.Figure()

    for lag in lags_to_plot:
        val = pacf_vals[lag]
        x_pos = lag_to_pos[lag]
        fig.add_shape(
            type="line",
            x0=x_pos,
            x1=x_pos,
            y0=0,
            y1=float(val),
            line=dict(color="#2ca02c", width=2),
        )

    pacf_custom_data = [[lag] for lag in lags_to_plot]
    fig.add_trace(
        go.Scatter(
            x=[lag_to_pos[lag] for lag in lags_to_plot],
            y=[pacf_vals[lag] for lag in lags_to_plot],
            mode="markers",
            marker=dict(color="#2ca02c", size=8),
            name="PACF",
            customdata=pacf_custom_data,
            hovertemplate="Lag %{customdata[0]}<br>PACF %{y:.4f}<extra></extra>",
        )
    )

    upper = confint[:, 1] - pacf_vals
    lower = confint[:, 0] - pacf_vals
    ci_custom_data = [[lag] for lag in lags_to_plot]
    fig.add_trace(
        go.Scatter(
            x=[lag_to_pos[lag] for lag in lags_to_plot],
            y=[upper[lag] for lag in lags_to_plot],
            mode="lines",
            line=dict(color="rgba(214, 39, 40, 0.7)", dash="dash"),
            name=f"Upper CI ({int((1-alpha)*100)}%)",
            customdata=ci_custom_data,
            hovertemplate="Lag %{customdata[0]}<br>Upper CI %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lag_to_pos[lag] for lag in lags_to_plot],
            y=[lower[lag] for lag in lags_to_plot],
            mode="lines",
            line=dict(color="rgba(214, 39, 40, 0.7)", dash="dash"),
            name=f"Lower CI ({int((1-alpha)*100)}%)",
            customdata=ci_custom_data,
            hovertemplate="Lag %{customdata[0]}<br>Lower CI %{y:.4f}<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.update_layout(
        title=title or f"Partial Autocorrelation Function (PACF) - {column}",
        xaxis_title="Lag",
        yaxis_title="Partial Autocorrelation",
        template="plotly_white",
        height=450,
        hovermode="x",
    )
    fig.update_xaxes(showticklabels=False, range=[-0.5, len(lags_to_plot) - 0.5])

    if valid_extra_lags:
        highlighted_lags = [lag for lag in valid_extra_lags if lag <= compute_nlags]
        if highlighted_lags:
            highlighted_values = [float(pacf_vals[lag]) for lag in highlighted_lags]
            for lag in highlighted_lags:
                x_pos = lag_to_pos[lag]
                fig.add_vline(
                    x=x_pos,
                    line_width=1,
                    line_dash="dot",
                    line_color="rgba(255, 127, 14, 0.8)",
                )

            fig.add_trace(
                go.Scatter(
                    x=[lag_to_pos[lag] for lag in highlighted_lags],
                    y=highlighted_values,
                    mode="markers+text",
                    text=[f"Lag {lag}" for lag in highlighted_lags],
                    textposition="top center",
                    marker=dict(color="#ff7f0e", size=10, symbol="diamond"),
                    name="Extra lags",
                    customdata=[[lag] for lag in highlighted_lags],
                    hovertemplate="Lag %{customdata[0]}<br>PACF %{y:.4f}<extra></extra>",
                )
            )

    _get_significant_lags_table(
        values=pacf_vals,
        confint=confint,
        alpha=alpha,
        column_name=column,
        series_name="PACF",
    )

    return fig


def create_time_series_plots(
    df: pd.DataFrame, time_col: str = None, numerical_cols: list = None
) -> go.Figure:
    """
    Create time series line plots for numerical columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    time_col : str, optional
        Column name for time/date. If None, uses index
    numerical_cols : list, optional
        List of numerical columns to plot. If None, plots all numerical columns

    Returns:
    --------
    go.Figure
        Plotly figure with time series plots
    """
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

    n_cols = min(len(numerical_cols), 3)
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=numerical_cols,
        specs=[[{"secondary_y": False}] * n_cols for _ in range(n_rows)],
    )

    for idx, col in enumerate(numerical_cols):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1

        if time_col and time_col in df.columns:
            x_data = df[time_col]
        else:
            x_data = df.index

        fig.add_trace(
            go.Scatter(x=x_data, y=df[col], mode="lines", name=col, line=dict(width=2)),
            row=row,
            col=col_pos,
        )

        fig.update_xaxes(
            title_text="Time" if time_col else "Index", row=row, col=col_pos
        )
        fig.update_yaxes(title_text=col, row=row, col=col_pos)

    fig.update_layout(
        title_text="Time Series Analysis",
        height=250 * n_rows,
        showlegend=True,
        hovermode="x unified",
    )

    return fig


def create_distribution_plots(
    df: pd.DataFrame, numerical_cols: list = None
) -> go.Figure:
    """
    Create histogram plots for distribution analysis of numerical columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numerical_cols : list, optional
        List of numerical columns to plot. If None, plots all numerical columns

    Returns:
    --------
    go.Figure
        Plotly figure with distribution plots
    """
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

    n_cols = min(len(numerical_cols), 3)
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=numerical_cols,
    )

    for idx, col in enumerate(numerical_cols):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1

        fig.add_trace(
            go.Histogram(x=df[col], name=col, nbinsx=30, showlegend=False),
            row=row,
            col=col_pos,
        )

        fig.update_xaxes(title_text=col, row=row, col=col_pos)
        fig.update_yaxes(title_text="Frequency", row=row, col=col_pos)

    fig.update_layout(
        title_text="Distribution Analysis",
        height=250 * n_rows,
        showlegend=False,
        hovermode="x",
    )

    return fig


def create_categorical_summary_plots(
    df: pd.DataFrame, categorical_cols: list = None
) -> dict:
    """
    Create visualizations for categorical variables.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_cols : list, optional
        List of categorical columns to plot. If None, plots all categorical columns

    Returns:
    --------
    dict
        Dictionary with figures for each categorical column
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    figures = {}

    for col in categorical_cols:
        value_counts = df[col].value_counts()

        fig = go.Figure(
            data=[
                go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    text=value_counts.values,
                    textposition="outside",
                )
            ]
        )

        fig.update_layout(
            title=f"Frequency Distribution - {col}",
            xaxis_title=col,
            yaxis_title="Frequency",
            height=400,
            hovermode="x",
        )

        figures[col] = fig

    return figures


def create_summary_statistics_table(
    df: pd.DataFrame, stats_df: pd.DataFrame
) -> go.Figure:
    """
    Create a table visualization of descriptive statistics.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    stats_df : pd.DataFrame
        Statistics dataframe from get_numerical_descriptive_stats

    Returns:
    --------
    go.Figure
        Plotly table figure
    """
    # Round numerical values for better display
    display_df = stats_df.copy()
    numeric_cols = display_df.select_dtypes(include=["number"]).columns
    display_df[numeric_cols] = display_df[numeric_cols].round(4)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(display_df.columns),
                    fill_color="paleturquoise",
                    align="left",
                    font=dict(color="black", size=12),
                ),
                cells=dict(
                    values=[display_df[col] for col in display_df.columns],
                    fill_color="lavender",
                    align="left",
                    font=dict(color="black", size=11),
                ),
            )
        ]
    )

    fig.update_layout(
        title="Descriptive Statistics Summary", height=400 + len(display_df) * 20
    )

    return fig


def plot_time_series_decomposition(
    decomposition_dict: dict, column_name: str
) -> go.Figure:
    """
    Create a plot of time series decomposition components.

    Parameters:
    -----------
    decomposition_dict : dict
        Dictionary from decompose_time_series function containing original, trend, seasonal, and residual
    column_name : str
        Name of the column being analyzed

    Returns:
    --------
    go.Figure
        Plotly figure with 4 subplots (original, trend, seasonal, residual)
    """
    original = decomposition_dict["original"]
    trend = decomposition_dict["trend"]
    seasonal = decomposition_dict["seasonal"]
    residual = decomposition_dict["residual"]

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=("Original", "Trend", "Seasonal", "Residual"),
        specs=[[{"secondary_y": False}] for _ in range(4)],
        vertical_spacing=0.08,
    )

    # Original series
    fig.add_trace(
        go.Scatter(
            x=original.index,
            y=original.values,
            mode="lines",
            name="Original",
            line=dict(color="blue", width=1),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Trend
    fig.add_trace(
        go.Scatter(
            x=trend.index,
            y=trend.values,
            mode="lines",
            name="Trend",
            line=dict(color="red", width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Seasonal
    if isinstance(seasonal, pd.DataFrame):
        for col in seasonal.columns:
            fig.add_trace(
                go.Scatter(
                    x=seasonal.index,
                    y=seasonal[col],
                    mode="lines",
                    name=col,
                    line=dict(width=1),
                    showlegend=True,
                ),
                row=3,
                col=1,
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=seasonal.index,
                y=seasonal.values,
                mode="lines",
                name="Seasonal",
                line=dict(color="green", width=1),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    # Residual
    fig.add_trace(
        go.Scatter(
            x=residual.index,
            y=residual.values,
            mode="lines",
            name="Residual",
            line=dict(color="orange", width=1),
            showlegend=False,
        ),
        row=4,
        col=1,
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Trend", row=2, col=1)
    fig.update_yaxes(title_text="Seasonal", row=3, col=1)
    fig.update_yaxes(title_text="Residual", row=4, col=1)

    fig.update_xaxes(title_text="Date", row=4, col=1)

    fig.update_layout(
        title_text=f"Time Series Decomposition (LOESS) - {column_name}",
        height=1000,
        hovermode="x unified",
        showlegend=False,
    )

    return fig


def plot_trend_and_components(decomposition_dict: dict, column_name: str) -> go.Figure:
    """
    Create a plot showing the original series with the trend overlay.

    Parameters:
    -----------
    decomposition_dict : dict
        Dictionary from decompose_time_series function
    column_name : str
        Name of the column being analyzed

    Returns:
    --------
    go.Figure
        Plotly figure with original and trend lines
    """
    original = decomposition_dict["original"]
    trend = decomposition_dict["trend"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=original.index,
            y=original.values,
            mode="lines",
            name="Original",
            line=dict(color="lightblue", width=1),
            opacity=0.7,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=trend.index,
            y=trend.values,
            mode="lines",
            name="Trend (LOESS)",
            line=dict(color="red", width=2),
        )
    )

    fig.update_layout(
        title=f"Original Series with Trend - {column_name}",
        xaxis_title="Date",
        yaxis_title="Value",
        height=400,
        hovermode="x unified",
        template="plotly_white",
    )

    return fig
