"""
Visualization functions for exploratory data analysis.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def create_time_series_plots(df: pd.DataFrame, time_col: str = None, numerical_cols: list = None) -> go.Figure:
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
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Limit to first 6 columns if too many
    if len(numerical_cols) > 6:
        numerical_cols = numerical_cols[:6]
    
    n_cols = min(len(numerical_cols), 3)
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=numerical_cols,
        specs=[[{"secondary_y": False}] * n_cols for _ in range(n_rows)]
    )
    
    for idx, col in enumerate(numerical_cols):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1
        
        if time_col and time_col in df.columns:
            x_data = df[time_col]
        else:
            x_data = df.index
        
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=df[col],
                mode='lines',
                name=col,
                line=dict(width=2)
            ),
            row=row, col=col_pos
        )
        
        fig.update_xaxes(title_text="Time" if time_col else "Index", row=row, col=col_pos)
        fig.update_yaxes(title_text=col, row=row, col=col_pos)
    
    fig.update_layout(
        title_text="Time Series Analysis",
        height=250 * n_rows,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def create_distribution_plots(df: pd.DataFrame, numerical_cols: list = None) -> go.Figure:
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
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Limit to first 6 columns if too many
    if len(numerical_cols) > 6:
        numerical_cols = numerical_cols[:6]
    
    n_cols = min(len(numerical_cols), 3)
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=numerical_cols,
    )
    
    for idx, col in enumerate(numerical_cols):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1
        
        fig.add_trace(
            go.Histogram(
                x=df[col],
                name=col,
                nbinsx=30,
                showlegend=False
            ),
            row=row, col=col_pos
        )
        
        fig.update_xaxes(title_text=col, row=row, col=col_pos)
        fig.update_yaxes(title_text="Frequency", row=row, col=col_pos)
    
    fig.update_layout(
        title_text="Distribution Analysis",
        height=250 * n_rows,
        showlegend=False,
        hovermode='x'
    )
    
    return fig


def create_categorical_summary_plots(df: pd.DataFrame, categorical_cols: list = None) -> dict:
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
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    figures = {}
    
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        
        fig = go.Figure(
            data=[go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                text=value_counts.values,
                textposition='outside'
            )]
        )
        
        fig.update_layout(
            title=f"Frequency Distribution - {col}",
            xaxis_title=col,
            yaxis_title="Frequency",
            height=400,
            hovermode='x'
        )
        
        figures[col] = fig
    
    return figures


def create_summary_statistics_table(df: pd.DataFrame, stats_df: pd.DataFrame) -> go.Figure:
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
    numeric_cols = display_df.select_dtypes(include=['number']).columns
    display_df[numeric_cols] = display_df[numeric_cols].round(4)
    
    fig = go.Figure(
        data=[go.Table(
            header=dict(
                values=list(display_df.columns),
                fill_color='paleturquoise',
                align='left',
                font=dict(color='black', size=12)
            ),
            cells=dict(
                values=[display_df[col] for col in display_df.columns],
                fill_color='lavender',
                align='left',
                font=dict(color='black', size=11)
            )
        )]
    )
    
    fig.update_layout(
        title="Descriptive Statistics Summary",
        height=400 + len(display_df) * 20
    )
    
    return fig
