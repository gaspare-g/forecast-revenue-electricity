# Forecasts of electricity demand in Victoria - Australia
This project aims to forecast revenues coming from electricity market in Victoria, Australia using data from Kaggle. By applying machine learning models, the goal is to capture seasonal and daily patterns, improve prediction accuracy, and support efficient investment planning.

## Analysis Documentation - Big Picture
The data covers the state district of Victoria - Australia from beginning 1st of January 2015 until 6th of October 2020. I forecast from 7th October 2020 to 31st December 2021.   
The project follows an end-to-end analytics and forecasting workflow:

1. Load and preprocess daily electricity data for Victoria, including demand, price (RRP), and weather-related features.
2. Run descriptive analysis to assess completeness, missing values, and variable behavior.
3. Perform time-series decomposition to separate trend/seasonality/residual components for both demand and RRP.
4. Build engineered features from calendar effects, weather, and trend components.
5. Train forecasting models on log-transformed targets to improve stability for long-horizon prediction.
6. Generate probabilistic forecasts (P10, P50, P90) for demand and RRP, then combine them into revenue scenarios.
7. Aggregate outcomes at quarterly level to support planning under pessimistic, expected, and optimistic scenarios.

Notebook narrative sections include:
- Descriptive analysis and missing-data treatment.
- Decomposition analysis for demand and RRP seasonality.
- Forecast evaluation comments.
- Revenue scenario construction and quarter-level summary.

## Installation and Environment Setup (uv)
Install `uv`:

- Windows (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
- macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate the virtual environment:

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
uv venv
source .venv/bin/activate
```

Install project dependencies:

```bash
uv sync
```

## Notebook Viewer
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://nbviewer.org/github/gaspare-g/forecast-revenue-electricity/blob/main/analysis.ipynb)

[Click here to see the Jupyter notebook.](https://nbviewer.org/github/gaspare-g/forecast-revenue-electricity/blob/main/analysis.ipynb)

## Data Access Note
A Kaggle API key is needed to download the dataset directly from Kaggle. If no key is available, the repository dataset (`complete_dataset.csv`) is used as fallback.

## Final Quarterly Scenario Table
Units:
- `total_demand_million_MWh`: million MWh (scaled to millions, 4 decimals)
- `expected_revenue_million_AUD`: million AUD (4 decimals)

| quarter | scenario | total_demand_million_MWh | average_rrp_AUD_MWh | expected_revenue_million_AUD |
|---|---|---:|---:|---:|
| 2020Q4 | Pessimistic | 9.5720 | 29.9252 | 290.1612 |
| 2020Q4 | Expected | 9.7426 | 40.8541 | 401.1519 |
| 2020Q4 | Optimistic | 9.9071 | 52.9740 | 528.2677 |
| 2021Q1 | Pessimistic | 9.6071 | 38.4441 | 374.4468 |
| 2021Q1 | Expected | 9.9369 | 51.8785 | 520.0144 |
| 2021Q1 | Optimistic | 10.3196 | 55.6201 | 578.4831 |
| 2021Q2 | Pessimistic | 10.5103 | 34.5473 | 371.2948 |
| 2021Q2 | Expected | 10.7545 | 45.8654 | 499.8290 |
| 2021Q2 | Optimistic | 10.9220 | 62.0734 | 692.5044 |
| 2021Q3 | Pessimistic | 11.0501 | 44.6026 | 500.7112 |
| 2021Q3 | Expected | 11.2551 | 57.4185 | 657.3090 |
| 2021Q3 | Optimistic | 11.4370 | 67.3675 | 783.7723 |
| 2021Q4 | Pessimistic | 9.5757 | 30.3347 | 292.8484 |
| 2021Q4 | Expected | 9.7598 | 41.9039 | 410.4880 |
| 2021Q4 | Optimistic | 9.9144 | 54.7599 | 544.2795 |

