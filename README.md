# Global Water Consumption Prediction

A Streamlit application and machine-learning workflow for exploring global water-consumption data and predicting **Per Capita Water Use (Liters per Day)**.

## Project overview

The project includes two interactive dashboards, a notebook for exploratory analysis, scripts for model training and analysis, two synchronized copies of the cleaned dataset, and a pre-trained model artifact. The dashboards let users explore the dataset, compare model results, and generate predictions for supported countries.

## Problem statement

Water use varies with location, year, and environmental and consumption factors. This project applies supervised regression to estimate per-capita water use from those features and compares three models. It is an educational analysis tool; predictions should not be treated as operational or policy guidance.

## Dataset

The bundled CSV contains **501 records and 10 columns** across **20 countries**. The two copies, `cleaned_global_water_consumption.csv` and `Global_Water_Consumption/cleaned_global_water_consumption.csv`, are synchronized.

Columns:

- `Country`
- `Year`
- `Total Water Consumption (Billion Cubic Meters)`
- `Per Capita Water Use (Liters per Day)` — prediction target
- `Agricultural Water Use (%)`
- `Industrial Water Use (%)`
- `Household Water Use (%)`
- `Rainfall Impact (Annual Precipitation in mm)`
- `Groundwater Depletion Rate (%)`
- `Water Scarcity Level`

`Water Scarcity Level` is excluded from model features. The dataset files are included so training and the dashboards work from a clone without external data downloads. Dataset provenance and redistribution terms were not present in the project files; confirm the source and license before publishing.

## Preprocessing and training

`Global_Water_Consumption/train_models.py`:

1. Uses an 80/20 train/test split with `random_state=42`.
2. One-hot encodes `Country`.
3. Calculates IQR outlier bounds from the training split and caps the selected numeric input features in both splits.
4. Fits `StandardScaler` on the training features and applies it to both splits.
5. Trains the three regressors and calculates MAE, MSE, RMSE, and R² from their test-set predictions.
6. Saves the trained estimators, preprocessing values, feature order, and evaluation metrics to `Global_Water_Consumption/water_models.pkl`.

The project uses the same target variable and regression approaches throughout the training workflow; the saved artifact supplies the dashboards with the evaluation metrics.

## Algorithms

- **Linear Regression**
- **Decision Tree Regressor** (`random_state=42`)
- **Random Forest Regressor** (`n_estimators=100`, `random_state=42`)

All three models use the scaled feature matrix in the training script.

## Evaluation results

These are the current saved test-split metrics, rounded to four decimal places. They are generated during training; the dashboards read them from the model artifact rather than embedding metric values in source code.

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 35.4157 | 2121.7897 | 46.0629 | -0.1235 |
| Decision Tree | 49.2987 | 4135.7887 | 64.3101 | -1.1900 |
| Random Forest | 34.5844 | 2113.1591 | 45.9691 | -0.1190 |

All three R² scores are negative on this split, meaning these models perform worse than a constant mean-target baseline on the test data. Treat the results as a baseline for further data and model improvements, not as evidence of strong predictive performance.

## Requirements

Python **3.11** is the tested runtime. Direct dependencies and versions used to validate the project are listed in [`requirements.txt`](requirements.txt). The pre-trained model was generated with the pinned scikit-learn version; matching the listed environment is recommended when loading it.

## Installation and execution

Run commands from the repository root.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run a dashboard

From the repository root:

```bash
# WaterSense dashboard
streamlit run app_dashboard.py

# Alternative dashboards
streamlit run Global_Water_Consumption/app_dashboard.py
streamlit run Global_Water_Consumption/app.py
```

The scripts resolve their data and model files relative to their own source files, so they do not depend on the shell's current directory. The root dashboard uses the bundled model in `Global_Water_Consumption/` when no root-level model exists.

### Retrain the models

```bash
python Global_Water_Consumption/train_models.py
```

This command reads the nested CSV and recreates `Global_Water_Consumption/water_models.pkl`.

### Run the analysis script

```bash
python Global_Water_Consumption/model.py
```

### Open the notebook (optional)

The notebook is at the repository root and reads the root CSV. Open the repository root as the notebook workspace so its relative dataset path resolves. In an environment with Jupyter installed:

```bash
jupyter notebook Untitled.ipynb
```

The notebook checkpoint directory is excluded from Git; the original notebook remains included.

## Repository structure

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── Untitled.ipynb
├── app_dashboard.py
├── cleaned_global_water_consumption.csv
└── Global_Water_Consumption/
    ├── app.py
    ├── app_dashboard.py
    ├── cleaned_global_water_consumption.csv
    ├── model.py
    ├── train_models.py
    └── water_models.pkl
```

Python bytecode, local environments, editor settings, Streamlit secrets, and `.ipynb_checkpoints/` are excluded by `.gitignore`. The datasets, notebook, source code, and trained model artifact are not ignored and are intended to be included in the repository.

## Security and publishing notes

No likely API keys, passwords, private-key blocks, or credential-bearing URLs were detected in the project files during the publishing scan. Keep real credentials out of source control; Streamlit secrets and `.env` files are ignored. Review dataset provenance and license before making the repository public.
