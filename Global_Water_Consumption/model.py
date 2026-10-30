# ==========================================
# GLOBAL WATER CONSUMPTION PREDICTION
# Machine Learning Model
# ==========================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline


# ==========================================
# 1. LOAD DATASET
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(PROJECT_DIR / "cleaned_global_water_consumption.csv")

print("\n========== DATASET INFORMATION ==========")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 2. BASIC DATA CHECK
# ==========================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print("Duplicates:", df.duplicated().sum())


# ==========================================
# 3. DEFINE TARGET
# ==========================================

target = "Per Capita Water Use (Liters per Day)"

X = df.drop(columns=[target, "Water Scarcity Level"])
y = df[target]

print("\n========== TARGET ==========")
print("Target:", target)

print("\nTarget Statistics:")
print(y.describe())


# ==========================================
# 4. ENCODE CATEGORICAL FEATURES
# ==========================================

X = pd.get_dummies(
    X,
    columns=["Country"],
    drop_first=True
)

print("\n========== AFTER ENCODING ==========")
print("Number of features:", X.shape[1])


# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n========== TRAIN TEST SPLIT ==========")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==========================================
# 6. OUTLIER CAPPING USING IQR
# ==========================================

numeric_columns = X_train.select_dtypes(
    include=np.number
).columns

lower_limits = {}
upper_limits = {}

for column in numeric_columns:

    Q1 = X_train[column].quantile(0.25)
    Q3 = X_train[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    lower_limits[column] = lower
    upper_limits[column] = upper

    X_train[column] = X_train[column].clip(
        lower=lower,
        upper=upper
    )

    X_test[column] = X_test[column].clip(
        lower=lower,
        upper=upper
    )

print("\n========== OUTLIER HANDLING ==========")
print("IQR capping applied.")


# ==========================================
# 7. STANDARDIZATION
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 8. LINEAR REGRESSION
# ==========================================

linear_model = LinearRegression()

linear_model.fit(
    X_train_scaled,
    y_train
)

linear_pred = linear_model.predict(X_test_scaled)


# ==========================================
# 9. DECISION TREE REGRESSION
# ==========================================

decision_tree = DecisionTreeRegressor(
    random_state=42
)

decision_tree.fit(
    X_train,
    y_train
)

dt_pred = decision_tree.predict(X_test)


# ==========================================
# 10. RANDOM FOREST REGRESSION
# ==========================================

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest.fit(
    X_train,
    y_train
)

rf_pred = random_forest.predict(X_test)


# ==========================================
# 11. EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print("--------------------------------")
    print("MAE :", round(mae, 4))
    print("MSE :", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))

    return mae, mse, rmse, r2


# ==========================================
# 12. EVALUATE ALL MODELS
# ==========================================

lr_metrics = evaluate_model(
    "Linear Regression",
    y_test,
    linear_pred
)

dt_metrics = evaluate_model(
    "Decision Tree Regressor",
    y_test,
    dt_pred
)

rf_metrics = evaluate_model(
    "Random Forest Regressor",
    y_test,
    rf_pred
)


# ==========================================
# 13. MODEL COMPARISON TABLE
# ==========================================

comparison = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "MAE": [
        lr_metrics[0],
        dt_metrics[0],
        rf_metrics[0]
    ],

    "MSE": [
        lr_metrics[1],
        dt_metrics[1],
        rf_metrics[1]
    ],

    "RMSE": [
        lr_metrics[2],
        dt_metrics[2],
        rf_metrics[2]
    ],

    "R²": [
        lr_metrics[3],
        dt_metrics[3],
        rf_metrics[3]
    ]

})


print("\n========== MODEL COMPARISON ==========")
print(comparison.round(4))


# ==========================================
# 14. 5-FOLD CROSS VALIDATION
# ==========================================

print("\n========== 5-FOLD CROSS VALIDATION ==========")

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

linear_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

cv_scores = cross_val_score(
    linear_pipeline,
    X,
    y,
    cv=cv,
    scoring="r2"
)

print("5-Fold R² Scores:")
print(cv_scores)

print(
    "Mean Cross-Validation R²:",
    round(cv_scores.mean(), 4)
)


# ==========================================
# 15. ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    linear_pred,
    alpha=0.7
)

min_value = min(y_test.min(), linear_pred.min())
max_value = max(y_test.max(), linear_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel(
    "Actual Per Capita Water Use (Liters per Day)"
)

plt.ylabel(
    "Predicted Per Capita Water Use (Liters per Day)"
)

plt.title(
    "Actual vs Predicted - Linear Regression"
)

plt.show()


# ==========================================
# 16. RESIDUAL ANALYSIS
# ==========================================

residuals = y_test - linear_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    linear_pred,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Per Capita Water Use (Liters per Day)"
)

plt.ylabel(
    "Residual (Actual - Predicted)"
)

plt.title(
    "Residual Analysis - Linear Regression"
)

plt.show()


# ==========================================
# 17. FEATURE IMPORTANCE - RANDOM FOREST
# ==========================================

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": random_forest.feature_importances_

})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== TOP 10 IMPORTANT FEATURES ==========")

print(
    feature_importance.head(10)
)


# ==========================================
# 18. FEATURE IMPORTANCE GRAPH
# ==========================================

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title(
    "Top 10 Feature Importances - Random Forest"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()


# ==========================================
# END
# ==========================================

print("\n==========================================")
print("ML PROJECT EXECUTION COMPLETED")
print("==========================================")