import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(PROJECT_DIR / "cleaned_global_water_consumption.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ==========================================
# 2. DEFINE TARGET
# ==========================================

target = "Per Capita Water Use (Liters per Day)"

# Remove target and Water Scarcity Level
X = df.drop(columns=[target, "Water Scarcity Level"])
y = df[target]


# ==========================================
# 3. ONE-HOT ENCODE COUNTRY
# ==========================================

X = pd.get_dummies(X, columns=["Country"], dtype=float)

feature_columns = X.columns.tolist()

print("Number of features after encoding:", len(feature_columns))


# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 5. IQR OUTLIER CAPPING
# ==========================================

numeric_columns = [
    "Year",
    "Total Water Consumption (Billion Cubic Meters)",
    "Agricultural Water Use (%)",
    "Industrial Water Use (%)",
    "Household Water Use (%)",
    "Rainfall Impact (Annual Precipitation in mm)",
    "Groundwater Depletion Rate (%)"
]

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

    X_train[column] = X_train[column].clip(lower, upper)
    X_test[column] = X_test[column].clip(lower, upper)


print("IQR capping applied.")


# ==========================================
# 6. STANDARD SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 7. TRAIN MODELS
# ==========================================

linear_model = LinearRegression()

decision_tree_model = DecisionTreeRegressor(
    random_state=42
)

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


linear_model.fit(X_train_scaled, y_train)

decision_tree_model.fit(X_train_scaled, y_train)

random_forest_model.fit(X_train_scaled, y_train)

model_metrics = {}
for name, model in {
    "linear_model": linear_model,
    "decision_tree_model": decision_tree_model,
    "random_forest_model": random_forest_model
}.items():
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    model_metrics[name] = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "MSE": float(mse),
        "RMSE": float(mse ** 0.5),
        "R²": float(r2_score(y_test, predictions))
    }


# ==========================================
# 8. SAVE EVERYTHING
# ==========================================

model_data = {

    "linear_model": linear_model,

    "decision_tree_model": decision_tree_model,

    "random_forest_model": random_forest_model,

    "scaler": scaler,

    "feature_columns": feature_columns,

    "numeric_columns": numeric_columns,

    "lower_limits": lower_limits,

    "upper_limits": upper_limits,

    "metrics": model_metrics
}


joblib.dump(model_data, PROJECT_DIR / "water_models.pkl")


print()
print("==========================================")
print("MODELS SAVED SUCCESSFULLY!")
print("==========================================")
print("File created: water_models.pkl")
print()
print("Models:")
print("1. Linear Regression")
print("2. Decision Tree Regressor")
print("3. Random Forest Regressor")
print()
print("Test-set metrics:")
for model_name, metrics in model_metrics.items():
    print(model_name, {name: round(value, 4) for name, value in metrics.items()})