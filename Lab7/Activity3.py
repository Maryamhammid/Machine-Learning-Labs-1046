import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
df = pd.read_csv("insurance.csv")

print("Dataset Shape:")
print(df.shape)
print("\nFirst 10 Rows:")
print(df.head(10))
print("\nMissing Values:")
print(df.isnull().sum())
print("\nSummary Statistics of Charges:")
print(df["charges"].describe())

# Encode sex: female = 0, male = 1
df["sex"] = df["sex"].map({
    "female": 0,
    "male": 1
})
# Encode smoker: no = 0, yes = 1
df["smoker"] = df["smoker"].map({
    "no": 0,
    "yes": 1
})
# One-hot encode region
df = pd.get_dummies(
    df,
    columns=["region"],
    drop_first=True,
    dtype=int
)
print("\nDataset After Encoding:")
print(df.head())

X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Ridge Regression with scaling
ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", Ridge(alpha=1.0))
])
# Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(
    random_state=42
)
ridge_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
print("\nIndividual models trained successfully!")

# Build Stacking Regressor
stacking_model = StackingRegressor(
    estimators=[
        ("ridge", ridge_model),
        ("random_forest", rf_model),
        ("gradient_boosting", gb_model)
    ],
    final_estimator=LinearRegression(),
    cv=5
)
# Train stacking model
stacking_model.fit(X_train, y_train)
print("\nStacking model trained successfully!")

models = {
    "Ridge Regression": ridge_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "Stacking Regressor": stacking_model
}
results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })
results_df = pd.DataFrame(results)
print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

best_model_name = results_df.loc[
    results_df["RMSE"].idxmin(), "Model"
]

print("\nBest Model Based on RMSE:", best_model_name)

final_model = stacking_model.final_estimator_
print("\nFinal Estimator Coefficients:")
base_model_names = [
    "Ridge",
    "Random Forest",
    "Gradient Boosting"
]
for name, coef in zip(
    base_model_names,
    final_model.coef_
):
    print(name, ":", coef)
print("\nFinal Estimator Intercept:")
print(final_model.intercept_)

stack_predictions = stacking_model.predict(X_test)
plt.figure(figsize=(8, 6))
plt.scatter(
    y_test,
    stack_predictions,
    alpha=0.6
)
minimum = min(y_test.min(), stack_predictions.min())
maximum = max(y_test.max(), stack_predictions.max())
plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)
plt.xlabel("Actual Medical Charges")
plt.ylabel("Predicted Medical Charges")
plt.title("Actual vs Predicted Charges - Stacking Regressor")
plt.tight_layout()
plt.show()

coefficients = pd.Series(
    final_model.coef_,
    index=base_model_names
)
print("\nMeta-model Coefficients:")
print(coefficients)
most_trusted_model = coefficients.abs().idxmax()
print(
    "\nBase model with the largest absolute coefficient:",
    most_trusted_model
)