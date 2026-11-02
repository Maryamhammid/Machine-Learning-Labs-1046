import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from xgboost import XGBClassifier

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Shape:")
print(df.shape)
print("\nFirst 10 Rows:")
print(df.head(10))
print("\nMissing Values:")
print(df.isnull().sum())
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
print("\nMissing TotalCharges:")
print(df["TotalCharges"].isnull().sum())

median_value = df["TotalCharges"].median()

df["TotalCharges"] = df["TotalCharges"].fillna(median_value)

print("\nMissing values after filling:")
print(df["TotalCharges"].isnull().sum())

df.drop("customerID", axis=1, inplace=True)

# Encode target variable
# Yes = 1, No = 0
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

X = df.drop("Churn", axis=1)
y = df["Churn"]

# One-hot encode categorical columns
X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)
print("\nEncoded Dataset Shape:")
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Gradient Boosting
gb_model = GradientBoostingClassifier(
    random_state=42
)
# AdaBoost
ada_model = AdaBoostClassifier(
    random_state=42
)
# XGBoost
xgb_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)
gb_model.fit(X_train, y_train)
ada_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

print("\nDefault boosting models trained successfully!")

# 10. Tune XGBoost using GridSearchCV
xgb_tuning = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0]
}
grid_search = GridSearchCV(
    estimator=xgb_tuning,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print("\nBest Parameters:")
print(grid_search.best_params_)
print("\nBest Cross-Validation F1 Score:")
print(grid_search.best_score_)

# Get tuned XGBoost model
best_xgb = grid_search.best_estimator_

models = {
    "Default XGBoost": xgb_model,
    "Tuned XGBoost": best_xgb
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    # Probability of positive class (Churn = 1)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_prob)
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

results_df = pd.DataFrame(results)
print("\n========== XGBOOST COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nClassification Report - Tuned XGBoost:")
tuned_predictions = best_xgb.predict(X_test)

print(
    classification_report(
        y_test,
        tuned_predictions,
        target_names=["No Churn", "Churn"]
    )
)

importance = best_xgb.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importance
})
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)
top_five = feature_importance.head(5)
print("\nTop Five Important Features:")
print(top_five.to_string(index=False))
plt.figure(figsize=(9, 5))
plt.barh(
    top_five["Feature"],
    top_five["Importance"]
)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 5 Important Features - Tuned XGBoost")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()