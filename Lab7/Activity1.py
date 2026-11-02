import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

df = pd.read_csv("water_potability.csv")

print("Dataset Shape:")
print(df.shape)
print("\nFirst 10 Rows:")
print(df.head(10))
print("\nMissing Values:")
print(df.isnull().sum())
print("\nClass Distribution:")
print(df["Potability"].value_counts())

X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X.columns,
    index=X_train.index
)
X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X.columns,
    index=X_test.index
)
print("\nMissing values after imputation:")
print(X_train.isnull().sum().sum())
print(X_test.isnull().sum().sum())


# Logistic Regression with scaling
lr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# SVC with scaling
svc_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(probability=True, random_state=42))
])
# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
# Single Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)

#Create Voting Classifiers

# Hard Voting
hard_voting = VotingClassifier(
    estimators=[
        ("lr", lr_model),
        ("svc", svc_model),
        ("rf", rf_model)
    ],
    voting="hard"
)

# Soft Voting
soft_voting = VotingClassifier(
    estimators=[
        ("lr", lr_model),
        ("svc", svc_model),
        ("rf", rf_model)
    ],
    voting="soft"
)
# Create Bagging Classifiers

bagging_10 = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=10,
    max_samples=0.5,
    bootstrap=True,
    random_state=42
)
# Bagging with 100 estimators and bootstrap=True
bagging_100 = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    max_samples=0.5,
    bootstrap=True,
    random_state=42
)
# Pasting with 100 estimators and bootstrap=False
pasting_100 = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    max_samples=0.5,
    bootstrap=False,
    random_state=42
)
models = {
    "Logistic Regression": lr_model,
    "SVC": svc_model,
    "Random Forest": rf_model,
    "Single Decision Tree": dt_model,
    "Hard Voting": hard_voting,
    "Soft Voting": soft_voting,
    "Bagging (10 trees)": bagging_10,
    "Bagging (100 trees)": bagging_100,
    "Pasting (100 trees)": pasting_100
}

results = []

for name, model in models.items():

    print("\nTraining:", name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
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
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1-Score",
    ascending=False
)
print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
print("\nBest Model:", best_model_name)
best_predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

print("\nConfusion Matrix:")
print(cm)
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Potable (0)", "Potable (1)"]
).plot(cmap="Blues")
plt.title("Confusion Matrix - " + best_model_name)
plt.show()

single_models = [
    "Logistic Regression",
    "SVC",
    "Random Forest",
    "Single Decision Tree"
]

ensemble_models = [
    "Hard Voting",
    "Soft Voting",
    "Bagging (10 trees)",
    "Bagging (100 trees)",
    "Pasting (100 trees)"
]

single_best = results_df[
    results_df["Model"].isin(single_models)
].iloc[0]
ensemble_best = results_df[
    results_df["Model"].isin(ensemble_models)
].iloc[0]
print("\nBest Single Model:")
print(single_best)
print("\nBest Ensemble Model:")
print(ensemble_best)

if ensemble_best["F1-Score"] > single_best["F1-Score"]:
    print("\nConclusion:")
    print("Voting or Bagging improved performance over")
    print("the best individual model based on F1-score.")
else:
    print("\nConclusion:")
    print("Voting and Bagging did not improve performance")
    print("over the best individual model based on F1-score.")
