import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from src.data_loader import load_data
from src.feature_engineering import create_features
from sklearn.model_selection import train_test_split

# -----------------------------
# Create images directory
# -----------------------------
os.makedirs("images", exist_ok=True)

# -----------------------------
# Load model
# -----------------------------
pipeline = joblib.load(
    "models/xgboost_pipeline.pkl"
)

# -----------------------------
# Load data
# -----------------------------
df = load_data("data/german_credit_data.csv")

df = create_features(df)

X = df.drop("Risk", axis=1)

y = df["Risk"].map({
    "bad": 0,
    "good": 1
})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Predictions
# -----------------------------
predictions = pipeline.predict(X_test)

probabilities = pipeline.predict_proba(X_test)[:, 1]

# -----------------------------
# Metrics
# -----------------------------
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "images/confusion_matrix_xgboost.png",
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(
    y_test,
    probabilities
)

plt.figure(figsize=(6, 5))

plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.savefig(
    "images/roc_curve_xgboost.png",
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# Precision-Recall Curve
# -----------------------------
precision, recall, _ = precision_recall_curve(
    y_test,
    probabilities
)

plt.figure(figsize=(6, 5))

plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.savefig(
    "images/precision_recall_curve_xgboost.png",
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# Feature Importance
# -----------------------------
model = pipeline.named_steps["model"]

importances = model.feature_importances_

plt.figure(figsize=(10, 6))

plt.bar(
    range(len(importances)),
    importances
)

plt.title("Feature Importance")

plt.savefig(
    "images/feature_importance_xgboost.png",
    bbox_inches="tight"
)

plt.close()

print("Evaluation completed.")
print("Plots saved in images/")