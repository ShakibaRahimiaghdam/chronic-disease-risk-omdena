# -----------------------------
#   Evaluation Functions
# -----------------------------
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def evaluate_unified_models(models, X_train, X_test, y_train, y_test):
    """
    Train each model and evaluate on the test set.
    Returns a DataFrame with Accuracy, Precision, Recall, F1 Score, ROC AUC, and CV ROC AUC Mean.
    """
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        results[name] = {
            "Accuracy": model.score(X_test, y_test),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
            "CV ROC AUC Mean": np.mean(cv_scores)
        }
    return pd.DataFrame(results).T


def plot_comparison(results_df, output_path):
    """Plot a bar chart comparing model metrics and save it."""
    plt.figure(figsize=(10, 6))
    results_df[["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "CV ROC AUC Mean"]].plot(kind="bar")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Model comparison plot saved at {output_path}")


def plot_model_roc_curve(model, X_test, y_test, output_path):
    """Plot the ROC curve for the given model and save it."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📊 ROC curve plot saved at {output_path}")
