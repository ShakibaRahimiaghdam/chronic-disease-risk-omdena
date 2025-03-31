# -----------------------------
#   Advanced Feature Selection Function
# -----------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif, RFE
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from config import FEATURE_IMPORTANCE_PATH


def advanced_feature_selection_data(X, y, top_k=15):
    """
    Select the top_k features using ensemble ranking based on:
      - RandomForest importances,
      - XGBoost importances,
      - Mutual information scores,
      - RFE with Logistic Regression.
    Saves a feature importance plot.

    Returns:
      selected_features: List of selected feature names.
      feature_importance_df: DataFrame with feature importance values.
    """
    # RandomForest importances.
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X, y)
    rf_importances = pd.Series(rf.feature_importances_, index=X.columns)

    # XGBoost importances.
    xgb_clf = xgb.XGBClassifier(eval_metric="logloss", random_state=42,
                                scale_pos_weight=(len(y) - sum(y)) / sum(y))
    xgb_clf.fit(X, y)
    xgb_importances = pd.Series(xgb_clf.feature_importances_, index=X.columns)

    # Mutual information scores.
    mi_scores = pd.Series(mutual_info_classif(X, y, discrete_features="auto"), index=X.columns)

    # RFE with Logistic Regression.
    lr = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")
    rfe = RFE(estimator=lr, n_features_to_select=top_k)
    rfe.fit(X, y)
    rfe_ranking = pd.Series(rfe.ranking_, index=X.columns)

    total_score = (rf_importances.rank(ascending=False) +
                   xgb_importances.rank(ascending=False) +
                   mi_scores.rank(ascending=False) +
                   rfe_ranking)

    selected_features = total_score.sort_values().head(top_k).index.tolist()

    avg_importance = (rf_importances[selected_features] + xgb_importances[selected_features]) / 2
    feature_importance_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": avg_importance.values
    })
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df.sort_values(by="Importance", ascending=True))
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH)
    plt.close()
    print(f"📊 Feature importance plot saved at {FEATURE_IMPORTANCE_PATH}")

    return selected_features, feature_importance_df
