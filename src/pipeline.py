# -----------------------------
#   Pipeline 1: Impute First Training Pipeline
# -----------------------------
from src.data_loader import load_data_and_convert
from src.preprocessing import impute_and_scale_data
from src.feature_selection import advanced_feature_selection_data
from src.evaluation import evaluate_unified_models, plot_comparison, plot_model_roc_curve
from config import MODEL_PATH, SCALER_PATH, FEATURES_PATH, METRICS_CSV_PATH, RESULTS_TABLE_PATH, ROC_CURVE_PATH

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb


def train_pipeline_impute_first():
    """
    Pipeline 1: Impute missing values first, perform feature selection,
    and train unified models.

    Steps:
      1. Load and convert data.
      2. Identify numeric and categorical features.
      3. Impute missing values and scale numeric features.
      4. Perform advanced feature selection (select top 15 features).
      5. Split data (stratified) and apply SMOTE.
      6. Train tuned models and ensembles.
      7. Evaluate models, save metrics and plots.
      8. Save the best model, scaler, and selected features.
    """
    df_full = load_data_and_convert().drop(columns=["SEQN"], errors="ignore").reset_index(drop=True)

    # Identify numeric and categorical features.
    categorical_feats = [col for col in df_full.columns if col != "CVD-Positive" and df_full[col].nunique() <= 10]
    numeric_feats = [col for col in df_full.columns if col not in categorical_feats + ["CVD-Positive"]]

    # Impute and scale.
    df_imputed, scaler = impute_and_scale_data(df_full, numeric_feats, categorical_feats)

    # Advanced feature selection.
    X_imp = df_imputed.drop(columns=["CVD-Positive"]).copy()
    y_imp = df_imputed["CVD-Positive"]
    selected_features, _ = advanced_feature_selection_data(X_imp, y_imp, top_k=15)
    joblib.dump(selected_features, FEATURES_PATH)

    # Prepare final training data.
    X_final = df_imputed[selected_features].copy()
    y_final = df_imputed["CVD-Positive"]

    # Split data and apply SMOTE.
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42, stratify=y_final)
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f"[Impute First] After SMOTE, training set distribution: {np.bincount(y_train)}")

    # Define tuned unified models.
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.1, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, eval_metric="logloss", random_state=42,
                                     scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train))
    }

    # Create ensemble models.
    voting_model = VotingClassifier(estimators=[
        ("lr", models["Logistic Regression"]),
        ("rf", models["Random Forest"]),
        ("gb", models["Gradient Boosting"]),
        ("ada", models["AdaBoost"]),
        ("xgb", models["XGBoost"])
    ], voting="soft")

    stacking_model = StackingClassifier(
        estimators=[
            ("lr", models["Logistic Regression"]),
            ("rf", models["Random Forest"]),
            ("gb", models["Gradient Boosting"]),
            ("ada", models["AdaBoost"]),
            ("xgb", models["XGBoost"])
        ],
        final_estimator=LogisticRegression(max_iter=2000, random_state=42),
        cv=5
    )

    models["Voting Ensemble"] = voting_model
    models["Stacking Ensemble"] = stacking_model

    # Evaluate models.
    results_df = evaluate_unified_models(models, X_train, X_test, y_train, y_test)
    plot_comparison(results_df, RESULTS_TABLE_PATH)
    results_df.to_csv(METRICS_CSV_PATH)
    print("[Impute First] Unified Model evaluation results:\n", results_df)

    # Select and save the best model.
    best_model_name = results_df["ROC AUC"].idxmax()
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    plot_model_roc_curve(best_model, X_test, y_test, ROC_CURVE_PATH)
    joblib.dump(best_model, MODEL_PATH)
    print(f"✅ Final unified model '{best_model_name}' trained and saved at {MODEL_PATH}")

    return results_df, models, X_train, X_test, y_train, y_test, selected_features, scaler, numeric_feats, categorical_feats
