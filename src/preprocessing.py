# -----------------------------
#   Improved Imputation and Scaling Function
# -----------------------------
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
import pandas as pd
import joblib
from config import SCALER_PATH


def impute_and_scale_data(df, numeric_feats, categorical_feats):
    """
    Impute missing values:
      - Numeric: IterativeImputer with ExtraTreesRegressor.
      - Categorical: SimpleImputer (most frequent).
    Then scale numeric features with RobustScaler.

    Returns:
      df_imputed: DataFrame with imputed and scaled numeric features and imputed categorical features.
      scaler: Fitted RobustScaler.
    """
    # Impute numeric features.
    imp_iter = IterativeImputer(random_state=42, estimator=ExtraTreesRegressor(n_estimators=50, random_state=42))
    df_numeric = pd.DataFrame(imp_iter.fit_transform(df[numeric_feats]), columns=numeric_feats)

    # Impute categorical features.
    imp_cat = SimpleImputer(strategy="most_frequent")
    df_categorical = pd.DataFrame(imp_cat.fit_transform(df[categorical_feats]), columns=categorical_feats)

    # Combine numeric, categorical, and target.
    df_imputed = pd.concat([df_numeric, df_categorical, df["CVD-Positive"]], axis=1)

    # Scale numeric features.
    scaler = RobustScaler()
    df_imputed[numeric_feats] = scaler.fit_transform(df_imputed[numeric_feats])
    joblib.dump(scaler, SCALER_PATH)

    return df_imputed, scaler
