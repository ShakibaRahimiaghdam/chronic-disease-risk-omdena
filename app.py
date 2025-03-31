# Streamlit User Interface for Cardiovascular Risk Prediction
import streamlit as st
import pandas as pd
import joblib
import os

from config import MODEL_PATH, SCALER_PATH, FEATURES_PATH, LOGO_PATH
from src.data_loader import load_data_and_convert
from src.pipeline import train_pipeline_impute_first
from src.utils import input_hints, ui_categorical, numeric_ranges


def main():
    """
    Streamlit UI for CVD risk estimation.
    Loads the saved best model, scaler, and selected features.
    Displays a user-friendly form with a heart logo beside the title.
    Numeric inputs are presented with sliders (using predefined ranges), and categorical inputs are shown as Yes/No radio buttons.
    Upon submission, the inputs are processed, scaled, and the model outputs the estimated CVD risk probability.
    """
    # Use columns to place a logo beside the title.
    col1, col2 = st.columns([1, 4])
    with col1:
        # Ensure the logo image exists at the specified path.
        st.image(LOGO_PATH, width=80)
    with col2:
        st.title("Cardiovascular Disease (CVD) Risk Estimation")

    st.write("Enter your details below. The model will estimate your risk of cardiovascular disease.")

    # Load saved artifacts.
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH):
        best_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        selected_features = joblib.load(FEATURES_PATH)
    else:
        st.info("Training model... This may take a few minutes.")
        _, _, _, _, _, _, selected_features, scaler, _, _ = train_pipeline_impute_first()
        best_model = joblib.load(MODEL_PATH)

    # Load full data to determine numeric and categorical features.
    df_full = load_data_and_convert().drop(columns=["SEQN"], errors="ignore").reset_index(drop=True)
    categorical_feats = [col for col in df_full.columns if col != "CVD-Positive" and df_full[col].nunique() <= 10]
    numeric_feats = [col for col in df_full.columns if col not in categorical_feats + ["CVD-Positive"]]

    st.header("Enter Your Details")
    user_inputs = {}
    # Create an input for each selected feature.
    for feature in selected_features:
        label, hint = input_hints.get(feature, (feature, ""))
        # For categorical features: use radio buttons.
        if feature in ui_categorical:
            choice = st.radio(f"{label} (e.g., {hint})", ["Yes", "No"])
            user_inputs[feature] = 1 if choice == "Yes" else 2
        # For numeric features: use a slider if a range is defined; else use number input.
        elif feature in numeric_feats:
            if feature in numeric_ranges:
                min_val, max_val, default_val = numeric_ranges[feature]
                user_inputs[feature] = st.slider(f"{label} (e.g., {hint})", float(min_val), float(max_val), float(default_val))
            else:
                user_inputs[feature] = st.number_input(f"{label} (e.g., {hint})", value=0.0, format="%.2f")
        else:
            user_inputs[feature] = st.text_input(f"{label} (e.g., {hint})", value="")

    if st.button("Estimate CVD Risk"):
        input_df = pd.DataFrame([user_inputs])
        try:
            input_numeric = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)
            scaled_array = scaler.transform(input_numeric)
            scaled_df = pd.DataFrame(scaled_array, columns=scaler.feature_names_in_)
            input_df.update(scaled_df[[col for col in selected_features if col in numeric_feats]])
            probability = best_model.predict_proba(input_df[selected_features])[:, 1][0] * 100
            st.success(f"Estimated CVD Risk Probability: {probability:.2f}%")
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
