# Streamlit User Interface for Cardiovascular Risk Prediction
import streamlit as st
import pandas as pd
import joblib
import os
import base64

from config import MODEL_PATH, SCALER_PATH, FEATURES_PATH, LOGO_PATH, Omdena_LOGO_2_PATH
from src.data_loader import load_data_and_convert
from src.pipeline import train_pipeline_impute_first
from src.utils import input_hints, ui_categorical, numeric_ranges


def get_image_base64(image_path):
    """Convert an image file to a base64 string for HTML embedding"""
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def main():
    """
    Streamlit UI for CVD risk estimation.
    Loads the saved best model, scaler, and selected features.
    Displays a user-friendly form with a heart logo beside the title.
    Numeric inputs are presented with sliders (using predefined ranges), and categorical inputs are shown as Yes/No radio buttons.
    Upon submission, the inputs are processed, scaled, and the model outputs the estimated CVD risk probability.
    """
    logo_base64 = get_image_base64(Omdena_LOGO_2_PATH)  # Adjust path if needed

    html_block = (
        f'<div style="display: flex; flex-direction: row; align-items: flex-start; '
        f'background-color: #f9f9f9; padding: 20px; border-radius: 10px; '
        f'border: 1px solid #ddd; font-family: Segoe UI, sans-serif; '
        f'font-size: 17px; line-height: 1.6;">'

        f'<div style="flex: 0 0 auto; padding-right: 20px;">'
        f'<img src="data:image/png;base64,{logo_base64}" width="140"/>'
        f'</div>'

        f'<div style="flex: 1;">'
        f'<h3 style="margin-top: 0; color:#0056C1;">Omdena Local Chapter Project</h3>'
        f'<p>This user interface is part of the <strong>Omdena San Jose, USA, Local Chapter</strong> project titled '
        f'<a href="https://www.omdena.com/chapter-challenges/exploring-the-chronic-disease-risk-using-nhanes-data-in-us" '
        f'target="_blank" style="text-decoration: none; color:#1a73e8;">'
        f'<em>“Exploring the Chronic Disease Risk Using NHANES Data in the U.S.”</em></a>, '
        f'implemented by <strong>Shakiba Rahimiaghdam</strong>.</p>'

        f'<p>The project developed a machine learning-based CVD risk estimation tool using '
        f'<strong>demographic</strong>, <strong>clinical</strong>, and <strong>lifestyle</strong> data from the NHANES dataset (2017–2020). '
        f'It empowers users to receive real-time personalized predictions, making advanced health insights more accessible.</p>'
        f'</div>'
        f'</div>'
    )

    st.markdown(html_block, unsafe_allow_html=True)

    # Now add the heart logo + main title
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(LOGO_PATH, width=120)
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
