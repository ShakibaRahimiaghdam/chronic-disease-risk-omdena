# -----------------------------
#   Fixed Test Sample Generation Function
# -----------------------------
import pandas as pd

# Shared variables for UI
input_hints = {
    "RIDAGEYR": ("Age (years)?", "Range: 0 - 100+"),
    "BPQ020": ("Ever Told: High Blood Pressure?", "Select Yes or No"),
    "BPQ090D": ("Angina?", "Select Yes or No"),
    "BPQ040A": ("High Cholesterol?", "Select Yes or No"),
    "OSQ230": ("Metal Objects in Body?",
               "Do you have any artificial joints, pins, plates, metal suture material, or other metal objects?"),
    "BPQ050A": ("Taking Cholesterol Meds?", "Select Yes or No"),
    "LBXTC": ("Total Cholesterol (mg/dL)?", "Range: 100 - 300 mg/dL"),
    "MCQ160A": ("Arthritis?", "Select Yes or No"),
    "MCQ366C": ("Hip Swelling?", "Select Yes or No"),
    "MCQ160P": ("Psoriatic Arthritis?", "Select Yes or No"),
    "LBDLDLM": ("LDL-Cholesterol (mg/dL)?", "Range: 0 - 300 mg/dL"),
    "BPQ030": ("Taking BP Medication?", "Select Yes or No"),
    "MCQ520": ("Osteoporosis?", "Select Yes or No"),
    "SMQ020": ("Smoking-Related Disease Diagnosis?", "Have you ever been told you have a smoking-related disease?"),
    "INDFMPIR": ("Poverty Income Ratio?", "Range: 0 - 5+"),
    "BMXWAIST": ("Waist Circumference (cm)?", "Range: 60 - 150 cm")
}

ui_categorical = ["BPQ020", "BPQ090D", "BPQ040A", "OSQ230", "BPQ050A", "MCQ160A", "MCQ366C", "MCQ160P", "BPQ030",
                  "MCQ520", "SMQ020"]

numeric_ranges = {
    "RIDAGEYR": (0, 100, 50),
    "LBXTC": (100, 300, 200),
    "LBDLDLM": (0, 300, 150),
    "INDFMPIR": (0, 5, 2.5),
    "BMXWAIST": (60, 150, 100)
}


def generate_fixed_test_sample(selected_features, numeric_features, categorical_features, scaler, df,
                               risk_profile='high'):
    """
    Generate a fixed test sample for a high-risk profile.
    For numeric features, use the median; for categorical, use the mode of positive cases.
    Reindex the DataFrame to match the scaler's expected feature names.

    Returns a DataFrame containing one test sample.
    """
    if risk_profile == 'high' and "CVD-Positive" in df.columns:
        df_risk = df[df["CVD-Positive"] == 1]
        if df_risk.empty:
            df_risk = df
    else:
        df_risk = df

    fixed_sample = {}
    for feature in numeric_features:
        fixed_sample[feature] = df_risk[feature].median()
    for feature in categorical_features:
        fixed_sample[feature] = df_risk[feature].mode()[0]

    sample_df = pd.DataFrame([fixed_sample])
    sample_numeric = sample_df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    scaled_array = scaler.transform(sample_numeric)
    scaled_df = pd.DataFrame(scaled_array, columns=scaler.feature_names_in_)
    sample_df.update(scaled_df[numeric_features])
    sample_df_final = sample_df[selected_features]
    return sample_df_final
