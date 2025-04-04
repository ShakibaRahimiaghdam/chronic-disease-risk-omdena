<p align="center">
  <img src="assets/Omdena_logo_1.png" alt="Omdena Logo" width="250"/>
</p>

# <img src="assets/Omdena_logo_2.png" alt="Omdena Logo" width="30" style="vertical-align:middle; margin-right:8px;"/> Omdena San Jose, USA, Local Chapter: [Exploring the Chronic Disease Risk Using NHANES Data in U.S.](https://www.omdena.com/chapter-challenges/exploring-the-chronic-disease-risk-using-nhanes-data-in-us)

> Developed and implemented by **Shakiba Rahimiaghdam** as part of the **Omdena San Jose, USA, Local Chapter** challenge, covering all core components: data preprocessing, imbalanced data handling, feature selection, ML model training, and the Streamlit UI.

<br>

## 📌 Project Overview  

This project explores **chronic disease risk prediction** using the **NHANES dataset**, with a focus on **Cardiovascular Disease (CVD) risk estimation**. It involves **data preprocessing, feature selection, machine learning modeling**, and a **user-friendly interface** for personalized risk assessment.

<br>

## 🛠 Key Features  

✔️ **Data Preprocessing & Imputation**: Missing values are handled using advanced techniques like **IterativeImputer** with **ExtraTreesRegressor**.

✔️ **Imbalanced Data Handling**: Applied **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic samples and balance the CVD classes.  

✔️ **Feature Selection**: Hybrid approach using **RandomForest, XGBoost, Mutual Information**, and **Recursive Feature Elimination (RFE)**.  

✔️ **Machine Learning Models**: Trained with **Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, XGBoost**, and **ensemble models** (Voting & Stacking classifiers).  

✔️ **Model Evaluation & Visualization**: Comprehensive evaluation with **Accuracy, Precision, Recall, F1 Score, ROC AUC**, along with **ROC Curve** and **Feature Importance** plots.  

✔️ **Interactive UI (Streamlit)**: Enter your medical info and receive a **personalized CVD risk score** instantly.

<br>

## 📂 Project Structure  
<pre>
CVD_Risk_Estimation/
│
├── data/                  # Raw and cleaned datasets
├── models/                # Trained models, scalers, and feature sets (ignored in Git)
├── assets/                # UI images/icons (e.g., heart logo)
├── src/                   # Modular Python code
│     ├── data_loader.py     # Data loading functions
│     ├── preprocessing.py   # Imputation and scaling
│     ├── feature_selection.py  # Feature selection logic
│     ├── evaluation.py      # Model metrics and visualizations
│     ├── pipeline.py        # Training pipeline
│     └── utils.py           # Helper functions
│
├── app.py                 # Streamlit app UI
├── config.py              # Central file paths and shared configs
├── requirements.txt       # Dependencies
├── .gitignore             # Ignored files
└── README.md              # Project documentation
</pre>
  
<br>

## 🚀 Getting Started 

**1️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

**2️⃣ Train & Evaluate Models (First time only)**
If you haven't saved a model yet, this step will run the full pipeline:
```bash
python -c "from src.pipeline import train_pipeline_impute_first; train_pipeline_impute_first()"
```

**3️⃣ Run the Streamlit App**
```bash
streamlit run app.py
```

<br>

## 🏆 Results & Insights

- The **best-performing model** was selected based on **ROC AUC** scores and optimized for predictive accuracy.<br>
- **Feature importance analysis** highlighted key demographic and medical factors influencing CVD risk.<br>
- The **interactive UI** allows users to estimate risk quickly and intuitively.<br>

<br>

## 🔍 Considerations

- This project was developed under time constraints and serves as a **baseline framework**.<br>
- Future improvements could include model fine-tuning, additional clinical features, or deployment options (e.g., Hugging Face Spaces or Docker).

<br>

## 👩‍💻 Author & Contributions

This project was developed by **Shakiba Rahimiaghdam** as part of the **Omdena San Jose Local Chapter**.

**Key contributions:**<br>
- Designed and implemented the full data preprocessing and machine learning pipeline.<br>
- Applied hybrid feature selection and ensemble learning techniques.<br>
- Developed a fixed test sample generator and applied SMOTE for balancing imbalanced CVD classes.<br>
- Built the full Streamlit UI for interactive CVD risk estimation.<br>
- Modularized the codebase for clarity, reusability, and scalability.<br>

<br>

> 📜 This project was developed during an Omdena Local Chapter challenge. All code is shared under the GPL-3 license with due credit to Omdena and the San Jose Local Chapter.
