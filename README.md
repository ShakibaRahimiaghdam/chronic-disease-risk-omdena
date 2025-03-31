# Omdena San Jose, USA Chapter: Cardiovascular Disease (CVD) Risk Prediction Using NHANES Data By Shakiba Rahimiaghdam


## 📌 Project Overview  
This project explores **chronic disease risk prediction** using the **NHANES dataset**, focusing on **Cardiovascular Disease (CVD) risk estimation**. The implementation includes **data preprocessing, feature selection, machine learning modeling, and a user-friendly interface for risk assessment**.



## 🛠 Key Features  
✔️ **Data Preprocessing & Imputation**: Missing values handled using advanced techniques like **IterativeImputer** with **ExtraTreesRegressor**.

✔️ **Feature Selection**: A hybrid approach combining **RandomForest, XGBoost, Mutual Information, and Recursive Feature Elimination (RFE)**.  

✔️ **Machine Learning Models**: Various classifiers including **Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, XGBoost, and ensemble models (Voting & Stacking classifiers)**.  

✔️ **Model Evaluation & Visualization**: Comprehensive metrics including **Accuracy, Precision, Recall, F1 Score, ROC AUC**, along with **ROC Curve and Feature Importance plots**.  

✔️ **User Interface (Streamlit)**: A **CVD risk estimation UI** where users input their medical data to get a **personalized risk score**.  



## 📂 Project Structure  
- 📁 `data/` – Dataset storage  
- 📁 `models/` – Saved models, scalers, selected features, evaluation metrics  
- 📄 `main.py` – Training pipeline for data preprocessing, feature selection & model training, Streamlit UI implementation  
- 📄 `requirements.txt` – Python dependencies  
- 📄 `README.md` – Project documentation  



## 🚀 How to Run the Project  

**1️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

**2️⃣ Train & Evaluate Models (For the first time)**
```bash
python main.py
```

**3️⃣ Run the Streamlit UI**
```bash
streamlit run main.py
```



## 🏆 Results & Insights
The best-performing model was selected based on ROC AUC scores and optimized for predictive accuracy.
Feature importance analysis highlighted key medical and demographic factors influencing CVD risk.
The user interface provides an interactive and accessible tool for risk estimation.



## 🔍 Considerations
The models and strategies were developed within a limited timeframe and serve as a baseline framework.
The implementation was done by a single contributor rather than a team effort.
Future work could include further optimization, additional features, or alternative modeling approaches.

