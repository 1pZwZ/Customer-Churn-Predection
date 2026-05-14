# Customer-Churn-Predection
# 📊 Telco Customer Churn Prediction: End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-orange?style=for-the-badge&logo=xgboost)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Predictive_Analytics-green?style=for-the-badge)

## 🚀 Live Application
Experience the interactive web app deployed on Streamlit Community Cloud:
👉 **[Click Here to Try the Churn Predictor](https://customers-churn-predict.streamlit.app/)**

## 📖 Project Overview
Customer attrition (churn) is one of the most critical challenges in the telecommunications industry. Acquiring a new customer is often much more expensive than retaining an existing one. 

This project aims to predict the likelihood of a customer leaving the service based on their demographic data, account information, and subscribed services. By accurately predicting churn, businesses can take proactive measures—such as offering targeted discounts or upgrading contracts—to retain high-value customers.

## 🧠 Key Business Insights (EDA)
During the Exploratory Data Analysis (EDA) phase, several critical business patterns emerged:
- **Contract Type is Crucial:** Customers on a **Month-to-month** contract are significantly more likely to churn due to the lack of commitment, compared to those on 1-year or 2-year contracts.
- **The Value of Bundled Services:** Customers lacking bundled security features (Online Backup, Device Protection, Tech Support) show a much higher churn rate. 

## ⚙️ Technical Pipeline & Approach
1. **Data Cleaning:** Handled hidden missing values within numerical columns (e.g., parsing empty strings in `TotalCharges`).
2. **Feature Engineering:** - **Dimensionality Reduction:** Consolidated redundant "No internet service" columns to reduce multicollinearity.
   - **Feature Creation:** Engineered a new aggregate feature, `Total_Security_Services`, to capture overall customer engagement and loyalty.
3. **Modeling & Evaluation:**
   - Evaluated baseline models, including **Random Forest**.
   - Selected **XGBoost** for its superior performance and capability with tabular data.
4. **Hyperparameter Tuning:** Utilized `GridSearchCV` to fine-tune XGBoost parameters (`learning_rate`, `max_depth`, `n_estimators`), successfully improving the model's accuracy to **~80%**.
5. **Deployment:** Developed a user-friendly, interactive UI using **Streamlit** to serve the trained machine learning model in real-time.

## 💻 How to Run Locally
If you wish to run this project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/Customer-Churn-Prediction.git](https://github.com/YOUR_GITHUB_USERNAME/Customer-Churn-Prediction.git)
   cd Customer-Churn-Prediction
   pip install -r requirements.txt
   streamlit run app.py

├── EDA.ipynb               # Jupyter notebook with Data Analysis & Model Training
├── app.py                  # Streamlit web application code
├── requirements.txt        # Python dependencies
├── tuned_xgb_model.pkl     # Saved XGBoost model (Tuned)
├── model_columns.pkl       # Saved column structure for the model
└── README.md               # Project documentation
