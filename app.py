import streamlit as st
import pandas as pd
import joblib

# 1. Load the model and columns
model = joblib.load('tuned_xgb_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# 2. App UI Configuration
st.set_page_config(page_title="Telco Churn Prediction", page_icon="📊")
st.title("📊 Telco Customer Churn Prediction")
st.markdown("Predict if a customer will leave the service based on their profile.")

# 3. User Inputs (Creating a clean layout)
col1, col2 = st.columns(2)

with col1:
    st.header("Billing Info")
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0)

with col2:
    st.header("Services & Contract")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No Internet"])
    
    # Our engineered feature!
    total_security = st.slider("Total Security Services (0 to 4)", min_value=0, max_value=4, value=1, help="Count of: Online Security, Backup, Device Protection, Tech Support")

# 4. Prediction Logic
if st.button("Predict Churn 🚀"):
    # Initialize a dictionary with zeros for all required columns
    input_dict = {col: 0 for col in model_columns}
    
    # Map user inputs to the dictionary
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['TotalCharges'] = total_charges
    input_dict['Total_Security_Services'] = total_security
    
    # Map categorical inputs
    if contract == "One year":
        input_dict['Contract_One year'] = 1
    elif contract == "Two year":
        input_dict['Contract_Two year'] = 1
        
    if internet_service == "Fiber optic":
        input_dict['InternetService_Fiber optic'] = 1
    elif internet_service == "No Internet":
        input_dict['No_Internet_Service'] = 1  # Using your awesome engineered feature
        
    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Ensure the order matches the training data exactly
    input_df = input_df[model_columns]
    
    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    # 5. Display Results
    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn! (Probability: {probability:.1%})")
        st.write("💡 Business Suggestion: Offer a discount or upgrade to a longer contract to keep them.")
    else:
        st.success(f"✅ Safe. Customer is likely to stay. (Probability of leaving: {probability:.1%})")