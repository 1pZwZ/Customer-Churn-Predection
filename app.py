import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Load the model and columns
@st.cache_resource # Cache the model to speed up the app
def load_assets():
    model = joblib.load('tuned_xgb_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

model, model_columns = load_assets()

# --- SIDEBAR INPUTS ---
st.sidebar.header("👤 Customer Profile")
st.sidebar.info("Adjust customer attributes below to see the prediction.")

with st.sidebar:
    st.subheader("Billing Details")
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)
    
    st.divider()
    
    st.subheader("Services & Contract")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No Internet"])
    total_security = st.select_slider("Total Security Services", options=[0, 1, 2, 3, 4], value=1)

# --- MAIN AREA ---
st.title("🧠 Customer Churn Intelligence Portal")
st.markdown("Analyze customer behavior and predict attrition risk using Advanced Machine Learning (XGBoost).")

# Dividing the main area into Tabs
tab1, tab2 = st.tabs(["🚀 Risk Predictor", "📊 Model Insights"])

with tab1:
    st.subheader("Real-time Prediction Analysis")
    
    # Process inputs
    input_dict = {col: 0 for col in model_columns}
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['TotalCharges'] = total_charges
    input_dict['Total_Security_Services'] = total_security
    
    if contract == "One year": input_dict['Contract_One year'] = 1
    elif contract == "Two year": input_dict['Contract_Two year'] = 1
        
    if internet_service == "Fiber optic": input_dict['InternetService_Fiber optic'] = 1
    elif internet_service == "No Internet": input_dict['No_Internet_Service'] = 1
    
    input_df = pd.DataFrame([input_dict])[model_columns]
    
    # Action Button
    if st.button("Analyze Risk Profile"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Displaying Results in a nice layout
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label="Churn Probability", value=f"{probability:.1%}")
        
        with res_col2:
            status = "🚨 HIGH RISK" if prediction == 1 else "✅ LOW RISK"
            st.metric(label="Customer Status", value=status)
            
        st.divider()
        
        if prediction == 1:
            st.error("### ⚠️ Retention Alert")
            st.write(f"The model indicates a **{probability:.1%}** chance that this customer will leave. **Immediate intervention suggested.**")
            st.expander("See Recommended Actions").write("""
                - Offer a 15% loyalty discount on the next 6 months.
                - Invite the customer to upgrade to a 'One year' contract.
                - Check for any recent technical support tickets.
            """)
        else:
            st.success("### ✅ Safe Customer")
            st.write(f"This customer is likely to stay. Retention probability is **{1-probability:.1%}**.")

with tab2:
    st.subheader("How the model works?")
    st.write("This model uses an **XGBoost Classifier** optimized via **GridSearchCV**.")
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info("**Key Features Used:**\n- Contract Type\n- Monthly Charges\n- Security Service Bundle\n- Customer Tenure")
    with col_info2:
        st.success("**Model Performance:**\n- Accuracy: ~80%\n- Balanced Precision & Recall")
        
    st.markdown("---")
    st.caption("Developed by AI Engineering Student @ Imam University")