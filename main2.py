import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Page configuration
st.set_page_config(
    page_title="Insurance Charge Predictor",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("🏥 Insurance Charge Prediction App")
st.write("Predict your insurance charges based on personal information")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Charges", "Model Performance"])

# Load and train model (you'll do this once)
@st.cache_resource
def load_model():
    # Load your data here
    # df = pd.read_csv('insurance.csv')
    
    # For demo, I'll create a simple placeholder
    # Replace this with your actual model training code
    st.info("Note: Replace this section with your actual data loading and model training")
    
    # Placeholder - replace with your actual trained model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return model

# HOME PAGE
if page == "Home":
    st.header("Welcome to the Insurance Predictor")
    
    st.write("""
    ### How it works:
    1. Enter your personal information
    2. Our machine learning model predicts your insurance charges
    3. See which factors influence your premium the most
    
    ### About the model:
    - Uses Random Forest algorithm
    - Trained on 1,338 insurance records
    - Accuracy: 86% (R² = 0.86)
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy", "86%", "R² Score")
    with col2:
        st.metric("Average Error", "$2,559", "MAE")
    with col3:
        st.metric("Training Records", "1,338", "samples")

# PREDICTION PAGE
elif page == "Predict Charges":
    st.header("💵 Predict Your Insurance Charges")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        
        age = st.slider("Age", min_value=18, max_value=100, value=30, step=1)
        
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        
        children = st.selectbox("Number of Children", options=[0, 1, 2, 3, 4, 5], index=0)
        
    with col2:
        st.subheader("Additional Details")
        
        sex = st.radio("Sex", options=["Male", "Female"])
        sex_male = 1 if sex == "Male" else 0
        
        smoker = st.radio("Do you smoke?", options=["No", "Yes"])
        smoker_encoded = 1 if smoker == "Yes" else 0
        
        region = st.selectbox("Region", options=["Northeast", "Northwest", "Southeast", "Southwest"])
        
    # Encode region
    region_northeast = 1 if region == "Northeast" else 0
    region_northwest = 1 if region == "Northwest" else 0
    region_southeast = 1 if region == "Southeast" else 0
    region_southwest = 1 if region == "Southwest" else 0
    
    # Predict button
    if st.button("🔮 Predict Insurance Charges", type="primary"):
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker_encoded],
            'sex_male': [sex_male],
            'region_northeast': [region_northeast],
            'region_northwest': [region_northwest],
            'region_southeast': [region_southeast],
            'region_southwest': [region_southwest]
        })
        
        # Make prediction (placeholder calculation)
        # Replace this with your actual model prediction
        # prediction = model.predict(input_data)[0]
        
        # Placeholder prediction formula (replace with actual model)
        base_charge = 3000
        age_factor = age * 100
        bmi_factor = (bmi - 25) * 50 if bmi > 25 else 0
        smoker_factor = 15000 if smoker_encoded == 1 else 0
        children_factor = children * 500
        
        prediction = base_charge + age_factor + bmi_factor + smoker_factor + children_factor
        
        # Display results
        st.success("Prediction Complete!")
        
        st.markdown(f"### Estimated Annual Insurance Charge: ${prediction:,.2f}")
        
        # Show breakdown
        st.subheader("Cost Breakdown:")
        breakdown_col1, breakdown_col2 = st.columns(2)
        
        with breakdown_col1:
            st.write(f"**Base charge:** ${base_charge:,.2f}")
            st.write(f"**Age factor:** ${age_factor:,.2f}")
            st.write(f"**BMI factor:** ${bmi_factor:,.2f}")
        
        with breakdown_col2:
            st.write(f"**Smoking factor:** ${smoker_factor:,.2f}")
            st.write(f"**Children factor:** ${children_factor:,.2f}")
        
        # Show important factors
        st.info("💡 **Tip:** Smoking status has the biggest impact on insurance charges!")

# MODEL PERFORMANCE PAGE
elif page == "Model Performance":
    st.header("📊 Model Performance Metrics")
    
    st.write("### Training Results")
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("Training RMSE", "$1,908.57")
        st.metric("Test RMSE", "$4,633.36")
    
    with metrics_col2:
        st.metric("Training R²", "0.9748")
        st.metric("Test R²", "0.8617")
    
    with metrics_col3:
        st.metric("Test MAE", "$2,559.04")
    
    st.write("---")
    
    st.write("### Feature Importance")
    st.write("Factors that influence insurance charges the most:")
    
    # Sample feature importance data
    importance_data = {
        'Feature': ['Smoker', 'Age', 'BMI', 'Children', 'Region Northeast', 'Sex Male', 'Region Northwest', 'Region Southeast', 'Region Southwest'],
        'Importance': [0.5234, 0.1523, 0.0891, 0.0756, 0.0421, 0.0389, 0.0312, 0.0289, 0.0185]
    }
    
    importance_df = pd.DataFrame(importance_data)
    
    st.bar_chart(importance_df.set_index('Feature'))
    
    st.dataframe(importance_df, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit 🎈")





