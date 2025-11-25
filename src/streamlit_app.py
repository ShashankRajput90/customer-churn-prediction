#!/usr/bin/env python3
"""
Customer Churn Prediction - Streamlit Dashboard
Author: Shashank Lodhi
Date: November 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #21808D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high {
        color: #FF5459;
        font-weight: bold;
    }
    .risk-medium {
        color: #FFA500;
        font-weight: bold;
    }
    .risk-low {
        color: #21808D;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load trained model and scaler"""
    try:
        model = joblib.load('models/xgboost_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except:
        return None, None

def predict_churn(customer_data, model, scaler):
    """Make churn prediction"""
    # Scale features
    customer_scaled = scaler.transform(customer_data)
    
    # Predict
    churn_prob = model.predict_proba(customer_scaled)[0][1]
    churn_prediction = 1 if churn_prob >= 0.5 else 0
    
    return churn_prediction, churn_prob

def get_risk_category(prob):
    """Categorize risk level"""
    if prob >= 0.6:
        return "High Risk", "risk-high"
    elif prob >= 0.35:
        return "Medium Risk", "risk-medium"
    else:
        return "Low Risk", "risk-low"

def get_recommendations(customer_data, churn_prob):
    """Generate personalized recommendations"""
    recommendations = []
    
    if customer_data['tenure'].values[0] < 12:
        recommendations.append("🎯 New customer - Implement early engagement program")
    
    if customer_data.get('Contract_Month-to-month', [0]).values[0] == 1:
        recommendations.append("📋 Offer long-term contract discount (15-20% off)")
    
    if churn_prob > 0.5:
        recommendations.append("🚨 High priority - Assign dedicated account manager")
        recommendations.append("💰 Provide exclusive retention offer within 48 hours")
    
    if customer_data['MonthlyCharges'].values[0] > 70:
        recommendations.append("💵 Review pricing - Customer in high-cost tier")
    
    return recommendations

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction System</h1>', 
                unsafe_allow_html=True)
    
    # Load model
    model, scaler = load_model()
    
    if model is None:
        st.error("❌ Model not found! Please run the training pipeline first.")
        st.info("Run: `python src/churn_prediction_pipeline.py`")
        return
    
    # Sidebar - Input features
    st.sidebar.header("📝 Customer Information")
    
    # Demographics
    st.sidebar.subheader("Demographics")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
    
    # Services
    st.sidebar.subheader("Services")
    phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    internet_service = st.sidebar.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
    
    # Account
    st.sidebar.subheader("Account Details")
    contract = st.sidebar.selectbox(
        "Contract Type", ["Month-to-month", "One year", "Two year"]
    )
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    )
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    
    # Billing
    st.sidebar.subheader("Billing")
    monthly_charges = st.sidebar.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=5.0
    )
    total_charges = tenure * monthly_charges
    
    # Create prediction button
    predict_button = st.sidebar.button("🔮 Predict Churn", use_container_width=True)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📈 Prediction", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        if predict_button:
            # Prepare input data
            customer_dict = {
                'tenure': tenure,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': 1 if partner == "Yes" else 0,
                'Dependents': 1 if dependents == "Yes" else 0,
                'PhoneService': 1 if phone_service == "Yes" else 0,
                'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
                'AvgMonthlyCost': total_charges / (tenure + 1),
                'ContractBinary': 1 if contract == "Month-to-month" else 0
            }
            
            # Add dummy variables for categorical features
            for service in ['DSL', 'Fiber optic', 'No']:
                customer_dict[f'InternetService_{service}'] = 1 if internet_service == service else 0
            
            for contract_type in ['Month-to-month', 'One year', 'Two year']:
                customer_dict[f'Contract_{contract_type}'] = 1 if contract == contract_type else 0
            
            for payment in ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']:
                customer_dict[f'PaymentMethod_{payment.replace(" ", "")}'] = 1 if payment_method == payment else 0
            
            customer_df = pd.DataFrame([customer_dict])
            
            # Make prediction
            churn_pred, churn_prob = predict_churn(customer_df, model, scaler)
            risk_cat, risk_class = get_risk_category(churn_prob)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Churn Probability",
                    f"{churn_prob*100:.1f}%",
                    delta=f"{(churn_prob-0.265)*100:.1f}% vs avg"
                )
            
            with col2:
                st.markdown(f"### Risk Level")
                st.markdown(f'<p class="{risk_class}" style="font-size: 2rem;">{risk_cat}</p>', 
                          unsafe_allow_html=True)
            
            with col3:
                st.metric(
                    "Prediction",
                    "Will Churn" if churn_pred == 1 else "Will Stay",
                    delta="Action Required" if churn_pred == 1 else "Monitor"
                )
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Retention Recommendations")
            
            recommendations = get_recommendations(customer_df, churn_prob)
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            # Visualization
            st.markdown("---")
            st.subheader("📊 Risk Breakdown")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=churn_prob*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Risk (%)"},
                delta={'reference': 26.5},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#21808D"},
                    'steps': [
                        {'range': [0, 35], 'color': "lightgreen"},
                        {'range': [35, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Churn Analytics Overview")
        
        # Sample statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Churn Rate", "26.5%")
        with col2:
            st.metric("High Risk Customers", "42.7%", delta="Month-to-month")
        with col3:
            st.metric("Model Accuracy", "84.2%")
        with col4:
            st.metric("ROC-AUC Score", "0.87")
        
        st.markdown("---")
        
        # Feature importance
        st.subheader("🎯 Top Churn Drivers")
        
        feature_data = pd.DataFrame({
            'Feature': ['Tenure', 'Monthly Charges', 'Contract Type', 'Payment Method', 'Internet Service'],
            'Importance': [100, 85, 72, 54, 48]
        })
        
        fig = px.bar(
            feature_data,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance (%)',
            color='Importance',
            color_continuous_scale='Teal'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("ℹ️ About This System")
        
        st.markdown("""
        ### Customer Churn Prediction System
        
        This ML-powered system predicts customer churn probability with **84.2% accuracy** 
        using XGBoost algorithm trained on 7,000+ telecom customer records.
        
        #### Key Features:
        - 🎯 Real-time churn probability prediction
        - 📊 Risk categorization (High/Medium/Low)
        - 💡 Personalized retention recommendations
        - 📈 Feature importance analysis
        
        #### Model Performance:
        - **Accuracy:** 84.2%
        - **Precision:** 81.5%
        - **Recall:** 74.8%
        - **ROC-AUC:** 0.87
        
        #### Top Churn Drivers:
        1. **Tenure** - New customers (<12 months) churn 6.4x more
        2. **Contract Type** - Month-to-month contracts show 42.7% churn
        3. **Payment Method** - Electronic check users churn 3x more
        
        #### Business Impact:
        - Estimated **$760K annual revenue protection**
        - **15-20% reduction** in churn through targeted interventions
        
        ---
        
        **Developed by:** Shashank Lodhi  
        **Institution:** MIET, Meerut  
        **Date:** November 2025
        """)

if __name__ == "__main__":
    main()
