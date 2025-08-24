import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import re
from io import StringIO
import matplotlib.pyplot as plt
import json

# Page configuration
st.set_page_config(
    page_title="N.H.E - Loan Approval Classifier",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .logo {
        text-align: center;
        font-size: 2.8rem;
        font-weight: bold;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    .feature-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .feature-table th, .feature-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .feature-table th {
        background-color: #f2f2f2;
    }
    .confidence-bar {
        height: 20px;
        background-color: #f0f0f0;
        border-radius: 10px;
        margin: 10px 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    .tab-container {
        padding: 1rem;
    }
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        
        # Get the actual feature names the model was trained with
        if hasattr(model, 'feature_names_in_'):
            actual_features = list(model.feature_names_in_)
        else:
            # Fallback based on common patterns in error message
            actual_features = [
                "no_of_dependents", "income_annum", "loan_amount", "loan_term", "cibil_score",
                "residential_assets_value", "commercial_assets_value", "luxury_assets_value",
                "bank_asset_value", "education_ Not Graduate", "self_employed_ Yes"
            ]
        
        return model, actual_features
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, []

# Function to normalize column names to match model expectations
def normalize_column_name(col_name, model_features):
    """
    Normalize column names to match model's expected feature names
    
    Parameters:
    col_name (str): Original column name
    model_features (list): List of feature names the model expects
    
    Returns:
    str: Normalized column name that matches model expectations
    """
    # Convert to lowercase and trim
    normalized = col_name.lower().strip()
    
    # Replace spaces and hyphens with underscores
    normalized = re.sub(r'[\s\-]+', '_', normalized)
    
    # Remove non-word characters (except underscores)
    normalized = re.sub(r'[^\w_]', '', normalized)
    
    # Special handling for the binary features to match model expectations
    # Check if our normalized name is close to any model feature
    for model_feature in model_features:
        model_simple = re.sub(r'[^\w]', '', model_feature.lower())
        if normalized == model_simple:
            return model_feature
    
    return normalized

# Function to process input data
def process_input_data(df, model_features, drop_extra=True):
    """
    Process input DataFrame to match model requirements
    
    Parameters:
    df (pd.DataFrame): Input data
    model_features (list): Features the model expects
    drop_extra (bool): Whether to drop extra columns
    
    Returns:
    pd.DataFrame: Processed data with expected features
    """
    df_processed = df.copy()
    
    # Normalize column names to match model expectations
    df_processed.columns = [normalize_column_name(col, model_features) for col in df_processed.columns]
    
    # Check if we have raw categorical columns that need encoding
    education_features = [feat for feat in model_features if 'education' in feat.lower()]
    employed_features = [feat for feat in model_features if 'employed' in feat.lower() or 'self_employed' in feat.lower()]
    
    if 'education' in df_processed.columns and education_features:
        target_feature = education_features[0]
        df_processed[target_feature] = (df_processed['education'].str.lower().str.strip() == 'not graduate').astype(int)
        df_processed.drop('education', axis=1, inplace=True)
    
    if 'self_employed' in df_processed.columns and employed_features:
        target_feature = employed_features[0]
        df_processed[target_feature] = (df_processed['self_employed'].str.lower().str.strip() == 'yes').astype(int)
        df_processed.drop('self_employed', axis=1, inplace=True)
    
    # Ensure all expected features are present
    for feature in model_features:
        if feature not in df_processed.columns:
            # For binary features, set default value of 0 if missing
            if any(x in feature.lower() for x in ['education', 'employed']):
                df_processed[feature] = 0
            else:
                st.error(f"Missing required feature: {feature}")
                return None
    
    # Drop extra columns if requested
    if drop_extra:
        extra_cols = [col for col in df_processed.columns if col not in model_features]
        if extra_cols:
            df_processed.drop(extra_cols, axis=1, inplace=True)
    
    # Reorder columns to match expected feature order
    df_processed = df_processed[model_features]
    
    # Convert numeric columns
    numeric_features = [feat for feat in model_features if not any(x in feat.lower() for x in ['education', 'employed'])]
    
    for feature in numeric_features:
        if feature in df_processed.columns:
            try:
                df_processed[feature] = pd.to_numeric(df_processed[feature])
            except:
                st.error(f"Could not convert {feature} to numeric")
                return None
    
    return df_processed

# Function to make predictions
def make_predictions(model, data, threshold=0.5):
    """
    Make predictions using the trained model
    
    Parameters:
    model: Trained model object
    data (pd.DataFrame): Processed input data
    threshold (float): Decision threshold
    
    Returns:
    tuple: (predictions, probabilities, inference_time)
    """
    start_time = time.time()
    
    try:
        # Check feature alignment first
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
            if list(data.columns) != expected_features:
                st.error(f"Feature mismatch!\nExpected: {expected_features}\nGot: {list(data.columns)}")
                return None, None, None
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(data)[:, 1]
        else:
            # For models without predict_proba, use decision function
            probabilities = model.decision_function(data)
            probabilities = 1 / (1 + np.exp(-probabilities))  # Sigmoid transformation
        
        # Apply threshold
        predictions = (probabilities >= threshold).astype(int)
        
        inference_time = time.time() - start_time
        
        return predictions, probabilities, inference_time
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None, None, None

# Function to create confidence bar
def confidence_bar(probability, threshold):
    """Create a visual confidence bar for prediction probability"""
    color = "SeaGreen" if probability >= threshold else "Tomato"
    
    bar_html = f"""
    <div class="confidence-bar">
        <div class="confidence-fill" style="width: {probability*100}%; background-color: {color};">
            {probability:.1%}
        </div>
    </div>
    """
    return bar_html

# Main app
def main():
    # Logo and branding
    st.markdown('<div class="logo">By N.H.E </div>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">Loan Approval Classifier</h1>', unsafe_allow_html=True)
    
    # Model selection
    col1, col2 = st.columns([1, 2])
    with col1:
        model_option = st.selectbox(
            "Select Model",
            ["Random Forest", "XGBoost"],
            help="Choose the machine learning model for prediction"
        )
    
    with col2:
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Approved if probability ≥ threshold; lowering increases recall, raising increases precision"
        )
    
    # Load selected model
    model_path = "Models/best_random_forest_model.pkl" if model_option == "Random Forest" else "Models/best_xgboost_model.pkl"
    
    try:
        model, model_features = load_model(model_path)
        if model is None:
            st.error("Failed to load model. Please check if the model file exists.")
            st.stop()
            
        # Show model diagnostics in sidebar
        st.sidebar.subheader("Model Diagnostics")
        if hasattr(model, 'feature_names_in_'):
            st.sidebar.write("Model features:", list(model.feature_names_in_))
        else:
            st.sidebar.warning("Model doesn't have feature_names_in_ attribute")
            st.sidebar.write("Using features:", model_features)
            
    except Exception as e:
        st.error(f"Could not load model from {model_path}. Error: {str(e)}")
        st.stop()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Predictions"])
    
    # Single Prediction Tab
    with tab1:
        st.header("Single Loan Application")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Applicant Information")
            
            no_of_dependents = st.slider("Number of Dependents", 0, 5, 0)
            income_annum = st.number_input("Annual Income", min_value=0, value=500000, step=10000)
            loan_amount = st.number_input("Loan Amount", min_value=0, value=300000, step=10000)
            loan_term = st.slider("Loan Term (years)", 1, 30, 10)
            cibil_score = st.slider("CIBIL Score", 300, 900, 700)
            
        with col2:
            st.subheader("Asset Information")
            
            residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=100000, step=10000)
            commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=50000, step=10000)
            luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=50000, step=10000)
            bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=100000, step=10000)
            
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        
        # Find the actual binary feature names the model expects
        education_feature = next((feat for feat in model_features if 'education' in feat.lower()), "education_ Not Graduate")
        employed_feature = next((feat for feat in model_features if 'employed' in feat.lower() or 'self_employed' in feat.lower()), "self_employed_ Yes")
        
        # Create input DataFrame with exact feature names
        input_data = {}
        for feature in model_features:
            if feature == 'no_of_dependents':
                input_data[feature] = no_of_dependents
            elif feature == 'income_annum':
                input_data[feature] = income_annum
            elif feature == 'loan_amount':
                input_data[feature] = loan_amount
            elif feature == 'loan_term':
                input_data[feature] = loan_term
            elif feature == 'cibil_score':
                input_data[feature] = cibil_score
            elif feature == 'residential_assets_value':
                input_data[feature] = residential_assets_value
            elif feature == 'commercial_assets_value':
                input_data[feature] = commercial_assets_value
            elif feature == 'luxury_assets_value':
                input_data[feature] = luxury_assets_value
            elif feature == 'bank_asset_value':
                input_data[feature] = bank_asset_value
            elif feature == education_feature:
                input_data[feature] = 1 if education == "Not Graduate" else 0
            elif feature == employed_feature:
                input_data[feature] = 1 if self_employed == "Yes" else 0
        
        input_df = pd.DataFrame([input_data])[model_features]
        
        # Show feature preview (excluding encoded columns)
        preview_data = {
            "no_of_dependents": no_of_dependents,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value,
            "education": education,
            "self_employed": self_employed
        }
        
        st.subheader("Final Feature Order (Preview)")
        preview_df = pd.DataFrame([preview_data])
        st.dataframe(preview_df, use_container_width=True)
        
        # Make prediction
        if st.button("Predict Loan Approval", type="primary"):
            predictions, probabilities, inference_time = make_predictions(model, input_df, threshold)
            
            if predictions is not None:
                prediction = predictions[0]
                probability = probabilities[0]
                
                st.subheader("Prediction Result")
                
                # Use Streamlit's native components for better UI
                if prediction == 1:
                    st.success(f"✅ **Loan Approved** (Probability: {probability:.3f})")
                else:
                    st.error(f"❌ **Loan Rejected** (Probability: {probability:.3f})")
                
                # Show confidence bar
                st.markdown("**Confidence Level:**")
                st.markdown(confidence_bar(probability, threshold), unsafe_allow_html=True)
                
                st.write(f"Inference time: {inference_time:.4f} seconds")
    
    # Batch Predictions Tab
    with tab2:
        st.header("Batch Loan Applications")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        drop_extra = st.checkbox("Automatically drop extra columns", value=True)
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                df = pd.read_csv(uploaded_file)
                
                # Show original column names for reference
                st.subheader("Uploaded File Columns")
                st.write(f"Detected columns: {list(df.columns)}")
                
                # Process data
                processed_df = process_input_data(df, model_features, drop_extra)
                
                if processed_df is not None:
                    # Show preview (excluding encoded columns)
                    education_feature = next((feat for feat in model_features if 'education' in feat.lower()), "education_ Not Graduate")
                    employed_feature = next((feat for feat in model_features if 'employed' in feat.lower() or 'self_employed' in feat.lower()), "self_employed_ Yes")
                    
                    preview_df = processed_df.copy()
                    preview_df['education'] = np.where(preview_df[education_feature] == 1, 'Not Graduate', 'Graduate')
                    preview_df['self_employed'] = np.where(preview_df[employed_feature] == 1, 'Yes', 'No')
                    
                    numeric_features = [feat for feat in model_features if not any(x in feat.lower() for x in ['education', 'employed'])]
                    preview_cols = numeric_features + ['education', 'self_employed']
                    
                    st.subheader("Processed Data Preview")
                    st.dataframe(preview_df[preview_cols], use_container_width=True)
                    
                    # Make predictions
                    if st.button("Run Batch Predictions", type="primary"):
                        with st.spinner("Making predictions..."):
                            predictions, probabilities, inference_time = make_predictions(model, processed_df, threshold)
                        
                        if predictions is not None:
                            # Add predictions to results
                            results_df = df.copy()
                            results_df['prediction_probability'] = probabilities
                            results_df['prediction'] = np.where(predictions == 1, 'Approved', 'Rejected')
                            
                            st.subheader("Prediction Results")
                            st.write(f"Processed {len(results_df)} applications in {inference_time:.4f} seconds")
                            
                            # Summary statistics with metric cards
                            approved_count = sum(predictions)
                            rejected_count = len(predictions) - approved_count
                            approval_rate = approved_count / len(predictions) if len(predictions) > 0 else 0
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.markdown('<div class="metric-card">Total Applications</div>', unsafe_allow_html=True)
                                st.markdown(f'<div style="text-align: center; font-size: 24px; font-weight: bold; color: white;">{len(results_df)}</div>', unsafe_allow_html=True)
                            with col2:
                                st.markdown('<div class="metric-card">Approved</div>', unsafe_allow_html=True)
                                st.markdown(f'<div style="text-align: center; font-size: 24px; font-weight: bold; color: SeaGreen;">{approved_count}</div>', unsafe_allow_html=True)
                            with col3:
                                st.markdown('<div class="metric-card">Rejected</div>', unsafe_allow_html=True)
                                st.markdown(f'<div style="text-align: center; font-size: 24px; font-weight: bold; color: Tomato;">{rejected_count}</div>', unsafe_allow_html=True)
                            with col4:
                                st.markdown('<div class="metric-card">Approval Rate</div>', unsafe_allow_html=True)
                                st.markdown(f'<div style="text-align: center; font-size: 24px; font-weight: bold; color: white;">{approval_rate:.1%}</div>', unsafe_allow_html=True)
                            
                            # Pie chart (smaller size)
                            fig, ax = plt.subplots(figsize=(3, 3))  # Reduced from default (6,6) to (5,5)
                            labels = ['Approved', 'Rejected']
                            sizes = [approved_count, rejected_count]
                            colors = ['SeaGreen', 'Tomato']
                            
                            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                            ax.axis('equal')
                            
                            st.pyplot(fig)
                            
                            # Results table
                            st.subheader("Detailed Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="loan_predictions.csv",
                                mime="text/csv"
                            )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Feature guide
    st.sidebar.header("Feature Guide")
    st.sidebar.markdown("""
    The model uses the following features for prediction:
    """)
    
    # Create feature table
    feature_descriptions = {
        "no_of_dependents": "Number of dependents (0-5)",
        "income_annum": "Annual income (in currency units)",
        "loan_amount": "Requested loan amount (in currency units)",
        "loan_term": "Loan term in years",
        "cibil_score": "Credit score (300-900)",
        "residential_assets_value": "Value of residential assets",
        "commercial_assets_value": "Value of commercial assets",
        "luxury_assets_value": "Value of luxury assets",
        "bank_asset_value": "Value of bank assets",
        "education": "Education level (Graduate/Not Graduate)",
        "self_employed": "Self-employed status (Yes/No)"
    }
    
    feature_table = "<table class='feature-table'><tr><th>Feature</th><th>Description</th></tr>"
    for feature, description in feature_descriptions.items():
        feature_table += f"<tr><td>{feature}</td><td>{description}</td></tr>"
    feature_table += "</table>"
    
    st.sidebar.markdown(feature_table, unsafe_allow_html=True)
    
    # Column normalization info
    st.sidebar.markdown("""
    **Column Name Normalization**:
    - Converted to lowercase
    - Spaces/hyphens replaced with underscores
    - Non-word characters removed
    - Examples: 
      - 'Income Annum' → 'income_annum'
      - 'Residential Assets-Value' → 'residential_assets_value'
      - 'CIBIL Score!' → 'cibil_score'
    """)

if __name__ == "__main__":
    main()




