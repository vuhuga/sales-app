import streamlit as st
import joblib
import numpy as np
import pandas as pd
!pip install setuptools
!pip install --upgrade pip



# Set app title
st.title('📈 Sales Prediction App')

# Load your trained model and preprocessing objects
@st.cache_resource  # Cache the model for better performance
def load_model():
    model = joblib.load('sales_prediction_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_model()

# Create input fields based on your model's features
st.sidebar.header('User Input Features')

# Example input fields - modify according to your model's requirements
def user_input_features():
    feature1 = st.sidebar.number_input('Feature 1 (e.g., Advertising Budget)', min_value=0.0, value=1000.0)
    feature2 = st.sidebar.number_input('Feature 2 (e.g., Price)', min_value=0.0, value=50.0)
    feature3 = st.sidebar.selectbox('Feature 3 (e.g., Season)', ['Spring', 'Summer', 'Fall', 'Winter'])
    # Add all your model features here
    
    data = {
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3
        # Add all features to this dictionary
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user inputs
st.subheader('User Input Features')
st.write(input_df)

# Preprocess and predict
if st.button('Predict Sales'):
    # Preprocess the input
    processed_input = preprocessor.transform(input_df)
    
    # Make prediction
    prediction = model.predict(processed_input)
    
    # Display prediction
    st.subheader('Prediction Result')
    st.success(f'Predicted Sales: ${prediction[0]:,.2f}')

# Optional: Add more information
st.markdown("""
### About This App
This app predicts sales based on machine learning model.
- **Python libraries:** streamlit, scikit-learn, pandas, numpy
- **Model:** [Your model type] trained on [your dataset]
""")
