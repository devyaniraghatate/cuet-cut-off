import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page title and styling
st.set_page_config(page_title="CUET Cut-off Predictor", layout="centered")

st.title("🎓 CUET Cut-off Predictor")
st.write("Predict the expected cut-off figure based on historical models.")

# Load trained pipeline
@st.cache_resource
def load_model():
    return joblib.load('best_gradient_boosting_pipeline.pkl')

try:
    pipeline = load_model()
    
    # Setup input widgets
    university = st.selectbox("Select University", [
        'Banaras Hindu University', 'University of Delhi', 
        'Guru Gobind Singh Indraprastha University', 'Jawaharlal Nehru University'
    ])
    
    category = st.selectbox("Select Category", ['UR', 'OBC-NCL', 'SC', 'ST', 'EWS', 'PwBD'])
    
    seat_type = st.selectbox("Select Seat Type", ['general', 'Paid Fee Course', 'Delhi region'])
    
    year = st.slider("Target Year", min_value=2024, max_value=2030, value=2026, step=1)
    
    if st.button("🔮 Predict Cut-off"):
        input_df = pd.DataFrame([{
            'university': university,
            'category': category,
            'seat_type': seat_type,
            'year': year
        }])
        
        prediction = pipeline.predict(input_df)[0]
        st.success(f"🎯 Expected Cut-off Figure: **{prediction:.2f}**")

except Exception as e:
    st.error(f"Error loading model components: {e}")
