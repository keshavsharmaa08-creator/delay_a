%%writefile app.py
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('logi.sav')

# Streamlit app title
st.title('Delivery Delay Prediction App')

st.write("Enter the details below to predict if there will be a delivery delay.")

# Create input fields for each feature
# Feature names from X.columns: 'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
# 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
# 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
# 'Warehouse_Processing_Time'

delivery_distance = st.number_input('Delivery Distance', min_value=0.0, value=20.0)
traffic_congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1-5)', 1, 5, 2)
delivery_slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
driver_experience = st.number_input('Driver Experience (years)', min_value=0, value=5)
num_stops = st.number_input('Number of Stops', min_value=0, value=2)
vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, value=3)
road_condition_score = st.slider('Road Condition Score (1-5)', 1, 5, 3)
package_weight = st.number_input('Package Weight', min_value=0.0, value=10.0)
fuel_efficiency = st.number_input('Fuel Efficiency', min_value=0.0, value=15.0)
warehouse_processing_time = st.number_input('Warehouse Processing Time (minutes)', min_value=0, value=60)

# Create a prediction button
if st.button('Predict Delivery Delay'):
    # Prepare the input features as a NumPy array
    features = np.array([delivery_distance, traffic_congestion, weather_condition,
                         delivery_slot, driver_experience, num_stops, vehicle_age,
                         road_condition_score, package_weight, fuel_efficiency,
                         warehouse_processing_time]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)[0]
    
    st.subheader('Prediction Results:')
    if prediction[0] == 1:
        st.error('Prediction: Delivery Delay Expected')
    else:
        st.success('Prediction: No Delivery Delay Expected')
        
    st.write(f'Probability of No Delay: {prediction_proba[0]:.2f}')
    st.write(f'Probability of Delay: {prediction_proba[1]:.2f}')
