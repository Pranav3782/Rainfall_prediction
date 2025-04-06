import streamlit as st
import pandas as pd
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Rainfall Prediction",
    page_icon="🌧️",
    layout="centered"
)

# Title and description
st.title("🌧️ Rainfall Prediction System")
st.write("Enter the weather parameters to predict rainfall probability")

# Create input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pressure = st.number_input("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1015.9, step=0.1)
        dewpoint = st.number_input("Dewpoint (°C)", min_value=-10.0, max_value=40.0, value=19.9, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=95)
        cloud = st.number_input("Cloud Cover (%)", min_value=0, max_value=100, value=81)
    
    with col2:
        sunshine = st.number_input("Sunshine (hours)", min_value=0.0, max_value=24.0, value=0.0, step=0.1)
        winddirection = st.number_input("Wind Direction (degrees)", min_value=0, max_value=360, value=40)
        windspeed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=13.7, step=0.1)

    submit_button = st.form_submit_button("Predict Rainfall")

# Load the model
@st.cache_resource
def load_model():
    model_path = "rainfall_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            return pickle.load(file)
    else:
        st.error("Model file not found! Please ensure the model is trained and saved.")
        return None

model = load_model()

# Make prediction when form is submitted
if submit_button:
    if model is not None:
        # Create input dataframe
        input_data = pd.DataFrame([[pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]], 
                                columns=['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed'])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Display result
        st.write("---")
        
        # Determine rainfall intensity based on probability
        rain_prob = probability[1]  # Probability of rainfall
        
        if prediction == 1:
            if rain_prob >= 0.8:
                st.error("🌧️ High chance of rainfall ({}%)".format(int(rain_prob * 100)))
            elif rain_prob >= 0.6:
                st.warning("🌦️ Moderate chance of rainfall ({}%)".format(int(rain_prob * 100)))
            else:
                st.info("🌤️ Low chance of rainfall ({}%)".format(int(rain_prob * 100)))
        else:
            if rain_prob <= 0.2:
                st.success("☀️ Very low chance of rainfall ({}%)".format(int((1 - rain_prob) * 100)))
            else:
                st.info("🌤️ Slight chance of rainfall ({}%)".format(int((1 - rain_prob) * 100)))
        
        # Display input parameters summary
        st.write("### Input Parameters Summary")
        summary_df = input_data.T
        summary_df.columns = ['Value']
        st.dataframe(summary_df)
        
        # Display weather conditions analysis
        st.write("### Weather Conditions Analysis")
        
        # Analyze humidity
        if humidity > 80:
            st.write("- High humidity ({}%) indicates increased rainfall potential".format(humidity))
        elif humidity < 40:
            st.write("- Low humidity ({}%) suggests dry conditions".format(humidity))
            
        # Analyze cloud cover
        if cloud > 75:
            st.write("- Heavy cloud cover ({}%) increases rainfall chances".format(cloud))
        elif cloud < 30:
            st.write("- Clear skies ({}%) suggest dry conditions".format(cloud))
            
        # Analyze pressure
        if pressure < 1000:
            st.write("- Low pressure ({}hPa) often associated with rainfall".format(pressure))
        elif pressure > 1020:
            st.write("- High pressure ({}hPa) typically indicates stable weather".format(pressure))
