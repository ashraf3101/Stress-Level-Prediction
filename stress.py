import numpy as np
import pickle
import streamlit as st

# Load the trained model
MODEL_PATH = r'/Users/adnan/Desktop/Stress/stress_trained.sav'

# Sidebar for branding
st.sidebar.title("Stress Predictor")
st.sidebar.markdown("Made by Adnan Alvi")

try:
    with open(MODEL_PATH, 'rb') as file:
        loaded_model = pickle.load(file)
    st.sidebar.success("Model loaded successfully!")
except FileNotFoundError:
    st.sidebar.error(f"Model file not found at path: {MODEL_PATH}. Please check the file path.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Function for stress level prediction
def stresslevel_prediction(input_data):
    try:
        # Convert the input data to a numpy array and reshape it
        id_np_array = np.asarray(input_data, dtype=float)
        id_reshaped = id_np_array.reshape(1, -1)

        # Make a prediction
        prediction = loaded_model.predict(id_reshaped)

        # Interpret the prediction
        if prediction[0] == 0:
            return "Stress Level: LOW"
        elif prediction[0] == 1:
            return "Stress Level: MEDIUM"
        else:
            return "Stress Level: HIGH"
    except ValueError as e:
        st.error(f"Invalid input data: {e}")
        return f"Invalid input data: {e}"
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return f"An error occurred during prediction: {e}"

# Main application
def main():
    # Page header with custom background and title
    st.markdown(
        """
        <style>
        body {
            background-color: #F0F2F6;
        }
        .main-title {
            font-size: 36px;
            color: #0078D4;
            font-weight: bold;
            text-align: center;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("<div class='main-title'>Stress Level Prediction Web App</div>", unsafe_allow_html=True)

    # st.image("https://cdn.pixabay.com/photo/2016/12/06/18/27/meditation-1886921_1280.jpg", use_column_width=True, caption="Stay Calm. Predict Your Stress Levels!")

    # Input Section
    with st.form(key='input_form'):
        st.markdown("### Input Your Data Below")
        Humidity = st.slider('Humidity (%)', min_value=10.0, max_value=100.0, value=40.0)
        Temperature = st.slider('Body Temperature (°F)', min_value=60.0, max_value=100.0, value=98.6)
        Step_count = st.number_input('Step Count', min_value=0, step=500, value=5000)
        
        submit = st.form_submit_button(label="Predict")

    # Prediction Section
    if submit:
        input_data = [Humidity, Temperature, Step_count]
        diagnosis = stresslevel_prediction(input_data)
        st.markdown(
            f"""
            <div style="padding: 15px; background-color: #E8F5E9; border-radius: 10px; margin-top: 20px;">
                <h3 style="color: #388E3C; text-align: center;">{diagnosis}</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )

if __name__ == '__main__':
    main()
