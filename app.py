import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Predicción de Enfermedad Cardíaca")

# Cargar el modelo de ML
try:
    model = joblib.load('modelo_heart_disease.pkl')
    st.success("Modelo cargado correctamente")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# Formulario para que el usuario ingrese sus datos
st.subheader("Ingrese los datos del paciente:")

# Ejemplo de variables de entrada (ajústalas a las de tu modelo)
age = st.number_input("Edad", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sexo", options=[0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
cp = st.selectbox("Tipo de dolor en el pecho (CP)", options=[0, 1, 2, 3])
trestbps = st.number_input("Presión arterial en reposo (trestbps)", value=120)
chol = st.number_input("Colesterol (chol)", value=200)

# Botón para realizar la predicción
if st.button("Realizar Predicción"):
    input_data = np.array([[age, sex, cp, trestbps, chol]]) # Ajusta el orden exacto de tus variables
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ El modelo indica un alto riesgo de enfermedad cardíaca.")
    else:
        st.success("✅ El modelo indica un bajo riesgo de enfermedad cardíaca.")
