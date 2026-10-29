import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Predicción de Enfermedad Cardíaca")

# Cargar el modelo
try:
    model = joblib.load('modelo_heart_disease.pkl')
    st.success("Modelo cargado correctamente")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

st.subheader("Ingrese los datos del paciente:")

# Formulario con las 13 características estándar
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Edad (age)", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sexo (sex)", options=[0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
    cp = st.selectbox("Tipo de dolor en el pecho (cp)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Presión arterial en reposo (trestbps)", value=120)
    chol = st.number_input("Colesterol (chol)", value=200)
    fbs = st.selectbox("Azúcar en sangre en ayunas > 120 mg/dl (fbs)", options=[0, 1])
    restecg = st.selectbox("Resultados electrocardiógrafo en reposo (restecg)", options=[0, 1, 2])

with col2:
    thalach = st.number_input("Frecuencia cardíaca máxima alcanzada (thalach)", value=150)
    exang = st.selectbox("Angina inducida por ejercicio (exang)", options=[0, 1])
    oldpeak = st.number_input("Depresión del ST (oldpeak)", value=1.0, step=0.1)
    slope = st.selectbox("Pendiente del segmento ST (slope)", options=[0, 1, 2])
    ca = st.selectbox("Número de vasos principales coloreados (ca)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thal (thal)", options=[0, 1, 2, 3])

# Construcción del DataFrame con las 13 variables
input_df = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}])

if st.button("Realizar Predicción"):
    try:
        prediction = model.predict(input_df)
    except Exception:
        prediction = model.predict(input_df.values)

    # Lógica corregida según la etiqueta real de UCI:
    if prediction[0] == 0:
        st.error("⚠️ El modelo indica un alto riesgo de enfermedad cardíaca.")
    else:
        st.success("✅ El modelo indica un bajo riesgo de enfermedad cardíaca.")
