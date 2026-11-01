%%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Predicción de Enfermedad Cardíaca - ACA 1",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Sistema de Predicción de Enfermedad Cardíaca")
st.write("Proyecto de Aprendizaje Automático (ACA 1) - Análisis y Diagnóstico Clínico")

# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    return joblib.load('modelo_heart_disease.pkl')

try:
    model = load_model()

    with st.expander("ℹ️ Información del Modelo en Ejecución", expanded=False):
        st.write(f"**Algoritmo Activo:** `{type(model).__name__}`")
        if hasattr(model, "n_estimators"):
            st.write(f"**Número de Árboles (n_estimators):** `{model.n_estimators}`")
        if hasattr(model, "max_depth"):
            st.write(f"**Profundidad Máxima (max_depth):** `{model.max_depth}`")
        st.write("**Estado:** Modelo cargado y operativo ✅")
except Exception as e:
    st.error(f"Error al cargar el archivo 'modelo_heart_disease.pkl': {e}")
    st.stop()

# --- PESTAÑAS DE LA APLICACIÓN ---
tab_pred, tab_graficas = st.tabs(["📋 Realizar Predicción", "📊 Gráficas del Colab (EDA y Evaluación)"])

with tab_pred:
    st.subheader("Ingrese los datos del paciente:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Edad (age)", min_value=1, max_value=120, value=29)
        sex = st.selectbox("Sexo (sex)", options=[0, 1], format_func=lambda x: "Femenino (0)" if x == 0 else "Masculino (1)")
        # En el dataset de Cleveland los tipos de dolor (cp) van de 1 a 4
        cp = st.selectbox("Tipo de dolor en el pecho (cp)", options=[1, 2, 3, 4], 
                          format_func=lambda x: {1: "1: Angina Típica", 2: "2: Angina Atípica", 3: "3: Dolor No Anginoso", 4: "4: Asintomático"}[x])
        trestbps = st.number_input("Presión arterial en reposo mm Hg (trestbps)", value=112)
        chol = st.number_input("Colesterol sérico mg/dl (chol)", value=180)
        fbs = st.selectbox("Azúcar en sangre en ayunas > 120 mg/dl (fbs)", options=[0, 1])
        restecg = st.selectbox("Resultados electrocardiógrafo en reposo (restecg)", options=[0, 1, 2])

    with col2:
        thalach = st.number_input("Frecuencia cardíaca máxima alcanzada (thalach)", value=180)
        exang = st.selectbox("Angina inducida por ejercicio (exang)", options=[0, 1])
        oldpeak = st.number_input("Depresión del ST inducida por ejercicio (oldpeak)", value=0.0, step=0.1)
        slope = st.selectbox("Pendiente del segmento ST (slope)", options=[1, 2, 3])
        ca = st.selectbox("Número de vasos principales coloreados (ca)", options=[0, 1, 2, 3])
        # CORRECCIÓN DE THAL: Los valores numéricos reales en el dataset de entrenamiento son 3.0, 6.0 y 7.0
        thal = st.selectbox("Talasemia - Thal (thal)", options=[3.0, 6.0, 7.0], 
                            format_func=lambda x: {3.0: "Normal (3)", 6.0: "Defecto Fijo (6)", 7.0: "Defecto Reversible (7)"}[x])

    # Construcción del DataFrame original con el mismo orden exacto de columnas que en el entrenamiento
    input_df = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])

    if st.button("Realizar Predicción"):
        scaled_df = input_df.copy()
        
        # Medias y desviaciones según el StandardScaler del Colab
        means = {'age': 54.44, 'trestbps': 131.69, 'chol': 246.69, 'thalach': 149.61, 'oldpeak': 1.04}
        stds = {'age': 9.04, 'trestbps': 17.60, 'chol': 51.78, 'thalach': 22.88, 'oldpeak': 1.16}

        for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
            scaled_df[col] = (scaled_df[col] - means[col]) / stds[col]

        try:
            prediction = model.predict(scaled_df)
        except Exception:
            prediction = model.predict(scaled_df.values)

        if prediction[0] == 1:
            st.error("⚠️ El modelo indica un **ALTO RIESGO** de enfermedad cardíaca.")
        else:
            st.success("✅ El modelo indica un **BAJO RIESGO** de enfermedad cardíaca.")
