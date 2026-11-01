

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Predicción de Enfermedad Cardíaca - ACA 1",
    page_icon="🫀",
    layout="wide",
)

st.title("🫀 Sistema de Predicción de Enfermedad Cardíaca")
st.write(
    "Proyecto de Aprendizaje Automático (ACA 1) - Análisis y Diagnóstico"
    " Clínico"
)


# --- CARGA DEL MODELO ---
@st.cache_resource
def load_model():
    return joblib.load("modelo_heart_disease.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(f"Error al cargar el archivo 'modelo_heart_disease.pkl': {e}")
    st.stop()

# --- PESTAÑAS DE LA APLICACIÓN ---
tab_pred, tab_graficas = st.tabs(
    ["📋 Realizar Predicción", "📊 Gráficas del Colab (EDA y Evaluación)"]
)

with tab_pred:
    st.subheader("Ingrese los datos del paciente:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Edad (age)", min_value=1, max_value=120, value=29
        )
        sex = st.selectbox(
            "Sexo (sex)",
            options=[0, 1],
            format_func=lambda x: (
                "Femenino (0)" if x == 0 else "Masculino (1)"
            ),
        )

        # MAPEO CORRECTO DE CP (0 a 3 según el entrenamiento en Colab)
        cp = st.selectbox(
            "Tipo de dolor en el pecho (cp)",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0: Sin dolor / Angina Típica",
                1: "1: Angina Atípica",
                2: "2: Dolor No Anginoso",
                3: "3: Asintomático",
            }[x],
        )

        trestbps = st.number_input(
            "Presión arterial en reposo mm Hg (trestbps)", value=112
        )
        chol = st.number_input("Colesterol sérico mg/dl (chol)", value=180)
        fbs = st.selectbox(
            "Azúcar en sangre en ayunas > 120 mg/dl (fbs)", options=[0, 1]
        )
        restecg = st.selectbox(
            "Resultados electrocardiógrafo en reposo (restecg)", options=[0, 1, 2]
        )

    with col2:
        thalach = st.number_input(
            "Frecuencia cardíaca máxima alcanzada (thalach)", value=150
        )
        exang = st.selectbox(
            "Angina inducida por ejercicio (exang)", options=[0, 1]
        )
        oldpeak = st.number_input(
            "Depresión del ST inducida por ejercicio (oldpeak)",
            value=0.0,
            step=0.1,
        )

        # MAPEO CORRECTO DE SLOPE (0, 1, 2 según sklearn/Cleveland)
        slope = st.selectbox(
            "Pendiente del segmento ST (slope)", options=[0, 1, 2]
        )
        ca = st.selectbox(
            "Número de vasos principales coloreados (ca)", options=[0, 1, 2, 3]
        )

        # MAPEO CORRECTO DE THAL (3, 6, 7 del dataset original UCI)
        thal = st.selectbox(
            "Talasemia - Thal (thal)",
            options=[3.0, 6.0, 7.0],
            format_func=lambda x: {
                3.0: "Normal (3)",
                6.0: "Defecto Fijo (6)",
                7.0: "Defecto Reversible (7)",
            }[x],
        )

    # DataFrame con los datos en el orden exacto de las columnas de entrenamiento
    input_df = pd.DataFrame([
        {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
    ])

    if st.button("Realizar Predicción"):
        scaled_df = input_df.copy()

        # Estandarización idéntica al StandardScaler de Colab
        means = {
            "age": 54.44,
            "trestbps": 131.69,
            "chol": 246.69,
            "thalach": 149.61,
            "oldpeak": 1.04,
        }
        stds = {
            "age": 9.04,
            "trestbps": 17.60,
            "chol": 51.78,
            "thalach": 22.88,
            "oldpeak": 1.16,
        }

        for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
            scaled_df[col] = (scaled_df[col] - means[col]) / stds[col]

        try:
            prediction = model.predict(scaled_df)
        except Exception:
            prediction = model.predict(scaled_df.values)

        if prediction[0] == 1:
            st.error("⚠️ El modelo indica un **ALTO RIESGO** de enfermedad cardíaca.")
        else:
            st.success(
                "✅ El modelo indica un **BAJO RIESGO** de enfermedad cardíaca."
            )
        if prediction[0] == 1:
            st.error("⚠️ El modelo indica un **ALTO RIESGO** de enfermedad cardíaca.")
        else:
            st.success("✅ El modelo indica un **BAJO RIESGO** de enfermedad cardíaca.")
