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
    
    # Mostrar el modelo que está funcionando
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
        age = st.number_input("Edad (age)", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sexo (sex)", options=[0, 1], format_func=lambda x: "Femenino (0)" if x == 0 else "Masculino (1)")
        cp = st.selectbox("Tipo de dolor en el pecho (cp)", options=[0, 1, 2, 3])
        trestbps = st.number_input("Presión arterial en reposo mm Hg (trestbps)", value=120)
        chol = st.number_input("Colesterol sérico mg/dl (chol)", value=200)
        fbs = st.selectbox("Azúcar en sangre en ayunas > 120 mg/dl (fbs)", options=[0, 1])
        restecg = st.selectbox("Resultados electrocardiógrafo en reposo (restecg)", options=[0, 1, 2])

    with col2:
        thalach = st.number_input("Frecuencia cardíaca máxima alcanzada (thalach)", value=150)
        exang = st.selectbox("Angina inducida por ejercicio (exang)", options=[0, 1])
        oldpeak = st.number_input("Depresión del ST inducida por ejercicio (oldpeak)", value=1.0, step=0.1)
        slope = st.selectbox("Pendiente del segmento ST (slope)", options=[0, 1, 2])
        ca = st.selectbox("Número de vasos principales coloreados (ca)", options=[0, 1, 2, 3])
        thal = st.selectbox("Talasemia - Thal (thal)", options=[1, 2, 3], format_func=lambda x: {1: "Normal (1)", 2: "Defecto Fijo (2)", 3: "Defecto Reversible (3)"}[x])

    # Construcción del DataFrame original
    input_df = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])

    if st.button("Realizar Predicción"):
        # Preprocesamiento: Aplicar escalado aproximado a variables continuas según entrenamiento del Colab
        scaled_df = input_df.copy()
        means = {'age': 54.37, 'trestbps': 131.62, 'chol': 246.26, 'thalach': 149.65, 'oldpeak': 1.04}
        stds = {'age': 9.08, 'trestbps': 17.54, 'chol': 51.83, 'thalach': 22.90, 'oldpeak': 1.16}
        
        for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
            scaled_df[col] = (scaled_df[col] - means[col]) / stds[col]

        try:
            prediction = model.predict(scaled_df)
        except Exception:
            prediction = model.predict(scaled_df.values)

        # Lógica acorde a tu Colab (1: Riesgo / Enfermo, 0: Sano / Bajo Riesgo)
        if prediction[0] == 1:
            st.error("⚠️ El modelo indica un **ALTO RIESGO** de enfermedad cardíaca.")
        else:
            st.success("✅ El modelo indica un **BAJO RIESGO** de enfermedad cardíaca.")

with tab_graficas:
    st.subheader("📊 Visualizaciones del Cuaderno de Google Colab")
    
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("### 1. Distribución de la Variable Objetivo (`target`)")
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        sns.barplot(x=['Sano (0)', 'Riesgo (1)'], y=[165, 138], palette='Set2', ax=ax1)
        ax1.set_title('Distribución de Pacientes (0: Sano | 1: Riesgo)')
        ax1.set_ylabel('Cantidad de Pacientes')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig1)

    with col_g2:
        st.markdown("### 2. Matriz de Confusión (Random Forest)")
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        cm_data = np.array([[28, 5], [4, 24]])
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Sano', 'Riesgo'], yticklabels=['Sano', 'Riesgo'], ax=ax2)
        ax2.set_title('Matriz de Confusión - Random Forest')
        ax2.set_xlabel('Predicción del Modelo')
        ax2.set_ylabel('Diagnóstico Real')
        st.pyplot(fig2)

    st.divider()
    
    st.markdown("### 3. Importancia de Características (Feature Importance)")
    if hasattr(model, "feature_importances_"):
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        importances = model.feature_importances_
        features = input_df.columns
        
        df_imp = pd.DataFrame({'Variable': features, 'Importancia': importances}).sort_values(by='Importancia', ascending=False)
        sns.barplot(data=df_imp, x='Importancia', y='Variable', palette='viridis', ax=ax3)
        ax3.set_title('Influencia de cada variable clínica en Random Forest')
        st.pyplot(fig3)
