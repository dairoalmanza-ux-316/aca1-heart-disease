import pandas as pd
import streamlit as st

# (Asegúrate de incluir tus imports y la carga de tu modelo previamente)

# 1. Inputs de la barra lateral (o formulario)
oldpeak = st.sidebar.number_input(
    "Depresión del segmento ST (oldpeak)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
)

# 2. Botón de predicción y lógica
if st.button("Ejecutar Evaluación Clínica"):
    datos_paciente = pd.DataFrame(
        [
            {
                "age": edad,
                "trestbps": presion,
                "chol": colesterol,
                "thalach": freq_max,
                "oldpeak": oldpeak,
            }
        ]
    )

    prediccion = model.predict(datos_paciente)
    probabilidad = model.predict_proba(datos_paciente)[0][1] * 100

    if prediccion[0] == 1:
        st.error(
            f"⚠️ **Riesgo Alto:** El modelo detecta indicios de riesgo cardíaco ({probabilidad:.1f}% de probabilidad)."
        )
    else:
        st.success(
            f"✅ **Riesgo Bajo:** El paciente no presenta indicios significativos de riesgo ({100 - probabilidad:.1f}% de seguridad)."
        )
