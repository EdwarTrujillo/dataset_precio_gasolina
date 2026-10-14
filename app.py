import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Cargar el modelo y el escalador entrenados
modelo = joblib.load('modelo_precio_gasolina.pkl')
scaler = joblib.load('scaler_gasolina.pkl')

st.title("⛽ Predicctor de Precio Mensual de Gasolina (USD/Galón)")
st.write("Ingrese los valores de las variables económicas, energéticas y geopolíticas para calcular la predicción del precio.")

# Crear campos de entrada para cada una de las 8 variables del modelo
st.sidebar.header("Parámetros de Entrada")

precio_petroleo = st.sidebar.number_input("Precio Petróleo (USD/Barril)", min_value=0.0, value=70.0, step=1.0)
tipo_cambio = st.sidebar.number_input("Tipo de Cambio (USD/COP)", min_value=0.0, value=3800.0, step=50.0)
demanda = st.sidebar.number_input("Demanda (miles barriles/día)", min_value=0.0, value=9500.0, step=10.0)
produccion_opep = st.sidebar.number_input("Producción OPEP (millones/día)", min_value=0.0, value=28.5, step=0.1)
inflacion = st.sidebar.number_input("Inflación (%)", value=3.5, step=0.1)
pib = st.sidebar.number_input("PIB (%)", value=2.5, step=0.1)
eventos_geo = st.sidebar.selectbox("Eventos Geopolíticos", options=[0, 1], format_func=lambda x: "Sí (1)" if x == 1 else "No (0)")
impuestos = st.sidebar.number_input("Impuestos Combustibles (%)", min_value=0.0, value=22.0, step=1.0)

# Botón para realizar la predicción
if st.button("Calcular Predicción del Precio"):
    # Organizar los datos ingresados en un DataFrame con los mismos nombres de columnas
    entrada_usuario = pd.DataFrame([[
        precio_petroleo, tipo_cambio, demanda, produccion_opep, 
        inflacion, pib, eventos_geo, impuestos
    ]], columns=[
        'Precio_Petroleo_USD_Barril', 'Tipo_Cambio_USD_COP', 
        'Demanda_miles_barriles_dia', 'Produccion_OPEP_millones_dia', 
        'Inflacion_Porcentaje', 'PIB_Porcentaje', 
        'Eventos_Geopoliticos', 'Impuestos_Combustibles_Porcentaje'
    ])
    
    # Normalizar los datos de entrada usando el mismo escalador del entrenamiento
    entrada_escalada = scaler.transform(entrada_usuario)
    
    # Realizar la predicción con el modelo
    prediccion = modelo.predict(entrada_escalada)[0]
    
    # Mostrar el resultado en pantalla de forma llamativa
    st.success(f"### Precio Estimado de la Gasolina: **${prediccion:.2f} USD/galón**")