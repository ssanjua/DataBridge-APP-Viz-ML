import streamlit as st
import pandas as pd
import numpy as np
import joblib
from geopy.geocoders import Nominatim
from textblob import TextBlob

st.set_page_config(page_title = 'ML - DataBrick APP', initial_sidebar_state='expanded')
st.image('./img/dataBRIDGE_logo_black.png')

st.markdown('***')

st.title("MACHINE LEARNING :alien:")

st.markdown('''
                <h3>¡Descubre las <strong>tendencias</strong> en el 
                <strong>mercado laboral</strong> con nuestro modelo de
                <strong>Análisis de sentimiento y Análisis de comportamiento de mercado!</strong></h3>
                <p>Con estos modelo podemos obtener un rápido e intuitivo análisis del sentimiento de las reviews de los usuarios y de la competitividad en las distintas zonas de EEUU. </p>
                <hr>
            ''', unsafe_allow_html=True)

st.subheader('ANÁLISIS DE SENTIMIENTO :heart_eyes:')

df_sentimiento = pd.read_csv('./data/df_sentimiento_small.csv')
rubro = st.selectbox('Selecciona un rubro', df_sentimiento['category'].unique())
estados = st.selectbox('Seleccionar Estado:', df_sentimiento['state'].unique())
sentimiento = st.radio("Filtrar por sentimiento:", ('Positivo', 'Neutral', 'Negativo'), horizontal=True)

filtro_anio = st.slider('Filtrar por años', 2007, 2022)

df_filtrado = df_sentimiento[(df_sentimiento['category'] == rubro) & (df_sentimiento['state'] == estados) & (df_sentimiento['year'] == filtro_anio) & (df_sentimiento['Sentimiento'] == sentimiento)]

if not df_filtrado.empty:
    nombre_local = df_filtrado['name'].iloc[0]
    sentimiento = df_filtrado['Sentimiento'].iloc[0]
    reseña = df_filtrado['text'].iloc[0]

    st.write('De acuerdo al rubro', rubro, 'el local', nombre_local, 'recibió en el año', filtro_anio, 'la siguiente reseña:', reseña, 'con sentimiento: ', sentimiento)
else:
    st.subheader('No se encontraron datos para los filtros seleccionados.')

st.markdown('***')

st.subheader('ANALISIS DE COMPORTAMIENTO DE MERCADO :moneybag:')
#df_filtrado = df[filtro_anio, estados, rubros, nombre, sentimiento]

geolocator = Nominatim(user_agent="my_app")
kmeans = joblib.load('./data/kmeans_model.pkl')
centers = joblib.load('./data/cluster_centers.pkl')
top_clusters = pd.read_csv('./data/top_clusters.csv')

def Calcular_Cluster(city):
        location = geolocator.geocode(city, timeout=None)

        distances = np.linalg.norm(centers[:, :2] - np.array([location.latitude, location.longitude]), axis=1)
        # Obtener el índice del clúster más cercano
        closest_cluster_index = np.argmin(distances)
        # Obtener los datos completos del clúster más cercano
        closest_cluster_data = top_clusters.loc[top_clusters['Cluster'] == closest_cluster_index]
        # Imprimir los datos completos del clúster más cercano
        return closest_cluster_data["Centroid Latitude"], closest_cluster_data["Centroid Longitude"],closest_cluster_data["Promedio_Puntaje_Reviews"],closest_cluster_data["Porcentaje de competición"],closest_cluster_data["Cantidad de Negocios"],closest_cluster_data["Negocios Competidores"],closest_cluster_data["Estado"],closest_cluster_data["Condado"], location.latitude, location.longitude

respuesta = st.text_input('Ingresá la localidad donde quieras averiguar la competitividad/contexto de mercado de tu negocio: ', value="Denver")
Centroid_Latitude, Centroid_Longitude, Promedio_Puntaje_Reviews, Porcentaje_de_competición, Cantidad_de_Negocios, Negocios_Competidores, Estado, Condado, Latitud, Longitud = Calcular_Cluster(respuesta)

if Latitud > 49 or Latitud < 25 or Longitud > -66 or Longitud < -126:
    st.write("Por favor ingresa una ubicación dentro del territorio estadounidense")
else:
    st.write('La zona de ' + Condado.values[0] + ' tiene ' + str(Cantidad_de_Negocios.values[0]) + ' cantidad de negocios, con un porcentaje de competencia del ' + str(Porcentaje_de_competición.values[0]) +'% y ', str(Negocios_Competidores.values[0]) + ' negocios competidores con un promedio de calificaciones del: ' + str(Promedio_Puntaje_Reviews.values[0]))
    localizacion = pd.DataFrame({'LAT': Latitud, 'LON': Longitud}, index= [0])
    st.map(localizacion, zoom= 8)

st.markdown('***')
            
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0:
        return 'Positivo'
    elif polarity < 0:
        return 'Negativo'
    else:
        return 'Neutral'
    
# Crear una disposición de columnas con un encabezado y un campo de entrada
header_col, input_col = st.columns([1, 2])

# Mostrar el encabezado en la primera columna
with header_col:
    st.header('Cómo te sentís respecto de las AI? :robot_face:')

# Mostrar el campo de entrada en la segunda columna
with input_col:
    mood = st.text_input(' :smiley:  :confused: :sob: :grin:')
    st.markdown('#### A la AI :robot_face: le parece que tu comentario fue ' + get_sentiment(mood))
