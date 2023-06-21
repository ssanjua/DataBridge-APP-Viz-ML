import streamlit as st
import pandas as pd
from textblob import TextBlob

st.set_page_config(page_title = 'DA - DataBrick APP', initial_sidebar_state='expanded')

st.image('./img/dataBRIDGE_logo_black.png')
if st.checkbox('elegi algo'):
    st.write('holandia')

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
    st.header('Creés que tu negocio se beneficia con nuestro servicio? :robot_face:')

# Mostrar el campo de entrada en la segunda columna
with input_col:
    mood = st.text_input(' :clap: :question:')
    st.markdown('#### A la AI :robot_face: le parece que tu comentario fue ' + get_sentiment(mood))

