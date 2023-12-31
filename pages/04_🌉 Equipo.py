import streamlit as st
import pandas as pd
from textblob import TextBlob

st.set_page_config(page_title = 'TEAM - DataBrick APP', initial_sidebar_state='expanded')
st.image('./img/dataBRIDGE_logo_black.png')
st.markdown('***')

st.header('Nuestro equipo')
st.code('SoyHenry_final(df10) = DataBrick')
st.write(['Analista Funcional: Pau Pallares', 'Data Engineer: Benja Zambelli', 'Data Engineer: Beder Riveros', 'Analista de datos: Claritzo Perez', 'ML engineer: Gonza']) # see *

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
   st.subheader("Pau Pallares")
   st.image("./img/x_pau.jpg")

with col2:
   st.subheader("Benjamin Zambelli")
   st.image("./img/x_benja.jpeg")

with col3:
   st.subheader("Gonza Schwerdt")
   st.image("./img/x_gonza.jpeg")

with col4:
   st.subheader("Claritzo Perez")
   st.image("./img/x_clari.jpeg")

with col5:
   st.subheader("Beder Rivera")
   st.image("./img/x_beder.png")

st.header('Roles')

tab1, tab2, tab3, tab4 = st.tabs(["Analista Funcional", "Data Engineer", "Analista de datos", "Machine Learning Engineer"])

with tab1:
   st.subheader("Analista Funcional")
   st.image("./img/r_analistafuncional.png")

with tab2:
   st.subheader("Data Engineer")
   st.image("./img/r_dataengineer.png")

with tab3:
   st.subheader("Analista de datos")
   st.image("./img/r_dataanalyst.png")

with tab4:
   st.subheader("ML Engineer")
   st.image("./img/r__mlengineer.png")


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
    st.header('Qué te pareció el trabajo del equipo? :robot_face:')

# Mostrar el campo de entrada en la segunda columna
with input_col:
    mood = st.text_input(' :+1: :neutral_face: :-1:')
    st.markdown('#### A la AI :robot_face: le parece que tu comentario fue ' + get_sentiment(mood))
