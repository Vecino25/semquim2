import streamlit as st
import tensorflow as tf
import numpy as np
from scipy.ndimage.interpolation import zoom
from streamlit_drawable_canvas import st_canvas
from utils import process_image
st.markdown("# Aplicación de :blue[Reconocimiento de] :green[Números] :pencil: 🦐")

# Load trained model
model = tf.keras.models.load_model('mi_modelo.h5')

st.write('Dibuja un número:')
# Display canvas for drawing
canvas_result = st_canvas(stroke_width=10, height=28*5, width=28*5)
  
# Process drawn image and make prediction using model
if np.any(canvas_result.image_data):
    # Convert drawn image to grayscale and resize to 28x28
    processed_image = process_image(canvas_result.image_data)
    # Make prediction using model
    prediction = model.predict(processed_image).argmax()
    # Display prediction
    st.header('Prediction:')
    st.markdown('This number appears to be a \n # :red[' + str(prediction) + ']')
    st.balloons()
else:
    # Display message if canvas is empty
    st.header('Predicción:')
    st.write('No se dibujó ningún número, dibuja uno para obtener una predicción.')
