import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import os

# Google Drive file ID
file_id = '1sZA79FXoZ12YnUoB7VGXI49-wMWPtCY6'
output = 'best_custom_model.h5'

# Function to download the model
def download_model():
    try:
        gdown.download(
            "https://drive.google.com/uc?export=download&id="
            "1sZA79FXoZ12YnUoB7VGXI49-wMWPtCY6"
, output, quiet=False)
        if not os.path.exists(output):
            st.error("Model file not found after download.")
        else:
            st.success("Model file downloaded successfully.")
    except Exception as e:
        st.error(f"Error downloading the model: {e}")

# Download model
download_model()

# Load the model
try:
    model = tf.keras.models.load_model(output)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None

# Class labels
labels = ['Healthy', 'Powdery', 'Rust']

# Streamlit app
st.title('Plant Disease Classification')

# File uploader
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None and model:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = image.resize((256, 256))  # Resize to match the model's input size
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict the class
    preds = model.predict(img)
    preds_class = np.argmax(preds)
    preds_label = labels[preds_class]
    confidence = preds[0][preds_class]

    st.write(f'Predicted Class: {preds_label}')
    st.write(f'Confidence Score: {confidence:.2f}')
else:
    st.warning("Model not loaded or no file uploaded.")





