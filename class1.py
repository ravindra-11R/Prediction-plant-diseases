import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model_path = r'C:\Users\rabid\plant_disease_classifier\best_model_complete.h5'  # Ensure this path is correct
model = tf.keras.models.load_model(model_path)

# Class labels
labels = ['Healthy', 'Powdery', 'Rust']

# Streamlit app
st.title('Plant Disease Classification')

# File uploader
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
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






