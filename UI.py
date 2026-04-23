import cv2
import numpy as np
from keras.models import load_model
import streamlit as st

# Load model
model = load_model("my_model.keras")

class_names = ['0', '1', '2', '3', '4', '5', '6']

class_messages = {
    '0': "Panch Rupaiya",
    '1': "Das Rupaiya",
    '2': "Bis Rupaiya",
    '3': "Pachas Rupaiya",
    '4': "Saye Rupaiya",
    '5': "Pach Saye Rupaiya",
    '6': "Ek Hajar Rupaiya"
}

def preprocess_img(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img

def predict_class(img):
    img = preprocess_img(img)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]
    class_index = np.argmax(pred)
    return class_names[class_index], pred[class_index]

st.title("💰 Nepali Currency Recognition")

# Upload image
uploaded_file = st.file_uploader("Upload a currency image", type=["jpg", "png", "jpeg"])

# OR use webcam
camera_image = st.camera_input("Or take a picture")

image = None

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

elif camera_image:
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

if image is not None:
    st.image(image, channels="BGR")

    predicted_class, confidence = predict_class(image)
    message = class_messages.get(predicted_class, "Unknown")

    st.success(f"Detected: {message}")
    st.info(f"Confidence: {confidence:.2f}")
