import cv2
import threading
import numpy as np
from keras.models import load_model
import streamlit as st
import pyttsx3

# Load the trained model
model = load_model("my_model.keras")

# Labels for classes (adjust according to your dataset)
class_names = ['0', '1', '2', '3', '4', '5', '6']

# Dictionary to map classes to their corresponding message
class_messages = {
    '0': "panch rupaiy",
    '1': "das rupaiya",
    '2': "bis rupaiya",
    '3': "pachas rupaiya",
    '4': "saye rupaiya",
    '5': "pach saye rupaiya",
    '6': "ek hajar rupaiya"
}

# Initialize the pyttsx3 engine for audio output
engine = pyttsx3.init()

# Function to speak out the recognized class
def speak_class(class_name):
    message = class_messages.get(class_name, "Unknown")
    engine.say(message)
    engine.runAndWait()

# Function to preprocess the image
def preprocess_img(img):
    img = cv2.resize(img, (224, 224))  # Resize to match model input shape
    img = img / 255.0  # Normalize pixel values
    return img

# Function to predict class
def predict_class(img):
    img = preprocess_img(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    pred = model.predict(img)[0]  # Perform prediction
    class_index = np.argmax(pred)  # Get index of predicted class
    class_name = class_names[class_index]  # Get class name
    return class_name

# Streamlit UI
st.title("Real-time Note Recognition")

# Create a layout with two columns: one for the camera feed and the other for buttons
col1, col2 = st.columns([3, 1])

# Placeholder for displaying the camera feed
frame_placeholder = col1.empty()

# Buttons in the second column
start_button = col2.button("Start")
pause_button = col2.button("Pause")
resume_button = col2.button("Resume")

# Initialize Streamlit state variable for the camera stream status
camera_status = st.session_state.get("camera_status", False)

# Function to capture and display camera frames
def capture_frames():
    cap = cv2.VideoCapture(0)  # Open the default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height

    while cap.isOpened() and camera_status:
        ret, frame = cap.read()  # Read a frame
        if not ret:
            break  # Exit the loop if frame reading fails

        # Display the frame in the Streamlit app
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)

    # Release the camera
    cap.release()

# Handle button clicks
if start_button:
    camera_status = True  # Set camera status to True to start capturing
    st.session_state["camera_status"] = camera_status  # Update session state
    threading.Thread(target=capture_frames).start()  # Start capturing frames in a thread

elif pause_button:
    # Set camera status to False to pause capturing
    camera_status = False
    st.session_state["camera_status"] = camera_status

elif resume_button:
    # Set camera status to True to resume capturing
    camera_status = True
    st.session_state["camera_status"] = camera_status
