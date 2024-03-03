import cv2
import numpy as np
from keras.models import load_model
import streamlit as st
import pyttsx3
import threading

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

# Placeholder for displaying the camera feed
frame_placeholder = st.empty()

# Add buttons below the camera feed
start_button = st.button("Start")
pause_button = st.button("Pause")
resume_button = st.button("Resume")

# Function to read frames from the camera
def read_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:
            break

        # Perform prediction on the frame
        predicted_class = predict_class(frame)

        # Speak out the recognized class in a separate thread
        threading.Thread(target=speak_class, args=(predicted_class,)).start()

        # Get the corresponding message based on the detected class
        message = class_messages.get(predicted_class, "Unknown")

        # Display the predicted class and confidence on the frame
        cv2.putText(frame, f'{predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Message: {message}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in the Streamlit app
        frame_placeholder.image(frame_rgb, channels='RGB')

        # Delay to control frame rate and reduce processing load
        cv2.waitKey(30)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Handle button clicks
if start_button:
    threading.Thread(target=read_camera).start()
elif pause_button:
    st.write("Pause button clicked.")
elif resume_button:
    st.write("Resume button clicked.")
