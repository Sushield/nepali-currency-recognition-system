import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model("my_model.keras")

# Labels for classes (adjust according to your dataset)
class_names = ['0', '1', '2', '3', '4', '5', '6']

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
    confidence = pred[class_index]  # Confidence score for the predicted class
    class_name = class_names[class_index]  # Get class name
    return class_name, confidence

# Dictionary to map classes to their corresponding message
class_messages = {
    '0': "five",
    '1': "ten",
    '2': "twenty",
    '3': "fifty",
    '4': "hundred",
    '5': "five hundred",
    '6': "thousand"
}

# Open the camera
cap = cv2.VideoCapture(0)

# Main loop for real-time prediction
while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break

    # Perform prediction on the frame
    predicted_class, confidence = predict_class(frame)

    # Get the corresponding message based on the detected class
    message = class_messages.get(predicted_class, "Unknown")

    # Display the predicted class and confidence on the frame
    cv2.putText(frame, f'{predicted_class} - {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Message: {message}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Real-time Prediction', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
