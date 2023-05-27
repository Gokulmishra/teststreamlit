import cv2
import streamlit as st
from keras.models import load_model
import numpy as np
import tensorflow as tf

def predict_age_gender(face_roi):
    # Preprocess the face image for age and gender prediction
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)  # Convert face_roi to grayscale
    face_roi = cv2.resize(face_roi, (128, 128))  # Resize to match model input size

    face_roi = face_roi.astype('float32') / 255.0  # Normalize pixel values
    face_roi = np.expand_dims(face_roi, axis=-1)  # Add channel dimension

    # Perform age and gender prediction
    
    pred = age_gender_model.predict(np.array([face_roi]))
    pred_gender = gender_dict[np.argmax(pred[0])]
    pred_age = int(pred[1])

    return pred_age, pred_gender

# Rest of the code...



# Load the pre-trained face detection model (such as Haar cascades)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the pre-trained age and gender prediction model (Keras .h5 format)
age_gender_model = load_model('epoch100.h5')

# Define the gender dictionary
gender_dict = {0: 'Male', 1: 'Female'}

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set Streamlit app configuration
st.set_page_config(page_title='Age and Gender Prediction', layout='wide')

# Set webcam feed width and height
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Create placeholders for displaying predicted age and gender
age_placeholder = st.empty()
gender_placeholder = st.empty()

# Main Streamlit app loop
st.title("Webcam Live Age and Gender Prediction")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

while run:
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Crop the face region
        face_roi = frame[y:y + h, x:x + w]

        # Perform age and gender prediction
        pred_age, pred_gender = predict_age_gender(face_roi)

        # Draw bounding box on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display predicted age and gender on the frame
        cv2.putText(frame, f"Age: {pred_age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Gender: {pred_gender}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with annotations
    FRAME_WINDOW.image(frame)

# Release the webcam
cap.release()
