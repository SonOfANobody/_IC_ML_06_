import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import tensorflow as tf

# Load model once and cache it to save memory
@st.cache_resource
def load_gesture_model():
    return tf.keras.models.load_model('hand_gesture_model_99.h5')

model = load_gesture_model()
class_names = ['Palm', 'L', 'Fist', 'Fist Moved', 'Thumb', 'Index', 'OK', 'Palm Moved', 'C', 'Down']

class GestureProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Preprocessing (Mirror flip + ROI)
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        roi_side = 250
        x1, y1 = (w // 2 - roi_side // 2), (h // 2 - roi_side // 2)
        
        # Prepare for Model
        roi = img[y1:y1+roi_side, x1:x1+roi_side]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        normalized = resized.astype('float32') / 255.0
        input_data = np.expand_dims(normalized, axis=(0, -1))

        # Predict
        prediction = model.predict(input_data, verbose=0)
        label = class_names[np.argmax(prediction)]
        
        # Draw on Screen
        cv2.rectangle(img, (x1, y1), (x1+roi_side, y1+roi_side), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🖐️ Real-Time Gesture AI")
webrtc_streamer(key="gesture", video_processor_factory=GestureProcessor)