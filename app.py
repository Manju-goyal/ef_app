import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tempfile

st.title("💓 Heart EF Prediction App")

# Load model
@st.cache_resource
def load_my_model():
    return load_model("final_ef_model.keras", compile=False)

model = load_my_model()

# Video processing
IMG_SIZE = 112
MAX_FRAMES = 16

def load_video_fast(path):
    cap = cv2.VideoCapture(path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return None
    
    indices = np.linspace(0, total_frames-1, MAX_FRAMES).astype(int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)
    
    cap.release()
    
    while len(frames) < MAX_FRAMES:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))
    
    return np.array(frames)

# Upload video
uploaded_file = st.file_uploader("Upload Echo Video", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Process video
    video = load_video_fast(tfile.name)
    
    if video is not None:
        video = np.expand_dims(video, axis=0)
        
        pred = model.predict(video)
        pred = pred * 100  # de-normalize
        
        st.success(f"Predicted EF: {pred[0][0]:.2f}")
    else:
        st.error("Error processing video")