import streamlit as st
import numpy as np
import cv2
import tempfile
import tensorflow as tf
from tensorflow.keras import layers, models

st.title("💓 Heart EF Prediction App")

IMG_SIZE = 112
MAX_FRAMES = 16

# 🔥 MODEL ARCHITECTURE (same as training)
def build_model():
    
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,   # ❗ important (weights later load honge)
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg"
    )
    
    inputs = layers.Input(shape=(MAX_FRAMES, IMG_SIZE, IMG_SIZE, 3))
    
    x = layers.TimeDistributed(base_model)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs, outputs)
    
    return model


# ✅ Load model + weights
@st.cache_resource
def load_my_model():
    model = build_model()
    model.load_weights("model.weights.h5")   # 👈 important
    return model

model = load_my_model()

# 🎥 Video processing
def load_video_fast(path):
    cap = cv2.VideoCapture(path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return None
    
    indices = np.linspace(0, total_frames - 1, MAX_FRAMES).astype(int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)
    
    cap.release()
    
    # padding
    while len(frames) < MAX_FRAMES:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))
    
    return np.array(frames)


# 📤 Upload video
uploaded_file = st.file_uploader("Upload Echo Video", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    # temp file save
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    video = load_video_fast(tfile.name)
    
    if video is not None:
        video = np.expand_dims(video, axis=0)
        
        # 🔮 Prediction
        pred = model.predict(video)
        pred = pred * 100   # de-normalize
        
        st.success(f"💓 Predicted EF: {pred[0][0]:.2f}")
    else:
        st.error("❌ Error processing video")
