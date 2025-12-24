import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import os
from PIL import Image

# --- PAGE SETUP ---
st.set_page_config(page_title="Deepfake Detector", layout="wide")

# --- MODEL PATHS ---
MODEL_PATH_EFFICIENT = os.path.join("models", "deepfake_detector_FINAL.weights.h5")
MODEL_PATH_V1 = os.path.join("models", "deepfake_detector_v1.h5")

# --- CONSTANTS ---
IMG_SIZE_EFFICIENT = (260, 260)
IMG_SIZE_V1 = (128, 128)

# --- BUILD MODELS ---
def build_efficientnet_model():
    base_model = tf.keras.applications.EfficientNetB2(
        weights="imagenet", include_top=False, input_shape=(260, 260, 3)
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(260, 260, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# --- CACHING (works with older Streamlit) ---
@st.cache(allow_output_mutation=True)
def load_model_and_detector(selected_model):
    if selected_model == "EfficientNetB2 (260x260)":
        model = build_efficientnet_model()
        model.load_weights(MODEL_PATH_EFFICIENT)
        input_size = IMG_SIZE_EFFICIENT
    else:
        model = tf.keras.models.load_model(MODEL_PATH_V1)
        input_size = IMG_SIZE_V1
    detector = MTCNN()
    return model, detector, input_size

# --- IMAGE PROCESSING ---
def process_image(image, model, detector, input_size, selected_model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    if not faces:
        st.warning("⚠️ No face detected.")
        return image

    for face_info in faces:
        x, y, w, h = face_info["box"]
        x, y = max(0, x), max(0, y)
        face = image_rgb[y:y+h, x:x+w]
        if face.size == 0:
            continue

        # Resize & preprocess
        face_resized = cv2.resize(face, input_size)
        face_batch = np.expand_dims(face_resized.astype("float32"), axis=0)
        if selected_model == "EfficientNetB2 (260x260)":
            face_batch = tf.keras.applications.efficientnet.preprocess_input(face_batch)
        else:
            face_batch = face_batch / 255.0

        # Prediction
        pred = float(model.predict(face_batch, verbose=0)[0][0])
        st.write(f"🔹 Raw model output: {pred:.4f}")

        # Corrected label logic —> model output > 0.5 = REAL
        if pred > 0.5:
            label = f"REAL ({pred*100:.1f}%)"
            color = (0, 255, 0)
        else:
            label = f"FAKE ({(1-pred)*100:.1f}%)"
            color = (0, 0, 255)

        # Draw results
        cv2.rectangle(image_rgb, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image_rgb, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

# --- STREAMLIT APP ---
st.title("🕵️ Deepfake Detector (Compare Models)")

# Model selector
selected_model = st.radio(
    "Select model for inference:",
    ["EfficientNetB2 (260x260)", "V1 Model (128x128)"]
)

model, detector, input_size = load_model_and_detector(selected_model)
st.success(f"✅ Loaded {selected_model}")

tab1, tab2 = st.tabs(["🖼️ Image Upload", "📸 Webcam Capture"])

# --- IMAGE UPLOAD TAB ---
with tab1:
    image_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if image_file:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, channels="BGR", caption="Original Image", use_column_width=True)
        st.write("🔍 Analyzing...")
        output = process_image(image, model, detector, input_size, selected_model)
        st.image(output, channels="BGR", caption="Prediction Complete", use_column_width=True)

# --- WEBCAM TAB ---
with tab2:
    camera_image = st.camera_input("Capture photo")
    if camera_image:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.write("🔍 Analyzing...")
        output = process_image(image, model, detector, input_size, selected_model)
        st.image(output, channels="BGR", caption="Prediction Complete", use_column_width=True)
