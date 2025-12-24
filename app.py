import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import os
from PIL import Image

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Deepfake Detector Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    local_css("style.css")
except FileNotFoundError:
    pass # CSS not found, continue with default

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

# --- CACHING ---
@st.cache_resource
def load_resources(selected_model_name):
    """Loads the model and the face detector."""
    try:
        detector = MTCNN()
    except Exception as e:
        st.error(f"Failed to load Face Detector: {e}")
        return None, None, None

    if selected_model_name == "EfficientNetB2 Pro":
        input_size = IMG_SIZE_EFFICIENT
        try:
            model = build_efficientnet_model()
            model.load_weights(MODEL_PATH_EFFICIENT)
        except Exception as e:
            st.error(f"Failed to load EfficientNet: {e}")
            return None, None, None
    else:
        input_size = IMG_SIZE_V1
        try:
            model = tf.keras.models.load_model(MODEL_PATH_V1)
        except Exception as e:
            st.error(f"Failed to load V1 Model: {e}")
            return None, None, None

    return model, detector, input_size

# --- ANALYSIS LOGIC ---
def analyze_image(image, model, detector, input_size, model_type, threshold=0.5):
    """
    Analyzes an image for deepfakes.
    Returns:
        processed_image: Image with bounding boxes
        results: List of dicts with details for each face
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    try:
        faces = detector.detect_faces(image_rgb)
    except Exception as e:
        st.error(f"Face detection error: {e}")
        return image, []
        
    if not faces:
        return image, []

    results = []
    
    for face_info in faces:
        x, y, w, h = face_info["box"]
        # Boundary checks
        x, y = max(0, x), max(0, y)
        face = image_rgb[y:y+h, x:x+w]
        
        if face.size == 0:
            continue

        # Preprocess
        try:
            face_resized = cv2.resize(face, input_size)
        except Exception:
            continue
            
        face_batch = np.expand_dims(face_resized.astype("float32"), axis=0)
        
        if model_type == "EfficientNetB2 Pro":
            face_batch = tf.keras.applications.efficientnet.preprocess_input(face_batch)
        else:
            face_batch = face_batch / 255.0

        # Predict
        try:
            pred = float(model.predict(face_batch, verbose=0)[0][0])
        except Exception as e:
            st.warning(f"Prediction failed for a face: {e}")
            continue
            
        is_real = pred > threshold
        confidence = pred if is_real else (1 - pred)
        label = "REAL" if is_real else "FAKE"
        color = (0, 255, 0) if is_real else (255, 0, 0) # Green vs Red

        results.append({
            "label": label,
            "confidence": confidence,
            "raw_pred": pred,
            "box": (x, y, w, h)
        })

        # Draw on image
        cv2.rectangle(image_rgb, (x, y), (x+w, y+h), color, 3)
        label_text = f"{label} {confidence*100:.0f}%"
        cv2.putText(image_rgb, label_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), results

# --- SIDEBAR ---
# --- SIDEBAR ---
with st.sidebar:
    # Removed potential broken image, using emoji header
    st.header("🕵️‍♂️ Deepfake Detection Tool")
    
    st.markdown("### 👨‍💻 Developed By: **Sagar**")
    
    # specific links requested by User
    st.markdown(
        """
        <div style="display: flex; gap: 10px;">
            <a href="https://github.com/sagar-grv" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-Profile-181717?style=flat&logo=github" alt="GitHub"/>
            </a>
            <a href="https://www.linkedin.com/in/sagargrv/" target="_blank">
                <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin" alt="LinkedIn"/>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    tab_settings, tab_info = st.tabs(["⚙️ Settings", "ℹ️ Model Info"])
    
    with tab_settings:
        st.markdown("### � AI Engine Selection")
        model_choice = st.selectbox(
            "Choose Model",
            ["EfficientNetB2 Pro", "V1 Basic Model"],
            help="Select the underlying neural network architecture."
        )
        
        st.markdown("### 🎚️ Detection Sensitivity")
        threshold = st.slider("Threshold (Real vs Fake)", 0.0, 1.0, 0.5, 0.05)
        st.caption(f"Current: {threshold} (Values > {threshold} are Real)")
        
        show_details = st.checkbox("Show Raw Confidence Scores", value=True)

    with tab_info:
        st.markdown("### 1️⃣ EfficientNetB2 Pro")
        st.info(
            "**Architecture**: EfficientNetB2 (Transfer Learning)\n\n"
            "**Specialization**: \n"
            "- High-frequency feature extraction.\n"
            "- Detects subtle texture inconsistencies & blending artifacts.\n"
            "- Best for: **High Accuracy Analysis**.\n\n"
            "**Params**: ~9.2 Million"
        )
        
        st.markdown("### 2️⃣ V1 Basic Model")
        st.warning(
            "**Architecture**: Custom Sequential CNN\n\n"
            "**Specialization**: \n"
            "- Lightweight & Fast inference.\n"
            "- Focuses on structural facial anomalies.\n"
            "- Best for: **Real-time / Low-resource**.\n\n"
            "**Params**: ~2.1 Million"
        )

    st.markdown("---")
    st.caption(f"System Status: {'🟢 GPU Active' if tf.config.list_physical_devices('GPU') else '🟠 CPU Mode'}")

# --- MAIN PAGE ---
st.markdown("<h1>🛡️ AI Deepfake Detector</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: #aaaaaa; margin-bottom: 30px;'>
    Advanced Forensic Analysis for Digital Media Integrity<br>
    <span style="font-size: 0.8em;">(Basic Image Recognition Project)</span>
    </div>
    """, 
    unsafe_allow_html=True
)

# Load Model
with st.spinner("Initializing AI Models..."):
    model, detector, input_size = load_resources(model_choice)

if not model:
    st.error("System failed to initialize. Please check logs.")
    st.stop()

# Tabs
tab_upload, tab_cam = st.tabs(["� Upload Image", "� Live Webcam"])

image = None
analyze_trigger = False

with tab_upload:
    uploaded_file = st.file_uploader("Drop your image here", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        analyze_trigger = True

with tab_cam:
    cam_input = st.camera_input("Take a photo")
    if cam_input:
        file_bytes = np.asarray(bytearray(cam_input.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        analyze_trigger = True

# Analysis Section
if analyze_trigger and image is not None:
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Input Header")
        st.image(image, channels="BGR", use_column_width=True)

    with col2:
        st.markdown("### 🔍 Analysis Status")
        with st.spinner("Scanning for anomalies..."):
            processed_img, results = analyze_image(
                image, model, detector, input_size, model_choice, threshold
            )
        
        if not results:
            st.warning("No faces detected in the image.")
        else:
            # Display results for each face
            for i, res in enumerate(results):
                st.markdown(f"#### Face #{i+1}")
                lab = res['label']
                conf = res['confidence']
                
                if lab == "REAL":
                    st.success(f"**AUTHENTIC** ({conf*100:.1f}% Confidence)")
                    st.progress(conf)
                else:
                    st.error(f"**DEEPFAKE DETECTED** ({conf*100:.1f}% Confidence)")
                    st.progress(conf)
                    
                with st.expander("See technical details"):
                    st.json(res)

    st.markdown("### 🖼️ Forensic Output")
    st.image(processed_img, channels="BGR", use_column_width=True, caption="Analyzed Regions")

else:
    if not image:
        st.info("Waiting for image input...")
