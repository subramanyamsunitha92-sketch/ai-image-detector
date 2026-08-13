import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64

st.set_page_config(page_title="AI Threat Detector", page_icon="🛡️", layout="centered")

def get_base64_video(video_path):
    with open(video_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

video_base64 = get_base64_video("background.mp4")

st.markdown(f"""
    <style>
    .stApp {{
        background-color: #0d1117;
    }}
    #bg-video {{
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        z-index: -1;
        object-fit: cover;
        opacity: 0.35;
    }}
    .title-text {{
        text-align: center;
        font-family: 'Courier New', monospace;
        font-size: 2.3rem;
        font-weight: 800;
        color: #39ff14;
        text-shadow: 0 0 10px #39ff14;
        margin-bottom: 0.3rem;
    }}
    .subtitle-text {{
        text-align: center;
        font-family: 'Courier New', monospace;
        color: #8b949e;
        font-size: 1rem;
        margin-bottom: 2rem;
    }}
    .result-box {{
        font-family: 'Courier New', monospace;
        padding: 1.3rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 1rem;
        border: 1px solid;
    }}
    .real-box {{
        background: rgba(10,31,10,0.85);
        color: #39ff14;
        border-color: #39ff14;
        box-shadow: 0 0 15px rgba(57,255,20,0.3);
    }}
    .fake-box {{
        background: rgba(42,10,10,0.85);
        color: #ff3131;
        border-color: #ff3131;
        box-shadow: 0 0 15px rgba(255,49,49,0.3);
    }}
    p, span, div, label {{
        color: #c9d1d9;
    }}
    </style>

    <video autoplay muted loop id="bg-video">
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
""", unsafe_allow_html=True)

st.markdown('<div class="title-text">🛡️ AI THREAT DETECTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">// Scan an image to verify authenticity — REAL or AI-GENERATED</div>', unsafe_allow_html=True)

model = load_model('ai_detector_model.h5')

uploaded_file = st.file_uploader("Upload image for analysis...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Target Image', use_container_width=True)

    img_resized = img.resize((32, 32))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner('🔎 Scanning pixels for anomalies...'):
        prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.markdown(f"""
            <div class="result-box real-box">
                ✅ VERIFIED: REAL IMAGE<br>
                <span style="font-size:0.9rem; font-weight:400;">Confidence: {prediction*100:.2f}%</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-box fake-box">
                ⚠️ ALERT: AI-GENERATED IMAGE<br>
                <span style="font-size:0.9rem; font-weight:400;">Confidence: {(1-prediction)*100:.2f}%</span>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="text-align:center; font-family:monospace; font-size:0.75rem; color:#484f58;">Powered by TensorFlow CNN | CIFAKE Dataset | v1.0</p>', unsafe_allow_html=True)
