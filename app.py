import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(page_title="AI Image Detector", page_icon="🔍")

st.title("🔍 AI vs Real Image Detector")
st.write("Upload an image and find out if it's REAL or AI-generated!")

model = load_model('ai_detector_model.h5')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)    
    img_resized = img.resize((32, 32))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    
    if prediction >0.5:
        st.success(f"✅ REAL Image (Confidence: {prediction*100:.2f}%)")
    else:
        st.error(f"🤖 AI-Generated Image (Confidence: {(1-prediction)*100:.2f}%)")
