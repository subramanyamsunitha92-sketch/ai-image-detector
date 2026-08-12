import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import tempfile

st.set_page_config(page_title="AI Image/Video Detector", page_icon="🔍")
st.title("🔍 AI vs Real Detector")
st.write("Upload an IMAGE or VIDEO to check if it's REAL or AI-generated!")

model = load_model('ai_detector_model.h5')

def predict_image(img):
    img_resized = img.resize((32, 32))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)[0][0]
    return prediction

tab1, tab2 = st.tabs(["📷 Image", "🎥 Video"])

with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="img")
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', use_container_width=True)
        prediction = predict_image(img)
        if prediction > 0.5:
            st.success(f"✅ REAL Image (Confidence: {prediction*100:.2f}%)")
        else:
            st.error(f"🤖 AI-Generated Image (Confidence: {(1-prediction)*100:.2f}%)")

with tab2:
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"], key="vid")
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        st.video(uploaded_video)

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        real_count = 0
        fake_count = 0
        total_checked = 0

        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if fps > 0 and frame_count % int(fps) == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                pred = predict_image(pil_img)
                if pred > 0.5:
                    real_count += 1
                else:
                    fake_count += 1
                total_checked += 1

            frame_count += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))

        cap.release()

        if total_checked > 0:
            real_pct = (real_count / total_checked) * 100
            fake_pct = (fake_count / total_checked) * 100

            st.write(f"**Frames analyzed:** {total_checked}")

            if fake_pct > 50:
                st.error(f"🤖 Likely AI-GENERATED Video ({fake_pct:.1f}% of frames flagged as AI)")
            else:
                st.success(f"✅ Likely REAL Video ({real_pct:.1f}% of frames flagged as Real)")
        else:
            st.warning("Could not analyze video frames.")
