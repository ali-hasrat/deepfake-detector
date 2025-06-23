import streamlit as st
import requests
import os
from PIL import Image

st.set_page_config(page_title="Deepfake Detector", page_icon="🎭", layout="centered")
st.title("🎭 Deepfake Detection App")
st.write("Upload an image or video to detect if it's a **Deepfake**. You'll get a confidence score and visual clues!")

option = st.radio("Choose input type:", ["Image", "Video"])

if option == "Image":
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file:
        with open(img_file.name, "wb") as f:
            f.write(img_file.read())

        st.image(img_file, caption="Uploaded Image", use_container_width=True)
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                response = requests.post("http://127.0.0.1:8000/analyze-image/", files={"file": open(img_file.name, "rb")})
            result = response.json()
            st.success(f"Prediction: **{result['label']}**")
            st.info(f"Confidence Score: **{result['confidence']}%**")
            st.image(result["heatmap_image"], caption="Grad-CAM Heatmap", use_container_width=True)

elif option == "Video":
    vid_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if vid_file:
        with open(vid_file.name, "wb") as f:
            f.write(vid_file.read())

        st.video(vid_file)
        if st.button("Analyze Video"):
            with st.spinner("Processing frames..."):
                response = requests.post("http://127.0.0.1:8000/analyze-video/", files={"file": open(vid_file.name, "rb")})
            result = response.json()
            st.success(f"Prediction: **{result['label']}**")
            st.info(f"Confidence Score: **{result['confidence']}% Fake** out of {result['frames_checked']} frames analyzed.")

# ---------- EDUCATIONAL SECTION ---------- #
with st.expander("📚 Learn About Deepfakes"):
    st.markdown("""
### What are Deepfakes?
Deepfakes are synthetic media where a person’s face, voice, or actions are manipulated using artificial intelligence (AI).

### Why Deepfakes are Dangerous:
- Can spread misinformation or fake news
- May damage someone's reputation
- Can be used in fraud or scams
- Pose a threat to democracy and personal safety

### How This App Works:
- Uses a deep learning model called **XceptionNet**
- Applies **Grad-CAM** to show manipulated regions in images
- For videos, detects faces and overlays **confidence scores**
- Predicts whether content is **Real** or **Fake**

### Accuracy & Limitations:
- This demo uses pretrained models with limited training data
- Accuracy improves with fine-tuned models on larger datasets
- Real-time webcam or audio-based detection is a future enhancement

### Future Scope:
- Audio-visual sync detection (lip movement + voice)
- Browser plugin for auto-checking online videos
- Report download for flagged content
""")