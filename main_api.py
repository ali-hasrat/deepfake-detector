from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from detector import predict_image, analyze_video
import shutil
import os
import cv2

app = FastAPI()

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    with open(file.filename, "wb") as f:
        shutil.copyfileobj(file.file, f)

    label, confidence, heatmap_img = predict_image(file.filename)
    result_path = f"heatmap_{file.filename}"
    cv2.imwrite(result_path, heatmap_img)

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "heatmap_image": result_path
    }


@app.post("/analyze-video/")
async def analyze_video_file(file: UploadFile = File(...)):
    with open(file.filename, "wb") as f:
        shutil.copyfileobj(file.file, f)

    label, percent_fake, total_frames = analyze_video(file.filename)
    return {
        "label": label,
        "confidence": percent_fake,
        "frames_checked": total_frames
    }
