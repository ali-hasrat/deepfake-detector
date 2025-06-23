import os
import cv2
from model_utils import load_model, preprocess_image
from gradcam_utils import get_gradcam_heatmap, overlay_heatmap

def predict_image(img_path, model_name='resnet'):
    model, preprocess = load_model(model_name)
    img_array = preprocess_image(img_path, (224, 224), preprocess)
    preds = model.predict(img_array)
    confidence = float(preds.mean())  # Fake if confidence < threshold
    label = "Fake" if confidence < 0.35 else "Real"

    heatmap = get_gradcam_heatmap(model, img_array, model.layers[-3].name)
    result_image = overlay_heatmap(heatmap, img_path)

    return label, confidence, result_image

def analyze_video(video_path, skip=30):
    if not os.path.exists("frames"):
        os.mkdir("frames")
    cap = cv2.VideoCapture(video_path)
    count, frame_id, fake_frames = 0, 0, 0
    model, preprocess = load_model()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % skip == 0:
            fname = f"frames/frame_{frame_id}.jpg"
            cv2.imwrite(fname, frame)
            img_array = preprocess_image(fname, (224, 224), preprocess)
            pred = model.predict(img_array)
            if pred.mean() < 0.5:
                fake_frames += 1
            frame_id += 1
        count += 1

    cap.release()
    percent_fake = int((fake_frames / max(frame_id, 1)) * 100)
    label = "Fake" if percent_fake > 50 else "Real"
    return label, percent_fake, frame_id