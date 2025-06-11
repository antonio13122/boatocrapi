from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
from collections import Counter
import re

from detect import detect_boats_and_text
from video_utils import process_video, capture_from_webcam
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.jpg")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    result = detect_boats_and_text(temp_path)
    return {"detections": result}

@app.post("/detect_video")
async def detect_video(video: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.mp4")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    output_path = temp_path.replace(".mp4", "_processed.avi")
    processed_path, detections = process_video(temp_path, output_path)

    
    all_texts = []
    for frame in detections:
        for det in frame["detections"]:
            for text in det["texts"]:
                clean_text = text.strip().upper()
                if re.fullmatch(r"[A-Z0-9]{3,}", clean_text):  # e.g., AB77
                    all_texts.append(clean_text)

    most_common = Counter(all_texts).most_common(1)
    best_match = most_common[0][0] if most_common else None

    return {
        "message": "Video processed successfully",
        "output_path": processed_path,
        "most_likely_boat_name": best_match
    }



@app.post("/detect_webcam")
async def detect_webcam():
    output_path = os.path.join(UPLOAD_DIR, f"webcam_{uuid.uuid4().hex}.avi")
    processed_path, detected_names = capture_from_webcam(output_path)
    return {
        "message": "Webcam capture complete",
        "output_path": processed_path,
        "detected_boat_names": detected_names
    }
