from fastapi import FastAPI, File, UploadFile
import shutil
import uuid
import os

from detect import detect_boats_and_text

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.jpg")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    result = detect_boats_and_text(temp_path)
    return {"detections": result}
