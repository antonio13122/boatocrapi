import cv2
import easyocr
from ultralytics import YOLO
import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

# Initialize YOLO and OCR

model = YOLO("models/best.pt")

reader = easyocr.Reader(['en'], gpu=True)

print("Model class names:", model.names)

def detect_boats_and_text(image_path):
    results = model(image_path)[0]
    img = cv2.imread(image_path)
    outputs = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        roi = img[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        texts_gray = [t[1] for t in reader.readtext(gray) if len(t[1]) > 1]
        texts_blurred = [t[1] for t in reader.readtext(blurred) if len(t[1]) > 1]
        texts_thresh = [t[1] for t in reader.readtext(thresh) if len(t[1]) > 1]

        outputs.append({
            "bbox": [x1, y1, x2, y2],
            "label": label,
            "texts_gray": texts_gray,
            "texts_blurred": texts_blurred,
            "texts_thresh": texts_thresh
        })

    return outputs
