import cv2
import easyocr
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
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

        # Stage 1: Grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ocr_gray = reader.readtext(gray)
        texts_gray = [item[1] for item in ocr_gray if len(item[1]) > 1]

        # Stage 2: Blurred
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        ocr_blurred = reader.readtext(blurred)
        texts_blurred = [item[1] for item in ocr_blurred if len(item[1]) > 1]

        # Stage 3: Adaptive Threshold
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        ocr_thresh = reader.readtext(thresh)
        texts_thresh = [item[1] for item in ocr_thresh if len(item[1]) > 1]

        outputs.append({
            "bbox": [x1, y1, x2, y2],
            "label": label,
            "texts_gray": texts_gray,
            "texts_blurred": texts_blurred,
            "texts_thresh": texts_thresh
        })

    return outputs
