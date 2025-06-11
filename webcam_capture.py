import cv2
from detect import detect_boats_and_text

def main():
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #single frame
        temp_path = "temp_webcam_frame.jpg"
        cv2.imwrite(temp_path, frame)

        #detection run
        results = detect_boats_and_text(temp_path)

       
        for item in results:
            x1, y1, x2, y2 = item["bbox"]
            label = item["label"]
            texts = item["texts_thresh"]  

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
