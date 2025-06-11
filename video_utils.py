import cv2
import os
from detect import detect_boats_and_text


def process_video(video_path, output_path="output_video.avi", skip_frames=10):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    resize_width, resize_height = 640, 384
    frame_id = 0

    all_detections = []  #collectin ocr

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_data = {
            "frame_id": frame_id,
            "detections": []
        }

        if frame_id % skip_frames == 0:
            resized_frame = cv2.resize(frame, (resize_width, resize_height))
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, resized_frame)
            detections = detect_boats_and_text(temp_path)
            os.remove(temp_path)

            for item in detections:
                x1, y1, x2, y2 = item["bbox"]
                scale_x = width / resize_width
                scale_y = height / resize_height
                x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

                label = item["label"]
                texts_all = item["texts_gray"] + item["texts_blurred"] + item["texts_thresh"]
                texts = list(set([t for t in texts_all if len(t) > 1]))

                frame_data["detections"].append({
                    "label": label,
                    "texts": texts,
                    "bbox": [x1, y1, x2, y2]
                })

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                for i, txt in enumerate(texts[:2]):
                    text_pos = (x1, min(height - 10, y2 + 20 + i * 20))
                    cv2.putText(frame, txt, text_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if frame_data["detections"]:
            all_detections.append(frame_data)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

    return output_path, all_detections



def capture_from_webcam(output_path="output_webcam.avi", frame_count=100):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam.")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = 10  

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    count = 0
    detected_names = set() 

    while count < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        temp_path = "temp_webcam.jpg"
        cv2.imwrite(temp_path, frame)
        detections = detect_boats_and_text(temp_path)
        os.remove(temp_path)

        for item in detections:
            x1, y1, x2, y2 = item["bbox"]
            label = item["label"]
            texts = item.get("texts_thresh", [])

           
            for txt in texts:
                if txt.strip():
                    detected_names.add(txt.strip())

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for i, txt in enumerate(texts[:2]):
                cv2.putText(frame, txt, (x1, y2 + 20 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Webcam Detection", frame)
        out.write(frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_path, list(detected_names)



