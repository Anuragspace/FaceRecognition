
# Suppress TensorFlow/mediapipe warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import cv2
import numpy as np
import pickle
import os

ENCODINGS_PATH = "face_encodings.pickle" 
TOLERANCE = 0.6

def cosine_similarity_manual(a, b):
    """Manual cosine similarity calculation"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

def extract_face_features(image):
    """Extract face features using OpenCV face detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load face cascade classifier
    cascade_path = r"C:\Users\anura\Desktop\face recognition\backend\.venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")

# --- Cleaned, robust mediapipe-based real-time face recognition ---
import cv2
import numpy as np
import pickle
import os
import mediapipe as mp

ENCODINGS_PATH = "face_encodings.pickle"
TOLERANCE = 0.4  # Lower is stricter, adjust as needed

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_face_embeddings(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_encodings = []
    face_locations = []
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        detection_results = face_detection.process(rgb)
        if not detection_results.detections:
            return [], []
        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, refine_landmarks=True) as face_mesh:
            for det in detection_results.detections:
                bboxC = det.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x1 = int(bboxC.xmin * iw)
                y1 = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                x2 = x1 + w
                y2 = y1 + h
                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw, x2), min(ih, y2)
                face_roi = rgb[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue
                mesh_results = face_mesh.process(face_roi)
                if not mesh_results.multi_face_landmarks:
                    continue
                landmarks = mesh_results.multi_face_landmarks[0].landmark
                embedding = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
                embedding = embedding - np.mean(embedding)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                face_encodings.append(embedding)
                face_locations.append((y1, x2, y2, x1))  # top, right, bottom, left
    return face_locations, face_encodings

def load_face_database():
    if not os.path.exists(ENCODINGS_PATH):
        return [], []
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

def recognize_face(face_encoding, known_encodings, known_names, tolerance=TOLERANCE):
    if not known_encodings:
        return "Unknown"
    similarities = [cosine_similarity(face_encoding, ke) for ke in known_encodings]
    similarities = np.array(similarities)
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]
    # Cosine similarity: 1 is identical, 0 is orthogonal
    if best_similarity >= (1 - tolerance):
        return known_names[best_idx]
    else:
        return "Unknown"

def main():
    known_encodings, known_names = load_face_database()
    if not known_encodings:
        print("No face database found! Please run face_encoder.py first.")
        return
    print(f"Loaded {len(known_encodings)} face encodings for recognition.")
    print("Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    frame_count = 0
    process_every_n_frames = 2
    results = []
    prev_time = cv2.getTickCount()
    fps = 0

    # Attendance system: store recognized names with timestamps
    attendance = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        # Flip the frame horizontally to correct mirror image
        frame = cv2.flip(frame, 1)
        frame_count += 1
        if frame_count % process_every_n_frames == 0:
            face_locations, face_encodings = extract_face_embeddings(frame)
            results = []
            for (face_location, face_encoding) in zip(face_locations, face_encodings):
                name = recognize_face(face_encoding, known_encodings, known_names)
                results.append((face_location, name))
                # Attendance: mark time for recognized names (not Unknown)
                if name != "Unknown":
                    if name not in attendance:
                        from datetime import datetime
                        attendance[name] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for (face_location, name) in results:
            top, right, bottom, left = face_location
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = name
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (left, bottom - label_size[1] - 15), 
                         (left + label_size[0] + 10, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 5, bottom - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # FPS overlay
        curr_time = cv2.getTickCount()
        time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
        if time_diff > 0:
            fps = 1.0 / time_diff
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Face Recognition System', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # Press 's' to save attendance to CSV
        if key == ord('s'):
            import csv
            with open('attendance.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Timestamp'])
                for name, timestamp in attendance.items():
                    writer.writerow([name, timestamp])
            print("Attendance saved to attendance.csv")
    cap.release()
    cv2.destroyAllWindows()
    print("Face recognition system stopped.")
    # Optionally save attendance on exit
    if attendance:
        import csv
        with open('attendance.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Timestamp'])
            for name, timestamp in attendance.items():
                writer.writerow([name, timestamp])
        print("Attendance saved to attendance.csv")

if __name__ == "__main__":
    main()
