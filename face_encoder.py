import cv2
import numpy as np

# --- Cleaned, robust mediapipe-based face encoder ---
import cv2
import numpy as np
import pickle
import os
import mediapipe as mp

ENCODINGS_PATH = "face_encodings.pickle"

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

def extract_face_embeddings(image):
    """Extract face embeddings using mediapipe face mesh landmarks."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_encodings = []
    face_locations = []
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        detection_results = face_detection.process(rgb)
        if not detection_results.detections:
            return [], []
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10, refine_landmarks=True) as face_mesh:
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
                # Use all 468 landmarks as embedding
                landmarks = mesh_results.multi_face_landmarks[0].landmark
                embedding = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
                # Normalize embedding
                embedding = embedding - np.mean(embedding)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                face_encodings.append(embedding)
                face_locations.append((y1, x2, y2, x1))
    return face_locations, face_encodings

def build_face_database():
    """Build face database from dataset folder using mediapipe embeddings."""
    dataset_dir = "dataset"
    all_encodings = {}
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' not found!")
        return
    print("Building face database with mediapipe embeddings...")
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        print(f"Processing images for {person_name}...")
        person_encodings = []
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            print(f"  Processing {img_path}...")
            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"    Warning: Could not read {img_path}")
                    continue
                _, face_encodings = extract_face_embeddings(image)
                for encoding in face_encodings:
                    person_encodings.append(encoding)
                    print(f"    Added face encoding for {person_name}")
            except Exception as e:
                print(f"    Error processing {img_path}: {str(e)}")
        all_encodings[person_name] = person_encodings
        print(f"  Total encodings for {person_name}: {len(person_encodings)}")
    # Balance encodings per person
    min_encodings = min(len(encs) for encs in all_encodings.values() if len(encs) > 0)
    print(f"Balancing: using {min_encodings} encodings per person.")
    known_encodings = []
    known_names = []
    for person_name, encodings in all_encodings.items():
        selected = encodings[:min_encodings]
        known_encodings.extend(selected)
        known_names.extend([person_name] * len(selected))
    if known_encodings:
        print(f"Saving {len(known_names)} face encodings to {ENCODINGS_PATH}")
        data = {"encodings": known_encodings, "names": known_names}
        with open(ENCODINGS_PATH, "wb") as f:
            pickle.dump(data, f)
        print("Face database created successfully!")
    else:
        print("No face encodings were created. Please check your images.")

def load_face_database():
    """Load the face database"""
    if not os.path.exists(ENCODINGS_PATH):
        return [], []
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

if __name__ == "__main__":
    build_face_database()
