import cv2
import numpy as np
import pickle
import os
import mediapipe as mp
from mediapipe.python.solutions.face_detection import FaceDetection
from mediapipe.python.solutions.face_mesh import FaceMesh
import logging
from datetime import datetime
import threading
import queue
from collections import deque, Counter

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

ENCODINGS_PATH = "face_encodings.npz"
TOLERANCE = 0.5  # Cosine similarity threshold

# Global persistent MediaPipe objects
mp_face_detection = None
mp_face_mesh = None

# Known faces
known_names = []
known_encodings = []

# Temporal smoothing
face_tracks = {}
track_id_counter = 0  # Initialize track ID counter

def initialize_mediapipe():
    """Initialize MediaPipe objects once"""
    global mp_face_detection, mp_face_mesh
    if mp_face_detection is None:
        mp_face_detection = FaceDetection(
            model_selection=0,
            min_detection_confidence=0.3
        )
        mp_face_mesh = FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

def load_face_database():
    """Load face database"""
    global known_names, known_encodings
    
    try:
        if os.path.exists(ENCODINGS_PATH):
            data = np.load(ENCODINGS_PATH, allow_pickle=True)
            encodings = data["encodings"]
            names = data["names"].tolist()
        else:
            with open("face_encodings.pickle", "rb") as f:
                data = pickle.load(f)
            encodings = np.array(data["encodings"])
            names = data["names"]
        
        if len(encodings) == 0:
            return [], []
        
        # Normalize encodings for cosine similarity
        norms = np.linalg.norm(encodings, axis=1, keepdims=True)
        known_encodings = encodings / (norms + 1e-10)
        known_names = names
        
        print(f"Loaded {len(encodings)} face encodings")
        return encodings.tolist(), names
    except Exception as e:
        print(f"Error loading face database: {e}")
        return [], []

def recognize_face_optimized(face_encoding):
    """Fast face recognition using vectorized cosine similarity"""
    if len(known_encodings) == 0:
        return "Unknown"
    
    query_norm = np.linalg.norm(face_encoding)
    if query_norm == 0:
        return "Unknown"
    
    query_normalized = face_encoding / query_norm
    similarities = np.dot(known_encodings, query_normalized)
    
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]
    
    if best_similarity >= TOLERANCE:
        return known_names[best_idx]
    else:
        return "Unknown"

def track_faces(face_locations, names):
    """Simple face tracking for temporal smoothing"""
    global face_tracks, track_id_counter

    current_time = datetime.now()
    matched_tracks = set()
    results = []
    
    for i, (location, name) in enumerate(zip(face_locations, names)):
        y1, x2, y2, x1 = location
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        best_track = None
        min_distance = float('inf')
        
        for track_id, track_data in face_tracks.items():
            if track_id in matched_tracks:
                continue
                
            last_center = track_data['last_center']
            distance = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
            
            if distance < 100 and distance < min_distance:
                min_distance = distance
                best_track = track_id
        
        if best_track is not None:
            matched_tracks.add(best_track)
            face_tracks[best_track]['names'].append(name)
            face_tracks[best_track]['last_center'] = center
            face_tracks[best_track]['last_time'] = current_time
            
            if len(face_tracks[best_track]['names']) > 8:
                face_tracks[best_track]['names'] = face_tracks[best_track]['names'][-8:]
            
            name_counter = Counter(face_tracks[best_track]['names'])
            smoothed_name = name_counter.most_common(1)[0][0]
            results.append((location, smoothed_name))
        else:
            track_id_counter += 1
            face_tracks[track_id_counter] = {
                'names': [name],
                'last_center': center,
                'last_time': current_time
            }
            matched_tracks.add(track_id_counter)
            results.append((location, name))
    
    # Remove old tracks
    expired_tracks = []
    for track_id, track_data in face_tracks.items():
        if (current_time - track_data['last_time']).total_seconds() > 1.5:
            expired_tracks.append(track_id)
    
    for track_id in expired_tracks:
        del face_tracks[track_id]
    
    return results

class FrameProcessor:
    """Threaded frame processor for better performance"""
    def __init__(self):
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=5)
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.running = True
        
    def start(self):
        self.processing_thread.start()
    
    def stop(self):
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join()
    
    def add_frame(self, frame):
        try:
            self.input_queue.put_nowait(frame)
        except queue.Full:
            pass
    
    def get_result(self):
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _process_frames(self):
        from face_encoder import extract_face_embeddings
        
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
                
                face_locations, face_encodings = extract_face_embeddings(frame)
                
                names = []
                for encoding in face_encodings:
                    name = recognize_face_optimized(encoding)
                    names.append(name)
                
                results = track_faces(face_locations, names)
                self.output_queue.put((results, len(face_locations)))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")

def main():
    """Main function for testing"""
    encodings, names = load_face_database()
    if not encodings:
        print("No face database found! Please run face_encoder.py first.")
        return
    
    initialize_mediapipe()
    
    print(f"Loaded {len(encodings)} face encodings for recognition.")
    print("Press 'q' to quit.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Pre-warm camera
    for _ in range(5):
        cap.read()
    
    processor = FrameProcessor()
    processor.start()
    
    frame_count = 0
    process_every_n_frames = 2
    results = []
    people_count = 0
    
    fps_counter = deque(maxlen=20)
    prev_time = cv2.getTickCount()
    attendance = {}
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            if frame_count % process_every_n_frames == 0:
                processor.add_frame(frame.copy())
            
            result = processor.get_result()
            if result:
                results, people_count = result
                
                for (face_location, name) in results:
                    if name != "Unknown":
                        if name not in attendance:
                            attendance[name] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculate FPS
            current_time = cv2.getTickCount()
            time_diff = (current_time - prev_time) / cv2.getTickFrequency()
            if time_diff > 0:
                fps_counter.append(1.0 / time_diff)
            fps = sum(fps_counter) / len(fps_counter) if fps_counter else 0
            prev_time = current_time
            
            # Draw results
            for (face_location, name) in results:
                y1, x2, y2, x1 = face_location
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y2 - 30), (x2, y2), color, cv2.FILLED)
                cv2.putText(frame, name, (x1 + 5, y2 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw info overlay
            cv2.putText(frame, f"People: {people_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, f"Recognized: {len(attendance)}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        processor.stop()
        cap.release()
        cv2.destroyAllWindows()
        
        if attendance:
            import csv
            with open('attendance.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Timestamp'])
                for name, timestamp in attendance.items():
                    writer.writerow([name, timestamp])
            print(f"Attendance saved with {len(attendance)} entries.")

if __name__ == "__main__":
    main()
