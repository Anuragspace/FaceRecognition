import cv2
import numpy as np
import pickle
import os
import mediapipe as mp
import mediapipe.python.solutions.face_detection as mp_face_detection_module
import mediapipe.python.solutions.face_mesh as mp_face_mesh_module
import logging

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Global persistent MediaPipe objects - CRITICAL OPTIMIZATION
mp_face_detection = None
mp_face_mesh = None

def initialize_mediapipe():
    """Initialize MediaPipe objects once - reuse across all calls"""
    global mp_face_detection, mp_face_mesh
    if mp_face_detection is None:
        print("Initializing MediaPipe models (one-time setup)...")
        mp_face_detection = mp_face_detection_module.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.3
        )
        mp_face_mesh = mp_face_mesh_module.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        print("MediaPipe initialization complete!")

def preprocess_image(image):
    """Enhanced image preprocessing for better face detection"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = image
    
    processed_images = []
    processed_images.append(("original", rgb))
    
    # Histogram equalization
    if len(rgb.shape) == 3:
        yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        processed_images.append(("equalized", equalized))
    
    # Gamma correction
    gamma_corrected = np.power(rgb / 255.0, 0.7) * 255.0
    gamma_corrected = gamma_corrected.astype(np.uint8)
    processed_images.append(("gamma_corrected", gamma_corrected))
    
    # Scale up small images
    h, w = rgb.shape[:2]
    if h < 480 or w < 640:
        scale_factor = max(480/h, 640/w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        scaled = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        processed_images.append(("scaled", scaled))
    
    return processed_images

def face_quality_check(face_img):
    """Enhanced face quality check"""
    if face_img.size == 0:
        return False, "Empty image"
    
    h, w = face_img.shape[:2]
    if h < 64 or w < 64:
        return False, f"Too small: {w}x{h}"
    
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY) if len(face_img.shape) == 3 else face_img
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < 50:
        return False, f"Too blurry: {laplacian_var:.1f}"
    
    mean_brightness = np.mean(gray)
    if mean_brightness < 20 or mean_brightness > 235:
        return False, f"Poor lighting: {mean_brightness:.1f}"
    
    return True, f"Good quality: {w}x{h}, blur: {laplacian_var:.1f}, brightness: {mean_brightness:.1f}"

def align_face(face_img, landmarks):
    """Enhanced face alignment using eye landmarks"""
    try:
        h, w = face_img.shape[:2]
        left_eye_points = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_points = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        left_eye_center = [0, 0]
        right_eye_center = [0, 0]
        
        for point_idx in left_eye_points:
            if point_idx < len(landmarks.landmark):
                x = int(landmarks.landmark[point_idx].x * w)
                y = int(landmarks.landmark[point_idx].y * h)
                left_eye_center[0] += x
                left_eye_center[1] += y
        
        for point_idx in right_eye_points:
            if point_idx < len(landmarks.landmark):
                x = int(landmarks.landmark[point_idx].x * w)
                y = int(landmarks.landmark[point_idx].y * h)
                right_eye_center[0] += x
                right_eye_center[1] += y
        
        left_eye_center[0] //= len(left_eye_points)
        left_eye_center[1] //= len(left_eye_points)
        right_eye_center[0] //= len(right_eye_points)
        right_eye_center[1] //= len(right_eye_points)
        
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        center = ((left_eye_center[0] + right_eye_center[0]) // 2, 
                 (left_eye_center[1] + right_eye_center[1]) // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(face_img, M, (w, h))
        aligned = cv2.resize(aligned, (224, 224))
        return aligned
        
    except Exception as e:
        return cv2.resize(face_img, (224, 224))

def extract_robust_features(aligned_face):
    """Enhanced feature extraction"""
    if len(aligned_face.shape) == 3:
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2GRAY)
    else:
        gray = aligned_face
    
    features = []
    
    # Multi-scale features
    for scale in [8, 16, 32, 64]:
        resized = cv2.resize(gray, (scale, scale))
        features.extend(resized.flatten())
    
    # Gradient features
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    features.extend(cv2.resize(magnitude, (16, 16)).flatten())
    
    # Statistical features
    features.extend([
        np.mean(gray), np.std(gray), np.median(gray),
        np.percentile(gray, 10), np.percentile(gray, 25), 
        np.percentile(gray, 75), np.percentile(gray, 90),
        np.min(gray), np.max(gray)
    ])
    
    # Texture features
    for ksize in [3, 5, 7]:
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        features.extend(cv2.resize(blurred, (8, 8)).flatten())
    
    feature_vector = np.array(features, dtype=np.float32)
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector = feature_vector / norm
    
    return feature_vector

def extract_face_embeddings(image):
    """Main function for face embedding extraction (renamed for consistency)"""
    initialize_mediapipe()
    
    processed_images = preprocess_image(image)
    best_results = ([], [])
    best_count = 0
    
    for preprocess_name, processed_img in processed_images:
        face_encodings = []
        face_locations = []
        
        if mp_face_detection is None:
            print("Error: MediaPipe face detection model is not initialized.")
            continue
            
        detection_results = mp_face_detection.process(processed_img)
        detections = getattr(detection_results, "detections", None)
        if not detections:
            continue

        print(f"    🔍 {preprocess_name}: Found {len(detections)} face(s)")

        for detection in detections:
            bbox = detection.location_data.relative_bounding_box
            h, w = processed_img.shape[:2]

            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))

            padding = int(min(x2-x1, y2-y1) * 0.2)
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            face_img = processed_img[y1:y2, x1:x2]

            is_good, quality_msg = face_quality_check(face_img)
            print(f"      📊 Quality: {quality_msg}")

            if not is_good:
                continue

            if mp_face_mesh is not None:
                mesh_results = mp_face_mesh.process(face_img)
            else:
                mesh_results = None

            multi_face_landmarks = getattr(mesh_results, "multi_face_landmarks", None) if mesh_results is not None else None
            if multi_face_landmarks and len(multi_face_landmarks) > 0:
                aligned_face = align_face(face_img, multi_face_landmarks[0])
            else:
                aligned_face = cv2.resize(face_img, (224, 224))

            features = extract_robust_features(aligned_face)
            face_encodings.append(features)
            face_locations.append((y1, x2, y2, x1))
        
        if len(face_encodings) > best_count:
            best_results = (face_locations, face_encodings)
            best_count = len(face_encodings)
    
    return best_results

def save_encodings_optimized(encodings_dict, filepath="face_encodings.npz"):
    """Save encodings as optimized numpy format"""
    if not encodings_dict:
        print("No encodings to save!")
        return
    
    all_encodings = []
    all_names = []
    
    for name, encodings_list in encodings_dict.items():
        for encoding in encodings_list:
            all_encodings.append(encoding)
            all_names.append(name)
    
    encodings_array = np.stack(all_encodings).astype(np.float32)
    names_array = np.array(all_names, dtype=object)
    
    np.savez_compressed(filepath, encodings=encodings_array, names=names_array)
    print(f"💾 Saved {len(all_encodings)} encodings to {filepath}")
    print(f"📐 Encoding shape: {encodings_array.shape}")

def build_face_database():
    """Build face database (main entry point)"""
    dataset_dir = "dataset"
    all_encodings = {}
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' not found!")
        return
    
    print("Building enhanced face database...")
    initialize_mediapipe()
    
    total_processed = 0
    total_faces_found = 0
    
    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        
        print(f"\n👤 Processing {person_name}...")
        person_encodings = []
        
        image_files = [f for f in os.listdir(person_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        
        for image_file in image_files:
            image_path = os.path.join(person_path, image_file)
            print(f"  📷 Processing {image_file}...")
            
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"    ❌ Could not read {image_file}")
                    continue
                
                h, w = image.shape[:2]
                print(f"    📏 Image size: {w}x{h}")
                
                face_locations, face_encodings = extract_face_embeddings(image)
                
                if face_encodings:
                    person_encodings.extend(face_encodings)
                    total_faces_found += len(face_encodings)
                    print(f"    ✅ Success: {len(face_encodings)} face(s) encoded")
                else:
                    print(f"    ⚠️ No faces detected in {image_file}")
                
                total_processed += 1
                
            except Exception as e:
                print(f"    💥 Error processing {image_file}: {e}")
        
        if person_encodings:
            all_encodings[person_name] = person_encodings
            print(f"  📊 Total encodings for {person_name}: {len(person_encodings)}")
        else:
            print(f"  ❌ No encodings generated for {person_name}")
    
    if all_encodings:
        save_encodings_optimized(all_encodings, "face_encodings.npz")
        
        # Save in pickle format for compatibility
        flattened = {"encodings": [], "names": []}
        for name, encodings_list in all_encodings.items():
            flattened["encodings"].extend(encodings_list)
            flattened["names"].extend([name] * len(encodings_list))
        
        with open("face_encodings.pickle", "wb") as f:
            pickle.dump(flattened, f)
        
        print(f"\n🎉 Enhanced database built successfully!")
        print(f"📊 Total people: {len(all_encodings)}")
        print(f"📊 Total encodings: {sum(len(encs) for encs in all_encodings.values())}")
        print(f"📊 Images processed: {total_processed}")
        print(f"📊 Faces found: {total_faces_found}")
        if total_processed > 0:
            print(f"📊 Success rate: {(total_faces_found/total_processed)*100:.1f}%")
    else:
        print("❌ No face encodings generated!")

def cleanup_mediapipe():
    """Clean up MediaPipe resources"""
    global mp_face_detection, mp_face_mesh
    if mp_face_detection:
        mp_face_detection.close()
    if mp_face_mesh:
        mp_face_mesh.close()
    mp_face_detection = None
    mp_face_mesh = None

if __name__ == "__main__":
    build_face_database()
    cleanup_mediapipe()
