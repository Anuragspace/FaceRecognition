import cv2
import numpy as np
import pickle
import os
from face_encoder import extract_face_embeddings

ENCODINGS_PATH = "face_encodings.pickle"
TOLERANCE = 0.5  # Same as realtime system

def load_face_database():
    """Load the face database and normalize encodings"""
    if not os.path.exists(ENCODINGS_PATH):
        return [], []
    
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    
    encodings = np.array(data["encodings"])
    names = data["names"]
    
    if len(encodings) == 0:
        return [], []
    
    # Normalize encodings for cosine similarity (matching realtime system)
    norms = np.linalg.norm(encodings, axis=1, keepdims=True)
    encodings = encodings / (norms + 1e-10)
    
    return encodings.tolist(), names

def recognize_face(face_encoding, known_encodings, known_names, tolerance=TOLERANCE):
    """Recognize a face using cosine similarity (matching realtime system)"""
    if not known_encodings:
        return "Unknown"
    
    # Normalize the query encoding
    query_norm = np.linalg.norm(face_encoding)
    if query_norm == 0:
        return "Unknown"
    
    query_normalized = face_encoding / query_norm
    
    # Normalize known encodings and compute similarities
    known_encodings_array = np.array(known_encodings)
    known_norms = np.linalg.norm(known_encodings_array, axis=1)
    
    # Avoid division by zero
    valid_indices = known_norms > 0
    if not np.any(valid_indices):
        return "Unknown"
    
    known_normalized = known_encodings_array[valid_indices] / known_norms[valid_indices, np.newaxis]
    similarities = np.dot(known_normalized, query_normalized)
    
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]
    
    if best_similarity >= tolerance:
        # Map back to original indices
        valid_names = [known_names[i] for i in range(len(known_names)) if valid_indices[i]]
        return valid_names[best_idx]
    else:
        return "Unknown"

def test_model_accuracy():
    """Test the model accuracy using the dataset images"""
    known_encodings, known_names = load_face_database()
    
    if not known_encodings:
        print("❌ No face database found! Please run face_encoder.py first.")
        return
    
    print(f"📊 Loaded {len(known_encodings)} face encodings for {len(set(known_names))} people")
    print(f"🎯 Using tolerance: {TOLERANCE}")
    
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory '{dataset_dir}' not found!")
        return
    
    total_tests = 0
    correct_predictions = 0
    results = {}
    
    print("\nTesting Model Accuracy...")
    print("=" * 60)
    
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        person_correct = 0
        person_total = 0
        
        print(f"\nTesting images for {person_name}:")
        
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            
            try:
                # Read image
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Extract face embeddings using current system
                face_locations, face_encodings = extract_face_embeddings(image)
                
                if not face_encodings:
                    print(f"  {img_name}: No faces detected")
                    continue
                
                # Test each detected face encoding
                for j, face_encoding in enumerate(face_encodings):
                    predicted_name = recognize_face(face_encoding, known_encodings, known_names)
                    
                    total_tests += 1
                    person_total += 1
                    
                    if predicted_name == person_name:
                        correct_predictions += 1
                        person_correct += 1
                        status = "✓ CORRECT"
                    else:
                        status = f"✗ WRONG (predicted: {predicted_name})"
                    
                    face_info = f" (face {j+1})" if len(face_encodings) > 1 else ""
                    print(f"  {img_name}{face_info}: {status}")
                    
            except Exception as e:
                print(f"  Error testing {img_path}: {str(e)}")
        
        if person_total > 0:
            person_accuracy = (person_correct / person_total) * 100
            results[person_name] = {
                'correct': person_correct,
                'total': person_total,
                'accuracy': person_accuracy
            }
            print(f"  {person_name} Accuracy: {person_accuracy:.1f}% ({person_correct}/{person_total})")
    
    # Overall accuracy
    if total_tests > 0:
        overall_accuracy = (correct_predictions / total_tests) * 100
        
        print("\n" + "=" * 60)
        print("🎯 MODEL ACCURACY RESULTS")
        print("=" * 60)
        
        print(f"📈 Overall Accuracy: {overall_accuracy:.1f}% ({correct_predictions}/{total_tests})")
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Correct Predictions: {correct_predictions}")
        print(f"❌ Wrong Predictions: {total_tests - correct_predictions}")
        print(f"🎯 Tolerance Used: {TOLERANCE}")
        
        print("\n📋 Per-Person Results:")
        for person, result in results.items():
            status_emoji = "🟢" if result['accuracy'] >= 80 else "🟡" if result['accuracy'] >= 60 else "🔴"
            print(f"  {status_emoji} {person}: {result['accuracy']:.1f}% ({result['correct']}/{result['total']})")
        
        # Performance analysis
        print("\n🔍 Performance Analysis:")
        if overall_accuracy >= 90:
            print("🟢 EXCELLENT - Model is performing very well!")
        elif overall_accuracy >= 80:
            print("🟡 GOOD - Model is performing well")
        elif overall_accuracy >= 70:
            print("🟠 FAIR - Model needs some improvement")
        else:
            print("🔴 POOR - Model needs significant improvement")
            
        print("\n💡 Recommendations:")
        if overall_accuracy < 90:
            print("  📸 Add more training images with different angles and lighting")
            print("  💡 Ensure images have clear, well-lit faces")
            print("  🧹 Remove blurry or low-quality images from dataset")
            print(f"  ⚙️ Consider adjusting TOLERANCE (current: {TOLERANCE})")
            
            # Specific recommendations based on results
            low_performers = [name for name, result in results.items() if result['accuracy'] < 70]
            if low_performers:
                print(f"  🎯 Focus on improving data for: {', '.join(low_performers)}")
    
    else:
        print("❌ No test cases found! Check your dataset directory structure.")
        print("💡 Expected structure: dataset/person_name/image_files")

def test_single_tolerance(tolerance_value):
    """Test model with a specific tolerance value"""
    global TOLERANCE
    original_tolerance = TOLERANCE
    TOLERANCE = tolerance_value
    
    print(f"\n🎯 Testing with tolerance: {tolerance_value}")
    print("-" * 40)
    
    test_model_accuracy()
    
    TOLERANCE = original_tolerance

def run_tolerance_analysis():
    """Run tests with different tolerance values"""
    print("🔬 Running Tolerance Analysis...")
    print("=" * 60)
    
    tolerance_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for tolerance in tolerance_values:
        test_single_tolerance(tolerance)
    
    print("\n💡 Tolerance Analysis Complete!")
    print("Choose the tolerance that gives the best balance of accuracy and false positives.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--tolerance-analysis":
        run_tolerance_analysis()
    else:
        test_model_accuracy()
        
        # Ask if user wants to run tolerance analysis
        print("\n" + "=" * 60)
        response = input("🔬 Would you like to run tolerance analysis? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            run_tolerance_analysis()
