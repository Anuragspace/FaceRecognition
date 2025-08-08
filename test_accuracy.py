import cv2
import numpy as np
import pickle
import os
from face_encoder import extract_face_features, cosine_similarity_manual

ENCODINGS_PATH = "face_encodings.pickle"
TOLERANCE = 0.6

def load_face_database():
    """Load the face database"""
    if not os.path.exists(ENCODINGS_PATH):
        return [], []
    
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

def recognize_face(face_encoding, known_encodings, known_names, tolerance=TOLERANCE):
    """Recognize a face using cosine similarity"""
    if not known_encodings:
        return "Unknown"
    
    # Compute similarities with all known faces
    similarities = []
    for known_encoding in known_encodings:
        similarity = cosine_similarity_manual(face_encoding, known_encoding)
        similarities.append(similarity)
    
    similarities = np.array(similarities)
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]
    
    # Convert similarity to distance (1 - similarity)
    distance = 1 - best_similarity
    
    if distance <= tolerance:
        name = known_names[best_idx]
    else:
        name = "Unknown"
    
    return name

def test_model_accuracy():
    """Test the model accuracy using the dataset images"""
    known_encodings, known_names = load_face_database()
    
    if not known_encodings:
        print("No face database found! Please run face_encoder.py first.")
        return
    
    dataset_dir = "dataset"
    total_tests = 0
    correct_predictions = 0
    results = {}
    
    print("Testing Model Accuracy...")
    print("=" * 50)
    
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
                
                # Extract face features
                face_locations, face_encodings = extract_face_features(image)
                
                # Test each detected face
                for face_encoding in face_encodings:
                    predicted_name = recognize_face(face_encoding, known_encodings, known_names)
                    
                    total_tests += 1
                    person_total += 1
                    
                    if predicted_name == person_name:
                        correct_predictions += 1
                        person_correct += 1
                        status = "✓ CORRECT"
                    else:
                        status = f"✗ WRONG (predicted: {predicted_name})"
                    
                    print(f"  {img_name}: {status}")
                    
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
        
        print("\n" + "=" * 50)
        print("MODEL ACCURACY RESULTS")
        print("=" * 50)
        
        print(f"Overall Accuracy: {overall_accuracy:.1f}% ({correct_predictions}/{total_tests})")
        print(f"Total Tests: {total_tests}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Wrong Predictions: {total_tests - correct_predictions}")
        
        print("\nPer-Person Results:")
        for person, result in results.items():
            print(f"  {person}: {result['accuracy']:.1f}% ({result['correct']}/{result['total']})")
        
        # Performance analysis
        print("\nPerformance Analysis:")
        if overall_accuracy >= 90:
            print("🟢 EXCELLENT - Model is performing very well")
        elif overall_accuracy >= 80:
            print("🟡 GOOD - Model is performing well")
        elif overall_accuracy >= 70:
            print("🟠 FAIR - Model needs improvement")
        else:
            print("🔴 POOR - Model needs significant improvement")
            
        print("\nRecommendations:")
        if overall_accuracy < 90:
            print("- Add more training images with different angles and lighting")
            print("- Ensure images have clear, well-lit faces")
            print("- Remove blurry or low-quality images from dataset")
            print("- Adjust TOLERANCE value in face_encoder.py")
    
    else:
        print("No test cases found!")

if __name__ == "__main__":
    test_model_accuracy()
