import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_file
import pickle
import os
import csv
import io
from datetime import datetime
import threading
import queue
from collections import Counter, deque

# Import optimized modules
from face_encoder import extract_face_embeddings
from realtime_face_recognition import (
    load_face_database, recognize_face_optimized, FrameProcessor,
    initialize_mediapipe, track_faces
)

app = Flask(__name__)

# Global variables with thread safety
known_encodings = []
known_names = []
attendance = {}
recognition_running = True
current_people_count = 0
frame_processor = None
processing_lock = threading.Lock()

# Load encodings once at startup
print("Loading face database...")
known_encodings, known_names = load_face_database()
if known_encodings:
    print(f"Loaded {len(known_encodings)} face encodings")
else:
    print("No face encodings found! Run face_encoder.py first")

def gen_frames():
    global recognition_running, current_people_count, frame_processor
    
    # Initialize MediaPipe
    initialize_mediapipe()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    
    # Optimized camera settings for fixed size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Pre-warm camera
    for _ in range(5):
        cap.read()
    
    # Initialize frame processor
    if frame_processor is None:
        frame_processor = FrameProcessor()
        frame_processor.start()
    
    frame_count = 0
    process_every_n_frames = 3
    results = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror correction
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Process frames when recognition is running
            if recognition_running and frame_count % process_every_n_frames == 0:
                frame_processor.add_frame(frame.copy())
            
            # Get processing results
            if recognition_running:
                result = frame_processor.get_result()
                if result:
                    results, people_count_temp = result
                    
                    with processing_lock:
                        current_people_count = people_count_temp
                        
                        # Update attendance with better timestamp
                        for (face_location, name) in results:
                            if name != "Unknown":
                                if name not in attendance:
                                    attendance[name] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Draw recognition results
                for (face_location, name) in results:
                    y1, x2, y2, x1 = face_location
                    
                    # Choose color based on recognition
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    
                    # Draw face rectangle with thicker border
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw name label with better styling
                    label_text = name
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    # Background rectangle for text
                    cv2.rectangle(frame, (x1, y2 - text_size[1] - 10), 
                                 (x1 + text_size[0] + 10, y2), color, cv2.FILLED)
                    
                    # Text with better visibility
                    cv2.putText(frame, label_text, (x1 + 5, y2 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw people count overlay
                cv2.putText(frame, f"People: {current_people_count}", (10, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                # Draw recognition status
                cv2.putText(frame, "Recognition Active", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Show "Recognition Stopped" when paused
                with processing_lock:
                    current_people_count = 0
                
                cv2.putText(frame, "Recognition Stopped", (10, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(frame, "Press Resume to continue", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Encode frame with consistent quality
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        print(f"Camera error: {e}")
    finally:
        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def get_attendance():
    with processing_lock:
        # Return data in format: [[name, timestamp], ...]
        attendance_list = [[name, timestamp] for name, timestamp in attendance.items()]
        return jsonify(attendance_list)

@app.route('/people_count')
def people_count():
    return jsonify({'count': current_people_count})

@app.route('/system_status')
def system_status():
    status = "Recognition Active" if recognition_running else "Recognition Stopped"
    return jsonify({'status': status, 'running': recognition_running})

@app.route('/download_attendance')
def download_attendance():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Name', 'First Scanned Time', 'Date'])
    
    with processing_lock:
        for name, timestamp in sorted(attendance.items(), key=lambda x: x[1]):
            # Parse timestamp and format for CSV
            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            date_str = dt.strftime('%Y-%m-%d')
            time_str = dt.strftime('%H:%M:%S')
            writer.writerow([name, time_str, date_str])
    
    output.seek(0)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"scanned_persons_report_{current_time}.csv"
    
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.route('/stop', methods=['POST'])
def stop_recognition():
    global recognition_running, current_people_count
    recognition_running = False
    with processing_lock:
        current_people_count = 0
    return ('', 204)

@app.route('/resume', methods=['POST'])
def resume_recognition():
    global recognition_running
    recognition_running = True
    return ('', 204)

@app.route('/clear_attendance', methods=['POST'])
def clear_attendance():
    global attendance
    with processing_lock:
        attendance.clear()
    return jsonify({'status': 'cleared'})

# Cleanup on shutdown
import atexit

def cleanup():
    global frame_processor
    if frame_processor:
        frame_processor.stop()

atexit.register(cleanup)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
