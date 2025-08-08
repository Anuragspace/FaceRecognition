import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_file
import pickle
import os
import csv
import io
from datetime import datetime

# Import your face recognition logic
from realtime_face_recognition import load_face_database, extract_face_embeddings, recognize_face

app = Flask(__name__)

# Load encodings once
known_encodings, known_names = load_face_database()

attendance = {}
recognition_running = True
current_people_count = 0

def gen_frames():
    global recognition_running, current_people_count
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    process_every_n_frames = 2
    frame_count = 0
    results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        if recognition_running and frame_count % process_every_n_frames == 0:
            face_locations, face_encodings = extract_face_embeddings(frame)
            results = []
            for (face_location, face_encoding) in zip(face_locations, face_encodings):
                name = recognize_face(face_encoding, known_encodings, known_names)
                results.append((face_location, name))
                if name != "Unknown":
                    if name not in attendance:
                        attendance[name] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Update people count only when recognition is running
        current_people_count = len(results) if recognition_running else 0
        
        # Only draw face rectangles and labels when recognition is running
        if recognition_running:
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
        
        # People count overlay
        status_text = f"People: {current_people_count}" if recognition_running else "Recognition Stopped"
        color = (255, 255, 0) if recognition_running else (0, 0, 255)
        cv2.putText(frame, status_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        ret2, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/attendance')
def get_attendance():
    # Return attendance as a list of dicts sorted by timestamp
    att = [
        {"name": name, "timestamp": timestamp}
        for name, timestamp in sorted(attendance.items(), key=lambda x: x[1])
    ]
    return jsonify(att)

@app.route('/people_count')
def get_people_count():
    return jsonify({"count": current_people_count})

@app.route('/download_attendance')
def download_attendance():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Name', 'Timestamp'])
    for name, timestamp in sorted(attendance.items(), key=lambda x: x[1]):
        writer.writerow([name, timestamp])
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment; filename=attendance_report.csv"}
    )

@app.route('/stop', methods=['POST'])
def stop_recognition():
    global recognition_running
    recognition_running = False
    return ('', 204)

@app.route('/resume', methods=['POST'])
def resume_recognition():
    global recognition_running
    recognition_running = True
    return ('', 204)
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
