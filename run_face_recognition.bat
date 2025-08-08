@echo off
echo Face Recognition System Setup
echo ============================
echo.

echo Step 1: Building face database...
"C:/Users/anura/Desktop/face recognition/backend/.venv/Scripts/python.exe" face_encoder.py
echo.

if %ERRORLEVEL% EQU 0 (
    echo Face database created successfully!
    echo.
    echo Step 2: Starting real-time face recognition...
    echo Press 'q' in the camera window to quit.
    echo.
    pause
    "C:/Users/anura/Desktop/face recognition/backend/.venv/Scripts/python.exe" realtime_face_recognition.py
) else (
    echo Error building face database!
    pause
)
