"""
diagnose.py
-----------
Run this BEFORE taking attendance to see what raw confidence scores
the system is producing for your face.

It opens the camera, detects your face, and prints the ACTUAL
chi-squared score WITHOUT any threshold filter.

Usage:  python diagnose.py
Press Q to quit.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import database as db
import face_recognizer as fr

MODEL_PATH   = db.MODEL_PATH
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

print("\n── Smart Attendance Diagnostic ─────────────────────────────────────")

if not os.path.isfile(MODEL_PATH):
    print("❌  No trained model found at:", MODEL_PATH)
    print("    Please run Training first, then run this script.")
    sys.exit(1)

# Load model with NO threshold (float('inf') means always return a result)
rec = fr.LBPHFaceRecognizer_create(threshold=float("inf"))
rec.read(MODEL_PATH)

# Show what students are in the model
students = db.get_all_students()
print(f"\n   Students in database: {len(students)}")
for s in students:
    imgs = db.count_face_images(s["student_id"])
    print(f"     ID {s['student_id']:3d} | {s['full_name']:30s} | {imgs} face images")

print("\n   READING CAMERA — look directly at the camera.")
print("   The CONF score is the raw distance. Lower = closer match.")
print("   Press Q to quit.\n")

cascade = cv2.CascadeClassifier(CASCADE_PATH)
cap     = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

font    = cv2.FONT_HERSHEY_SIMPLEX
scores  = []   # collect scores to suggest a threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq  = cv2.equalizeHist(gray_raw)

    faces = cascade.detectMultiScale(gray_eq, 1.3, 5, minSize=(80, 80))

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 30),
                    font, 0.8, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            roi = gray_raw[y:y+h, x:x+w]
            roi = cv2.resize(roi, (100, 100))

            label, conf = rec.predict(roi)
            name = db.get_student_name(label) if label != -1 else "???"
            scores.append(conf)

            # Colour code: green if conf < 200, yellow < 400, red = far
            color = ((0,200,0) if conf < 200
                     else (0,200,255) if conf < 400
                     else (0,0,255))

            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)

            # Show the raw score prominently
            cv2.putText(frame, f"CONF: {conf:.1f}",
                        (x, y-30), font, 0.9, color, 2)
            cv2.putText(frame, f"Best match: {name} (ID {label})",
                        (x, y-8), font, 0.65, color, 2)

            # Print to terminal too
            print(f"  Raw confidence: {conf:8.2f}   Best match: {name} (ID {label})")

    # Guide text
    cv2.putText(frame, "DIAGNOSTIC MODE – Q to quit",
                (10, frame.shape[0]-10), font, 0.6, (200,200,200), 1)

    cv2.imshow("Diagnostic – what confidence scores does your face get?", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# ── Suggest a threshold ───────────────────────────────────────────────────────
if scores:
    arr = np.array(scores)
    print(f"\n── Score Summary ({len(arr)} readings) ──────────────────────────────")
    print(f"   Min   : {arr.min():.1f}")
    print(f"   Median: {np.median(arr):.1f}")
    print(f"   Max   : {arr.max():.1f}")
    print(f"\n   SUGGESTED THRESHOLD: {arr.max() * 1.3:.0f}")
    print(f"\n   → Open attendance_engine.py and set:")
    print(f"       CONFIDENCE_THRESHOLD = {arr.max() * 1.3:.0f}")
    print(f"   → Open face_recognizer.py and change the default threshold= to the same value.")
    print(f"   → Open train_model.py and change threshold= to the same value.\n")
else:
    print("\n   No face readings collected. Make sure your face was visible.")
