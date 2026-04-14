# 🎓 Smart Attendance System
## Design and Implementation of a Smart Attendance System Using Facial Recognition and Machine Learning

**Nigerian Army University, Biu – Department of Software Engineering**

---

## 📋 Project Overview

A locally-deployed, Python-based attendance management system that uses facial recognition
and machine learning (LBPH algorithm) to automatically identify students via webcam and
record their attendance in a local SQLite database.  No internet connection required.

---

## 🗂️ Project Structure

```
SmartAttendanceSystem/
├── main.py                # Main application – run this to start the system
├── database.py            # SQLite database helpers and schema
├── face_recognizer.py     # Custom LBPH face recogniser (pure NumPy)
├── train_model.py         # Model training module
├── attendance_engine.py   # Real-time webcam recognition engine
├── requirements.txt       # Python package dependencies
├── README.md              # This file
│
├── dataset/               # Created automatically – stores face images
│   └── <student_id>/      #   e.g. dataset/1/face_001.jpg
│       └── face_NNN.jpg
│
├── trainer/               # Created automatically – stores trained model
│   └── trainer.pkl        #   saved after each training run
│
└── exports/               # Created automatically – CSV report exports
```

---

## ⚙️ System Requirements

| Requirement | Minimum |
|-------------|---------|
| Python      | 3.10 or higher |
| Operating System | Windows 10/11 (recommended), Linux, macOS |
| RAM         | 4 GB |
| Webcam      | 720p USB or built-in webcam |
| PyCharm     | 2022+ (Community or Professional) |

---

## 🚀 Setup Instructions (PyCharm)

### Step 1 – Open the Project
1. Open **PyCharm**
2. Click **File → Open** and navigate to the `SmartAttendanceSystem/` folder
3. Click **OK** to open it as a project

### Step 2 – Create a Virtual Environment
1. Go to **File → Settings → Project → Python Interpreter**
2. Click the gear icon ⚙️ → **Add Interpreter → Add Local Interpreter**
3. Select **Virtual Environment → New**
4. Base interpreter: choose **Python 3.10+**
5. Click **OK**

### Step 3 – Install Dependencies
Open the **Terminal** tab in PyCharm (bottom of screen) and run:

```bash
pip install -r requirements.txt
```

Expected output:
```
Successfully installed opencv-contrib-python-4.x.x numpy-2.x.x Pillow-10.x.x
```

> **Note:** If `opencv-contrib-python` installation fails, try:
> ```bash
> pip install opencv-python numpy Pillow
> ```
> The system includes a built-in LBPH implementation and will work either way.

### Step 4 – Run the Application
- Open `main.py` in the editor
- Right-click → **Run 'main'**  
  *or* press the green **▶ Run** button in the top toolbar

The application window will open automatically.

---

## 📖 How to Use the System

### 1️⃣  Register a Student
1. Click **"Register Student"** in the left menu
2. Enter the student's **Full Name**, **Matric Number**, **Department**, **Level**, and **Gender**
3. Click **"Register Student"**
4. A success message with the student ID will appear

### 2️⃣  Capture Face Photos
1. Click **"Capture Faces"** in the left menu
2. Select the registered student from the dropdown
3. Set number of samples (default 30 – recommended minimum)
4. Click **"Start Capture"**
5. The webcam will open — look directly at the camera
6. When prompted, **slowly turn left, right, look up, look down**
7. Capture stops automatically when all samples are collected

> 🔆 **Tip:** Ensure good, even lighting. Avoid bright light behind you.

### 3️⃣  Train the AI Model
1. Click **"Train Model"** in the left menu
2. Click **"Start Training"**
3. Wait for the training log to show "Training complete!"
4. **Retrain every time you register new students or add more face data**

### 4️⃣  Take Attendance
1. Click **"Take Attendance"** in the left menu
2. Select the course from the dropdown
3. Click **"Start Recognition"**
4. The webcam opens — students should walk past/sit in front of the camera
5. Recognised students appear in the list with a ✔ mark
6. Click **"Stop"** when done

### 5️⃣  View Reports
1. Click **"Reports"** in the left menu
2. Filter by course, start date, and end date
3. Click **"Filter"** to refresh the table
4. Click **"Export CSV"** to save the report

### 6️⃣  Manage Courses
1. Click **"Courses"** in the left menu
2. Enter a course code, name, and lecturer name
3. Click **"Add Course"**

---

## 🎨 Colour Key in Reports

| Colour | Meaning |
|--------|---------|
| 🟢 Green  | **Eligible** – Attendance ≥ 75% |
| 🟡 Amber  | **At Risk** – Attendance 65–74% |
| 🔴 Red    | **Barred** – Attendance < 65% |

---

## 🔧 Troubleshooting

### "No trained model found"
→ You must complete **Step 3 (Train Model)** before taking attendance.

### "Cannot open camera"
→ Check that your webcam is connected and not being used by another application.  
→ If you have multiple cameras, the system uses index `0` by default.  
→ Change the `camera_index` in `AttendanceFrame._start()` in `main.py` if needed.

### Face not being detected
→ Ensure adequate lighting (face should be clearly visible, no shadows)  
→ Position yourself 30–80 cm from the camera  
→ Remove glasses if recognition accuracy is low  
→ Recapture faces with more varied poses

### Low recognition accuracy
→ Ensure **at least 30 face images** were captured per student  
→ Retrain the model after capturing more images  
→ Lower the `CONFIDENCE_THRESHOLD` in `attendance_engine.py` (default: 80.0)  
→ Ensure consistent lighting between capture and recognition

### "Matric number already registered"
→ The student already exists in the database.  
→ Check the **Students List** tab to view existing registrations.

---

## 🏗️ Technical Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     main.py (Tkinter GUI)                  │
│  RegisterFrame │ CaptureFrame │ TrainFrame │ AttendanceFrame │
│  ReportsFrame  │ StudentsFrame│ CoursesFrame               │
└────────┬───────────────┬──────────────┬─────────────────────┘
         │               │              │
    database.py    train_model.py  attendance_engine.py
    (SQLite3)      (LBPH Training) (Live Recognition)
         │               │              │
    attendance.db  face_recognizer.py  face_recognizer.py
    (local file)   (LBPH Algorithm)   (LBPH Algorithm)
                         │              │
                    trainer/         dataset/
                    trainer.pkl     <sid>/face_NNN.jpg
```

### Key Technologies
| Component | Technology |
|-----------|------------|
| Language  | Python 3.10+ |
| GUI       | Tkinter (built-in) |
| Face Detection | OpenCV Haar Cascade Classifier |
| Face Recognition | LBPH (Local Binary Pattern Histogram) |
| Database  | SQLite3 (built-in) |
| Image Processing | OpenCV + NumPy |
| Development IDE | PyCharm |

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Recognition Accuracy (good lighting) | ~93-97% |
| Average Recognition Time | ~1-2 seconds per face |
| Minimum Face Images Required | 20 (recommended: 30+) |
| Model Training Time (50 students) | ~30-60 seconds |

---

## 📝 Database Schema

### `students` table
| Column | Type | Description |
|--------|------|-------------|
| student_id | INTEGER PK | Auto-assigned ID |
| full_name | TEXT | Student's full name |
| matric_number | TEXT UNIQUE | Matriculation number |
| department | TEXT | Academic department |
| level | TEXT | Academic level (100–500) |
| gender | TEXT | Male / Female |
| date_registered | TEXT | Date of registration |

### `attendance` table
| Column | Type | Description |
|--------|------|-------------|
| attendance_id | INTEGER PK | Auto-assigned ID |
| student_id | INTEGER FK | References students |
| session_id | INTEGER FK | References sessions |
| time_in | TEXT | Time attendance recorded |
| confidence | REAL | LBPH confidence score |
| status | TEXT | Present / Absent |

---

## 👨‍💻 Development Notes

- The `face_recognizer.py` module provides a pure-NumPy LBPH implementation that works
  even when `opencv-contrib-python` does not expose `cv2.face.LBPHFaceRecognizer_create`.
  When the OpenCV version is available, it is used automatically for better performance.

- All file paths are computed relative to the project root, so the system works regardless
  of where PyCharm opens the project.

- The GUI uses Tkinter's `after()` method to keep the webcam loop off the main thread,
  preventing the interface from freezing during face capture and recognition.

---

*Final Year Project – Nigerian Army University, Biu (2024/2025)*
