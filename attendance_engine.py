"""
attendance_engine.py
--------------------
Headless recognition engine used by the live attendance window.
Separates OpenCV / camera logic from the Tkinter GUI so it can be
tested without a display.
"""

import cv2
import numpy as np
import os
from database import (
    MODEL_PATH, get_student_name, record_attendance, get_session_attendance,
    DATASET_DIR
)
from face_recognizer import LBPHFaceRecognizer_create

# ── Constants ─────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 8959  # chi-squared × 100; tune lower = stricter, higher = looser
MIN_FACE_SIZE        = (80, 80)
SCALE_FACTOR         = 1.3
MIN_NEIGHBOURS       = 5

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class AttendanceEngine:
    """
    Manages the webcam loop, face detection, recognition, and DB writes.

    Usage
    -----
    engine = AttendanceEngine(session_id=1, on_recognised=callback)
    engine.start()      # blocks until stop() is called or 'q' pressed
    engine.stop()
    """

    def __init__(self, session_id: int,
                 on_recognised=None,
                 on_frame=None,
                 camera_index: int = 0):
        """
        Parameters
        ----------
        session_id     : int  – current session to record attendance against
        on_recognised  : callable(student_name, is_new) – fired when a face is matched
        on_frame       : callable(frame_bgr) – fired every processed frame (for GUI preview)
        camera_index   : int  – webcam device index
        """
        self.session_id    = session_id
        self.on_recognised = on_recognised
        self.on_frame      = on_frame
        self.camera_index  = camera_index

        self._running      = False
        self._recognised   : set[int] = set()   # student_ids already marked this session

        # Load cascade
        self._cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        if self._cascade.empty():
            raise RuntimeError("Haar cascade file not found.")

        # Load recogniser
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}.\n"
                "Please run Training first."
            )
        self._recognizer = LBPHFaceRecognizer_create()
        self._recognizer.read(MODEL_PATH)

    # ------------------------------------------------------------------
    def start(self):
        """Open the webcam and run the recognition loop (blocking)."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera (index {self.camera_index}). "
                "Make sure a webcam is connected."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self._running = True
        font = cv2.FONT_HERSHEY_SIMPLEX

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_eq  = cv2.equalizeHist(gray_raw)   # equalized → used for detection only

            faces = self._cascade.detectMultiScale(
                gray_eq,
                scaleFactor=SCALE_FACTOR,
                minNeighbors=MIN_NEIGHBOURS,
                minSize=MIN_FACE_SIZE,
            )

            for (x, y, w, h) in faces:
                # Use the RAW (non-equalized) ROI for recognition.
                # face_recognizer._preprocess() will equalise it exactly
                # once, matching what was saved during face capture.
                roi = gray_raw[y:y + h, x:x + w]
                roi = cv2.resize(roi, (100, 100))

                label, confidence = self._recognizer.predict(roi)

                if label != -1 and confidence < CONFIDENCE_THRESHOLD:
                    name     = get_student_name(label)
                    is_new   = record_attendance(label, self.session_id, confidence)
                    color    = (0, 255, 0)   # green

                    if label not in self._recognised:
                        self._recognised.add(label)
                        if self.on_recognised:
                            self.on_recognised(name, is_new)
                else:
                    name  = "Unknown"
                    color = (0, 0, 255)     # red

                # Annotate frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name,
                            (x, y - 10), font, 0.7, color, 2)
                cv2.putText(frame, f"Conf: {confidence:.1f}",
                            (x, y + h + 22), font, 0.55, color, 1)

            # Overlay: recognised count
            count_text = f"Recognised: {len(self._recognised)}"
            cv2.putText(frame, count_text,
                        (10, 30), font, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, "Press Q to stop",
                        (10, 60), font, 0.6, (200, 200, 200), 1)

            # Fire frame callback (GUI uses this to update preview)
            if self.on_frame:
                self.on_frame(frame.copy())

            # Fallback: show OpenCV window if no GUI callback
            if self.on_frame is None:
                cv2.imshow("Smart Attendance – Press Q to stop", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()
        self._running = False

    # ------------------------------------------------------------------
    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    def get_recognised_count(self) -> int:
        return len(self._recognised)

    def get_session_summary(self) -> list[dict]:
        return get_session_attendance(self.session_id)
