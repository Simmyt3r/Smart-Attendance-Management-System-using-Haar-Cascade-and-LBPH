"""
train_model.py
--------------
Reads all captured face images from dataset/, trains the LBPH model,
and saves it to trainer/trainer.pkl.

Can be called as:
    python train_model.py
or imported and called as:
    from train_model import train_model
    success, msg = train_model(progress_callback=print)
"""

import os
import cv2
import numpy as np
from database import DATASET_DIR, MODEL_PATH
from face_recognizer import LBPHFaceRecognizer_create


def train_model(progress_callback=None) -> tuple[bool, str]:
    """
    Scan dataset/, build the face / label arrays, train LBPH, save model.

    Parameters
    ----------
    progress_callback : callable(str), optional
        Called with status messages so a GUI can update a progress label.

    Returns
    -------
    (success: bool, message: str)
    """

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        print(msg)

    # ── Collect images ────────────────────────────────────────────────────
    if not os.path.isdir(DATASET_DIR):
        return False, "Dataset directory not found."

    student_dirs = [
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]

    if not student_dirs:
        return False, "No face data found. Please capture faces first."

    face_samples: list[np.ndarray] = []
    face_labels:  list[int]        = []
    total_images = 0

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    for sid_str in student_dirs:
        try:
            student_id = int(sid_str)
        except ValueError:
            continue

        student_dir = os.path.join(DATASET_DIR, sid_str)
        image_files = [
            f for f in os.listdir(student_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]

        if not image_files:
            continue

        log(f"Loading {len(image_files)} images for student ID {student_id}...")

        for img_file in image_files:
            img_path = os.path.join(student_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # _preprocess() inside train() handles resize + equalisation.
            face_samples.append(img)
            face_labels.append(student_id)
            total_images += 1

    if total_images == 0:
        return False, "No valid images found in dataset."

    log(f"Training on {total_images} images across {len(student_dirs)} student(s)...")

    # ── Train ─────────────────────────────────────────────────────────────
    recognizer = LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=8959
    )
    recognizer.train(face_samples, np.array(face_labels))

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    recognizer.write(MODEL_PATH)

    msg = (f"Training complete! {total_images} images, "
           f"{len(student_dirs)} student(s). Model saved.")
    log(msg)
    return True, msg


# ── CLI entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ok, msg = train_model()
    print("Result:", msg)
