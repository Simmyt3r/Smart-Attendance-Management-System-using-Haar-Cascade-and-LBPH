"""
main.py
-------
Smart Attendance System – Main GUI Application
Nigerian Army University, Biu – Dept. of Software Engineering

Run with:  python main.py
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import date, datetime
import csv

import cv2
from PIL import Image, ImageTk
import numpy as np

# Local modules
import database as db
from train_model import train_model
from attendance_engine import AttendanceEngine

# ── Colour palette ─────────────────────────────────────────────────────────────
BG       = "#0D1B40"
PANEL    = "#1E293B"
CARD     = "#FFFFFF"
ACCENT   = "#1A56DB"
GOLD     = "#F59E0B"
GREEN    = "#10B981"
RED      = "#EF4444"
GREY     = "#94A3B8"
TEXT     = "#1E293B"
TEXT_LT  = "#FFFFFF"
FONT     = "Segoe UI"


def _btn(parent, text, cmd, bg=ACCENT, fg=TEXT_LT, w=18):
    return tk.Button(parent, text=text, command=cmd,
                     bg=bg, fg=fg, font=(FONT, 10, "bold"),
                     relief="flat", cursor="hand2",
                     padx=10, pady=6, width=w,
                     activebackground=GOLD, activeforeground=TEXT)


def _lbl(parent, text, size=11, bold=False, fg=TEXT, bg=CARD):
    weight = "bold" if bold else "normal"
    return tk.Label(parent, text=text, font=(FONT, size, weight),
                    fg=fg, bg=bg)


def _entry(parent, textvariable, width=28):
    return tk.Entry(parent, textvariable=textvariable,
                    font=(FONT, 10), width=width,
                    relief="solid", bd=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Register Student Frame
# ═══════════════════════════════════════════════════════════════════════════════

class RegisterFrame(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=CARD)
        self.app = app
        self._build()

    def _build(self):
        # Title
        tk.Label(self, text="Register New Student",
                 font=(FONT, 16, "bold"), bg=CARD, fg=BG).pack(pady=(18, 4))
        tk.Label(self, text="Fill in the student details below",
                 font=(FONT, 10), bg=CARD, fg=GREY).pack()
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=20, pady=10)

        form = tk.Frame(self, bg=CARD)
        form.pack(padx=30, pady=4)

        self.vars = {}
        fields = [
            ("Full Name",       "full_name",     "e.g. Abdullahi Musa Ibrahim"),
            ("Matric Number",   "matric_number", "e.g. SOF/21U/0001"),
            ("Department",      "department",    "Software Engineering"),
            ("Level",           "level",         "100"),
        ]
        for i, (label, key, placeholder) in enumerate(fields):
            tk.Label(form, text=label + ":", font=(FONT, 10, "bold"),
                     bg=CARD, fg=TEXT, anchor="e", width=14).grid(
                row=i, column=0, padx=(0, 8), pady=6, sticky="e")
            v = tk.StringVar(value=placeholder if key in ("department", "level") else "")
            self.vars[key] = v
            _entry(form, v, width=30).grid(row=i, column=1, pady=6, sticky="w")

        # Gender dropdown
        tk.Label(form, text="Gender:", font=(FONT, 10, "bold"),
                 bg=CARD, fg=TEXT, anchor="e", width=14).grid(
            row=len(fields), column=0, padx=(0, 8), pady=6, sticky="e")
        self.vars["gender"] = tk.StringVar(value="Male")
        ttk.Combobox(form, textvariable=self.vars["gender"],
                     values=["Male", "Female"], state="readonly",
                     font=(FONT, 10), width=28).grid(
            row=len(fields), column=1, pady=6, sticky="w")

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=20, pady=12)
        btn_frame = tk.Frame(self, bg=CARD)
        btn_frame.pack()
        _btn(btn_frame, "✔  Register Student",
             self._register, bg=GREEN, w=20).pack(side="left", padx=8)
        _btn(btn_frame, "✖  Clear Form",
             self._clear, bg=GREY, fg=TEXT, w=14).pack(side="left", padx=8)

        # Status
        self.status_var = tk.StringVar()
        tk.Label(self, textvariable=self.status_var,
                 font=(FONT, 10), bg=CARD, fg=GREEN).pack(pady=8)

    def _register(self):
        name   = self.vars["full_name"].get().strip()
        matric = self.vars["matric_number"].get().strip()
        dept   = self.vars["department"].get().strip()
        level  = self.vars["level"].get().strip()
        gender = self.vars["gender"].get()

        if not name or not matric:
            messagebox.showwarning("Missing Fields",
                                   "Full Name and Matric Number are required.")
            return

        sid = db.add_student(name, matric, dept, level, gender)
        if sid is None:
            self.status_var.set(f"❌  Matric number '{matric}' already registered.")
        else:
            self.status_var.set(
                f"✔  Registered: {name} (ID: {sid}). "
                "Go to Face Capture next."
            )
            self._clear(keep_status=True)
            self.app.refresh_student_list()

    def _clear(self, keep_status=False):
        self.vars["full_name"].set("")
        self.vars["matric_number"].set("")
        self.vars["department"].set("Software Engineering")
        self.vars["level"].set("100")
        self.vars["gender"].set("Male")
        if not keep_status:
            self.status_var.set("")


# ═══════════════════════════════════════════════════════════════════════════════
# Face Capture Frame
# ═══════════════════════════════════════════════════════════════════════════════

class CaptureFrame(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=CARD)
        self.app      = app
        self._cap     = None
        self._running = False
        self._count   = 0
        self._target  = 30
        self._sid     = None
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._build()

    def _build(self):
        tk.Label(self, text="Capture Student Face",
                 font=(FONT, 16, "bold"), bg=CARD, fg=BG).pack(pady=(18, 2))
        tk.Label(self, text="Select a registered student and capture 30 face samples",
                 font=(FONT, 10), bg=CARD, fg=GREY).pack()
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=20, pady=10)

        top = tk.Frame(self, bg=CARD)
        top.pack(padx=24, pady=4)

        tk.Label(top, text="Select Student:", font=(FONT, 10, "bold"),
                 bg=CARD).grid(row=0, column=0, sticky="e", padx=(0, 8))
        self.student_var = tk.StringVar()
        self.student_cb  = ttk.Combobox(top, textvariable=self.student_var,
                                         state="readonly", font=(FONT, 10), width=34)
        self.student_cb.grid(row=0, column=1, pady=4)

        tk.Label(top, text="Samples to capture:", font=(FONT, 10, "bold"),
                 bg=CARD).grid(row=1, column=0, sticky="e", padx=(0, 8))
        self.samples_var = tk.IntVar(value=30)
        ttk.Spinbox(top, from_=10, to=1000, textvariable=self.samples_var,
                    font=(FONT, 10), width=6, state="readonly").grid(
            row=1, column=1, pady=4, sticky="w")

        btn_frame = tk.Frame(self, bg=CARD)
        btn_frame.pack(pady=8)
        self.start_btn = _btn(btn_frame, "▶  Start Capture", self._start, bg=ACCENT, w=16)
        self.start_btn.pack(side="left", padx=6)
        self.stop_btn  = _btn(btn_frame, "■  Stop",          self._stop,  bg=RED, w=10)
        self.stop_btn.pack(side="left", padx=6)
        self.stop_btn.config(state="disabled")

        # Video frame
        self.video_label = tk.Label(self, bg="#000000",
                                    width=400, height=300)
        self.video_label.pack(pady=6)

        # Progress
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var,
                                             maximum=100, length=380)
        self.progress_bar.pack(pady=4)
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status_var,
                 font=(FONT, 10), bg=CARD, fg=ACCENT).pack()

        self.refresh()

    def refresh(self):
        students = db.get_all_students()
        names    = [f"{s['full_name']}  ({s['matric_number']})" for s in students]
        self._student_map = {
            f"{s['full_name']}  ({s['matric_number']})": s["student_id"]
            for s in students
        }
        self.student_cb["values"] = names
        if names:
            self.student_cb.current(0)

    def _start(self):
        sel = self.student_var.get()
        if not sel:
            messagebox.showwarning("Select Student", "Please select a student first.")
            return
        self._sid    = self._student_map[sel]
        self._target = self.samples_var.get()
        self._count  = 0
        self._running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        save_dir = os.path.join(db.DATASET_DIR, str(self._sid))
        os.makedirs(save_dir, exist_ok=True)
        self._save_dir = save_dir

        self._loop()

    def _loop(self):
        if not self._running:
            return
        ret, frame = self._cap.read()
        if not ret:
            self._stop()
            return

        gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq  = cv2.equalizeHist(gray_raw)   # equalized for detection only
        faces = self._cascade.detectMultiScale(gray_eq, 1.3, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            if self._count < self._target:
                # Save RAW (non-equalized) ROI so face_recognizer._preprocess()
                # can apply equalization exactly once, matching recognition time.
                roi = gray_raw[y:y + h, x:x + w]
                roi = cv2.resize(roi, (100, 100))
                img_path = os.path.join(
                    self._save_dir, f"face_{self._count + 1:03d}.jpg")
                cv2.imwrite(img_path, roi)
                self._count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{self._count}/{self._target}",
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

        # Update progress
        pct = (self._count / self._target) * 100
        self.progress_var.set(pct)
        self.status_var.set(f"Capturing... {self._count}/{self._target}")

        # Display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img       = Image.fromarray(frame_rgb).resize((400, 300))
        imgtk     = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        if self._count >= self._target:
            self.status_var.set(
                f"✔  {self._target} samples captured for student ID {self._sid}!")
            self._stop(silent=True)
            return

        self.after(30, self._loop)

    def _stop(self, silent=False):
        self._running = False
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        if not silent and self._count < self._target:
            self.status_var.set(
                f"Stopped. {self._count} images saved for student {self._sid}.")


# ═══════════════════════════════════════════════════════════════════════════════
# Train Frame
# ═══════════════════════════════════════════════════════════════════════════════

class TrainFrame(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=CARD)
        self.app = app
        self._build()

    def _build(self):
        tk.Label(self, text="Train Face Recognition Model",
                 font=(FONT, 16, "bold"), bg=CARD, fg=BG).pack(pady=(18, 4))
        tk.Label(self,
                 text="This reads all captured face images and trains the AI model.\n"
                      "Run this after adding or updating student face data.",
                 font=(FONT, 10), bg=CARD, fg=GREY, justify="center").pack()
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=20, pady=14)

        info = tk.Frame(self, bg="#F0F4FF", relief="solid", bd=1)
        info.pack(padx=40, pady=6, fill="x")
        tk.Label(info,
                 text="ℹ️  Training may take 30-90 seconds depending on\n"
                      "the number of students and images captured.",
                 font=(FONT, 10), bg="#F0F4FF", fg=TEXT, pady=12).pack()

        self.train_btn = _btn(self, "🧠  Start Training", self._train,
                              bg=GOLD, fg=TEXT, w=22)
        self.train_btn.pack(pady=12)

        self.progress = ttk.Progressbar(self, mode="indeterminate", length=380)
        self.progress.pack(pady=4)

        self.log_text = tk.Text(self, height=10, width=58,
                                font=("Courier New", 9), bg="#1E293B",
                                fg="#A3E635", relief="flat", state="disabled")
        self.log_text.pack(padx=24, pady=10)

    def _log(self, msg: str):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.update_idletasks()

    def _train(self):
        self.train_btn.config(state="disabled")
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        self.progress.start(12)

        def _run():
            ok, msg = train_model(progress_callback=self._log)
            self.after(0, lambda: self._done(ok, msg))

        threading.Thread(target=_run, daemon=True).start()

    def _done(self, ok: bool, msg: str):
        self.progress.stop()
        self.train_btn.config(state="normal")
        if ok:
            messagebox.showinfo("Training Complete", msg)
        else:
            messagebox.showerror("Training Failed", msg)


# ═══════════════════════════════════════════════════════════════════════════════
# Attendance (live recognition) Frame
# ═══════════════════════════════════════════════════════════════════════════════

class AttendanceFrame(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=CARD)
        self.app     = app
        self._engine = None
        self._thread = None
        self._session_id = None
        self._build()

    def _build(self):
        tk.Label(self, text="Take Attendance",
                 font=(FONT, 16, "bold"), bg=CARD, fg=BG).pack(pady=(18, 2))
        tk.Label(self,
                 text="Select a course and start the camera to mark attendance automatically",
                 font=(FONT, 10), bg=CARD, fg=GREY).pack()
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=20, pady=10)

        top = tk.Frame(self, bg=CARD)
        top.pack(padx=24, pady=4)

        tk.Label(top, text="Select Course:", font=(FONT, 10, "bold"),
                 bg=CARD).grid(row=0, column=0, sticky="e", padx=(0, 8))
        self.course_var = tk.StringVar()
        self.course_cb  = ttk.Combobox(top, textvariable=self.course_var,
                                        state="readonly", font=(FONT, 10), width=36)
        self.course_cb.grid(row=0, column=1, pady=4)

        btn_frame = tk.Frame(self, bg=CARD)
        btn_frame.pack(pady=8)
        self.start_btn = _btn(btn_frame, "▶  Start Recognition",
                              self._start, bg=GREEN, w=20)
        self.start_btn.pack(side="left", padx=6)
        self.stop_btn  = _btn(btn_frame, "■  Stop",
                              self._stop, bg=RED, w=10)
        self.stop_btn.pack(side="left", padx=6)
        self.stop_btn.config(state="disabled")

        # Video + log side by side
        mid = tk.Frame(self, bg=CARD)
        mid.pack(fill="both", expand=True, padx=20)

        self.video_label = tk.Label(mid, bg="#000000", width=400, height=300)
        self.video_label.pack(side="left", padx=(0, 10))

        right = tk.Frame(mid, bg=CARD)
        right.pack(side="left", fill="both", expand=True)

        tk.Label(right, text="Recognised Students:",
                 font=(FONT, 10, "bold"), bg=CARD, fg=TEXT).pack(anchor="w")
        self.log_box = tk.Listbox(right, font=(FONT, 10),
                                   bg="#F0F4FF", fg=TEXT,
                                   selectmode="browse", height=12,
                                   relief="flat", bd=0)
        self.log_box.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Ready — select a course and press Start")
        tk.Label(self, textvariable=self.status_var,
                 font=(FONT, 10), bg=CARD, fg=ACCENT).pack(pady=4)

        self._refresh_courses()

    def _refresh_courses(self):
        courses = db.get_all_courses()
        names   = [f"{c['course_code']} – {c['course_name']}" for c in courses]
        self._course_map = {
            f"{c['course_code']} – {c['course_name']}": c["course_id"]
            for c in courses
        }
        self.course_cb["values"] = names
        if names:
            self.course_cb.current(0)

    def _start(self):
        sel = self.course_var.get()
        if not sel:
            messagebox.showwarning("Select Course", "Please select a course first.")
            return

        course_id = self._course_map[sel]
        self._session_id = db.create_session(course_id)
        self.log_box.delete(0, "end")
        self.status_var.set(f"Session {self._session_id} started. Camera active…")

        try:
            self._engine = AttendanceEngine(
                session_id    = self._session_id,
                on_recognised = self._on_recognised,
                on_frame      = self._on_frame,
            )
        except (FileNotFoundError, RuntimeError) as e:
            messagebox.showerror("Error", str(e))
            return

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self._thread = threading.Thread(
            target=self._engine.start, daemon=True)
        self._thread.start()

    def _stop(self):
        if self._engine:
            self._engine.stop()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        n = self._engine.get_recognised_count() if self._engine else 0
        self.status_var.set(
            f"Session ended. {n} student(s) marked present.")
        self.app.refresh_reports()

    def _on_recognised(self, name: str, is_new: bool):
        tag = "✔ NEW" if is_new else "✔ DUP"
        self.after(0, lambda: (
            self.log_box.insert("end", f"{tag}  {name}  {datetime.now().strftime('%H:%M:%S')}"),
            self.log_box.see("end"),
            self.status_var.set(f"Recognised: {name}")
        ))

    def _on_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img       = Image.fromarray(frame_rgb).resize((400, 300))
        imgtk     = ImageTk.PhotoImage(image=img)
        self.after(0, lambda: (
            setattr(self.video_label, "imgtk", imgtk),
            self.video_label.configure(image=imgtk)
        ))


# ═══════════════════════════════════════════════════════════════════════════════
# Reports Frame
# ═══════════════════════════════════════════════════════════════════════════════

class ReportsFrame(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=CARD)
        self.app = app
        self._build()

    def _build(self):
        tk.Label(self, text="Attendance Reports",
                 font=(FONT, 16, "bold"), bg=CARD, fg=BG).pack(pady=(18, 2))
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=20, pady=8)

        # Filter bar
        fbar = tk.Frame(self, bg=CARD)
        fbar.pack(padx=20, fill="x")

        tk.Label(fbar, text="Course:", font=(FONT, 10, "bold"),
                 bg=CARD).grid(row=0, column=0, sticky="e", padx=(0, 4))
        self.course_var = tk.StringVar(value="All Courses")
        self.course_cb  = ttk.Combobox(fbar, textvariable=self.course_var,
                                        state="readonly", font=(FONT, 10), width=28)
        self.course_cb.grid(row=0, column=1, padx=4, pady=4)

        tk.Label(fbar, text="From:", font=(FONT, 10, "bold"),
                 bg=CARD).grid(row=0, column=2, sticky="e", padx=(8, 4))
        self.start_var = tk.StringVar(value=f"{date.today().year}-01-01")
        _entry(fbar, self.start_var, width=12).grid(row=0, column=3, padx=4)

        tk.Label(fbar, text="To:", font=(FONT, 10, "bold"),
                 bg=CARD).grid(row=0, column=4, sticky="e", padx=(8, 4))
        self.end_var = tk.StringVar(value=str(date.today()))
        _entry(fbar, self.end_var, width=12).grid(row=0, column=5, padx=4)

        _btn(fbar, "🔍 Filter", self._load, bg=ACCENT, w=10).grid(
            row=0, column=6, padx=8)
        _btn(fbar, "📥 Export CSV", self._export, bg=GOLD, fg=TEXT, w=12).grid(
            row=0, column=7, padx=4)

        # Summary table
        cols = ("Name", "Matric No.", "Level",
                "Sessions", "Attended", "Pct %", "Status")
        self.tree = ttk.Treeview(self, columns=cols, show="headings",
                                  height=14)
        widths = [180, 120, 60, 80, 80, 70, 80]
        for col, w in zip(cols, widths):
            self.tree.heading(col, text=col, anchor="w")
            self.tree.column(col, width=w, anchor="w")

        # Colour tags
        self.tree.tag_configure("eligible", foreground="#065F46")
        self.tree.tag_configure("atrisk",   foreground="#92400E")
        self.tree.tag_configure("barred",   foreground="#991B1B")

        scroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(20, 0), pady=8)
        scroll.pack(side="left", fill="y", pady=8)

        self._refresh_courses()
        self._load()

    def _refresh_courses(self):
        courses = db.get_all_courses()
        names   = ["All Courses"] + [
            f"{c['course_code']} – {c['course_name']}" for c in courses]
        self._course_map = {
            f"{c['course_code']} – {c['course_name']}": c["course_id"]
            for c in courses
        }
        self.course_cb["values"] = names
        self.course_cb.current(0)

    def _load(self):
        sel        = self.course_var.get()
        course_id  = self._course_map.get(sel)
        start      = self.start_var.get() or None
        end        = self.end_var.get()   or None

        rows = db.get_attendance_report(course_id, start, end)

        for item in self.tree.get_children():
            self.tree.delete(item)

        for r in rows:
            tag = ("eligible" if r["status"] == "Eligible"
                   else "atrisk" if r["status"] == "At Risk"
                   else "barred")
            self.tree.insert("", "end", values=(
                r["full_name"],
                r["matric_number"],
                r["level"],
                r["total_sessions"],
                r["sessions_attended"],
                f"{r['attendance_pct']}%",
                r["status"],
            ), tags=(tag,))

    def _export(self):
        rows = db.get_detailed_attendance()
        if not rows:
            messagebox.showinfo("No Data", "No attendance records to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"attendance_{date.today()}.csv",
            initialdir=db.EXPORTS_DIR,
        )
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        messagebox.showinfo("Exported", f"Report saved to:\n{path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Students List Frame
# ═══════════════════════════════════════════════════════════════════════════════

class StudentsFrame(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=CARD)
        self.app = app
        self._build()

    def _build(self):
        tk.Label(self, text="Registered Students",
                 font=(FONT, 16, "bold"), bg=CARD, fg=BG).pack(pady=(18, 2))
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=20, pady=8)

        btn_row = tk.Frame(self, bg=CARD)
        btn_row.pack(padx=20, anchor="w")
        _btn(btn_row, "🔄  Refresh", self._load, bg=ACCENT, w=12).pack(
            side="left", padx=(0, 8))
        _btn(btn_row, "🗑  Delete Selected", self._delete,
             bg=RED, w=18).pack(side="left")

        cols = ("ID", "Name", "Matric No.", "Dept.", "Level",
                "Gender", "Registered", "Face Images")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=16)
        ws = [40, 170, 110, 160, 55, 65, 100, 90]
        for col, w in zip(cols, ws):
            self.tree.heading(col, text=col, anchor="w")
            self.tree.column(col, width=w, anchor="w")

        self.tree.tag_configure("noface", foreground="red")

        scroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(20, 0), pady=8)
        scroll.pack(side="left", fill="y", pady=8)

        self._load()

    def _load(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for s in db.get_all_students():
            n_imgs = db.count_face_images(s["student_id"])
            tag    = () if n_imgs > 0 else ("noface",)
            self.tree.insert("", "end", values=(
                s["student_id"],
                s["full_name"],
                s["matric_number"],
                s["department"],
                s["level"],
                s["gender"],
                s["date_registered"],
                f"{n_imgs} img(s)",
            ), tags=tag)

    def _delete(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("No Selection", "Please select a student first.")
            return
        sid  = int(self.tree.item(sel[0])["values"][0])
        name = self.tree.item(sel[0])["values"][1]
        if messagebox.askyesno("Confirm Delete",
                               f"Delete '{name}' and ALL their attendance records?"):
            db.delete_student(sid)
            # Remove face images too
            import shutil
            face_dir = os.path.join(db.DATASET_DIR, str(sid))
            if os.path.isdir(face_dir):
                shutil.rmtree(face_dir)
            self._load()
            self.app.refresh_student_list()


# ═══════════════════════════════════════════════════════════════════════════════
# Manage Courses Frame
# ═══════════════════════════════════════════════════════════════════════════════

class CoursesFrame(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=CARD)
        self.app = app
        self._build()

    def _build(self):
        tk.Label(self, text="Manage Courses",
                 font=(FONT, 16, "bold"), bg=CARD, fg=BG).pack(pady=(18, 2))
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=20, pady=8)

        form = tk.Frame(self, bg=CARD)
        form.pack(padx=30)

        labels = ["Course Code", "Course Name", "Lecturer"]
        keys   = ["code", "name", "lecturer"]
        self.vars = {}
        for i, (lbl, key) in enumerate(zip(labels, keys)):
            tk.Label(form, text=lbl + ":", font=(FONT, 10, "bold"),
                     bg=CARD, anchor="e", width=14).grid(
                row=i, column=0, padx=(0, 8), pady=6, sticky="e")
            v = tk.StringVar()
            self.vars[key] = v
            _entry(form, v, width=30).grid(row=i, column=1, pady=6, sticky="w")

        _btn(self, "➕  Add Course", self._add, bg=GREEN, w=18).pack(pady=10)

        self.status_var = tk.StringVar()
        tk.Label(self, textvariable=self.status_var,
                 font=(FONT, 10), bg=CARD, fg=GREEN).pack()

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=20, pady=10)

        cols = ("ID", "Code", "Name", "Lecturer")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=10)
        for col, w in zip(cols, [40, 80, 200, 160]):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w)
        self.tree.pack(padx=20, fill="both", expand=True)
        self._load()

    def _add(self):
        code     = self.vars["code"].get().strip()
        name     = self.vars["name"].get().strip()
        lecturer = self.vars["lecturer"].get().strip()
        if not code or not name:
            messagebox.showwarning("Missing Fields",
                                   "Course Code and Name are required.")
            return
        cid = db.add_course(code, name, lecturer)
        if cid is None:
            self.status_var.set(f"❌  Course code '{code}' already exists.")
        else:
            self.status_var.set(f"✔  Course '{code}' added (ID: {cid})")
            for v in self.vars.values():
                v.set("")
            self._load()

    def _load(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for c in db.get_all_courses():
            self.tree.insert("", "end", values=(
                c["course_id"], c["course_code"],
                c["course_name"], c["lecturer"]))


# ═══════════════════════════════════════════════════════════════════════════════
# Main Application Window
# ═══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Attendance System – Nigerian Army University, Biu")
        self.geometry("960x680")
        self.minsize(900, 620)
        self.configure(bg=BG)
        try:
            self.iconbitmap("")  # suppress default icon errors
        except Exception:
            pass

        db.init_db()
        self._build()

    # ── Layout ────────────────────────────────────────────────────────────
    def _build(self):
        # ── Header bar ───────────────────────────────────────────────────
        header = tk.Frame(self, bg=BG, height=70)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(header, text="🎓  Smart Attendance System",
                 font=(FONT, 18, "bold"), bg=BG, fg=TEXT_LT).pack(
            side="left", padx=20, pady=10)
        tk.Label(header,
                 text="Nigerian Army University, Biu  •  Dept. of Software Engineering",
                 font=(FONT, 9), bg=BG, fg=GREY).pack(
            side="left", padx=4, pady=10)

        # Gold bottom line under header
        tk.Frame(self, bg=GOLD, height=3).pack(fill="x")

        # ── Sidebar + content ────────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # Sidebar
        sidebar = tk.Frame(body, bg=PANEL, width=190)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="MENU", font=(FONT, 9, "bold"),
                 bg=PANEL, fg=GREY).pack(pady=(16, 4))

        self._active_btn = None
        self._nav_buttons = []
        nav = [
            ("📋  Register Student",  "register"),
            ("📸  Capture Faces",     "capture"),
            ("🧠  Train Model",       "train"),
            ("✅  Take Attendance",   "attendance"),
            ("📊  Reports",           "reports"),
            ("👥  Students List",     "students"),
            ("📚  Courses",           "courses"),
        ]
        for label, key in nav:
            btn = tk.Button(sidebar, text=label,
                            font=(FONT, 10), fg=TEXT_LT, bg=PANEL,
                            relief="flat", anchor="w", padx=14,
                            cursor="hand2", width=20,
                            activebackground=ACCENT, activeforeground=TEXT_LT,
                            command=lambda k=key: self._show(k))
            btn.pack(fill="x", ipady=8)
            self._nav_buttons.append((key, btn))

        # Content area
        self.content = tk.Frame(body, bg=CARD)
        self.content.pack(side="left", fill="both", expand=True)

        # Build frames
        self.frames = {
            "register":   RegisterFrame(self.content, self),
            "capture":    CaptureFrame(self.content,  self),
            "train":      TrainFrame(self.content,    self),
            "attendance": AttendanceFrame(self.content, self),
            "reports":    ReportsFrame(self.content,  self),
            "students":   StudentsFrame(self.content, self),
            "courses":    CoursesFrame(self.content,  self),
        }
        for f in self.frames.values():
            f.place(relwidth=1, relheight=1)

        self._show("register")

    def _show(self, key: str):
        self.frames[key].lift()
        # Highlight active nav button
        for k, btn in self._nav_buttons:
            btn.config(bg=ACCENT if k == key else PANEL)

    def refresh_student_list(self):
        self.frames["capture"].refresh()
        self.frames["students"]._load()

    def refresh_reports(self):
        self.frames["reports"]._load()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    db.init_db()
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()