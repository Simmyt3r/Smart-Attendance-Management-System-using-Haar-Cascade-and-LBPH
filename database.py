"""
database.py – all functions use try/finally to guarantee conn.close()
"""

import sqlite3
import os
from datetime import date, datetime

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DB_PATH      = os.path.join(BASE_DIR, "attendance.db")
DATASET_DIR  = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR  = os.path.join(BASE_DIR, "trainer")
EXPORTS_DIR  = os.path.join(BASE_DIR, "exports")
MODEL_PATH   = os.path.join(TRAINER_DIR, "trainer.pkl")

for _d in (DATASET_DIR, TRAINER_DIR, EXPORTS_DIR):
    os.makedirs(_d, exist_ok=True)


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name       TEXT    NOT NULL,
    matric_number   TEXT    UNIQUE NOT NULL,
    department      TEXT    NOT NULL DEFAULT 'Software Engineering',
    level           TEXT    NOT NULL DEFAULT '100',
    gender          TEXT    NOT NULL DEFAULT 'Male',
    date_registered TEXT    NOT NULL DEFAULT (date('now'))
);
CREATE TABLE IF NOT EXISTS courses (
    course_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code TEXT    UNIQUE NOT NULL,
    course_name TEXT    NOT NULL,
    lecturer    TEXT    NOT NULL DEFAULT 'Unknown'
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id    INTEGER NOT NULL,
    session_date TEXT    NOT NULL,
    start_time   TEXT    NOT NULL,
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);
CREATE TABLE IF NOT EXISTS attendance (
    attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id    INTEGER NOT NULL,
    session_id    INTEGER NOT NULL,
    time_in       TEXT    NOT NULL,
    confidence    REAL    NOT NULL DEFAULT 0.0,
    status        TEXT    NOT NULL DEFAULT 'Present',
    UNIQUE (student_id, session_id),
    FOREIGN KEY (student_id)  REFERENCES students(student_id),
    FOREIGN KEY (session_id)  REFERENCES sessions(session_id)
);
"""


def init_db():
    conn = get_conn()
    try:
        conn.executescript(SCHEMA)
        conn.execute("""INSERT OR IGNORE INTO courses (course_code, course_name, lecturer)
            VALUES ('SEP401','Software Engineering Project','Dr. [Supervisor Name]')""")
        conn.commit()
    finally:
        conn.close()


# ── Students ──────────────────────────────────────────────────────────────────

def add_student(full_name, matric_number,
                department="Software Engineering", level="100", gender="Male"):
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO students (full_name,matric_number,department,level,gender) "
            "VALUES (?,?,?,?,?)",
            (full_name.strip(), matric_number.strip().upper(), department, level, gender)
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        conn.rollback()
        return None
    finally:
        conn.close()


def get_all_students():
    conn = get_conn()
    try:
        return [dict(r) for r in
                conn.execute("SELECT * FROM students ORDER BY full_name").fetchall()]
    finally:
        conn.close()


def get_student_by_id(student_id):
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM students WHERE student_id=?", (student_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_student_name(student_id):
    s = get_student_by_id(student_id)
    return s["full_name"] if s else "Unknown"


def delete_student(student_id):
    conn = get_conn()
    try:
        conn.execute("DELETE FROM attendance WHERE student_id=?", (student_id,))
        conn.execute("DELETE FROM students    WHERE student_id=?", (student_id,))
        conn.commit()
    finally:
        conn.close()


def student_has_face_data(student_id):
    d = os.path.join(DATASET_DIR, str(student_id))
    return os.path.isdir(d) and len(os.listdir(d)) > 0


def count_face_images(student_id):
    d = os.path.join(DATASET_DIR, str(student_id))
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith((".jpg",".png"))])


# ── Courses ───────────────────────────────────────────────────────────────────

def add_course(course_code, course_name, lecturer=""):
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO courses (course_code,course_name,lecturer) VALUES (?,?,?)",
            (course_code.strip().upper(), course_name.strip(), lecturer.strip())
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        conn.rollback()
        return None
    finally:
        conn.close()


def get_all_courses():
    conn = get_conn()
    try:
        return [dict(r) for r in
                conn.execute("SELECT * FROM courses ORDER BY course_code").fetchall()]
    finally:
        conn.close()


# ── Sessions ──────────────────────────────────────────────────────────────────

def create_session(course_id):
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO sessions (course_id,session_date,start_time) VALUES (?,?,?)",
            (course_id, str(date.today()), datetime.now().strftime("%H:%M:%S"))
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


# ── Attendance ────────────────────────────────────────────────────────────────

def record_attendance(student_id, session_id, confidence):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO attendance (student_id,session_id,time_in,confidence) "
            "VALUES (?,?,?,?)",
            (student_id, session_id,
             datetime.now().strftime("%H:%M:%S"), round(confidence, 2))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        conn.rollback()
        return False
    finally:
        conn.close()


def get_session_attendance(session_id):
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT s.full_name, s.matric_number, a.time_in, a.confidence, a.status
            FROM   attendance a
            JOIN   students   s ON s.student_id = a.student_id
            WHERE  a.session_id=?
            ORDER BY a.time_in
        """, (session_id,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_attendance_report(course_id=None, start_date=None, end_date=None):
    conn = get_conn()
    try:
        params, where_clauses = [], []
        if course_id:
            where_clauses.append("sess.course_id=?"); params.append(course_id)
        if start_date:
            where_clauses.append("sess.session_date>=?"); params.append(start_date)
        if end_date:
            where_clauses.append("sess.session_date<=?"); params.append(end_date)
        w = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        sql = f"""
            SELECT st.student_id, st.full_name, st.matric_number, st.level,
                COUNT(DISTINCT sess.session_id) AS total_sessions,
                COUNT(DISTINCT a.session_id)   AS sessions_attended,
                ROUND(100.0*COUNT(DISTINCT a.session_id)
                      /MAX(COUNT(DISTINCT sess.session_id),1),1) AS attendance_pct
            FROM students st
            CROSS JOIN sessions sess
            LEFT JOIN attendance a
                   ON a.student_id=st.student_id AND a.session_id=sess.session_id
            {w}
            GROUP BY st.student_id ORDER BY st.full_name
        """
        rows = conn.execute(sql, params).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            pct = d["attendance_pct"] or 0
            d["status"] = ("Eligible" if pct >= 75
                           else "At Risk" if pct >= 65 else "Barred")
            result.append(d)
        return result
    finally:
        conn.close()


def get_detailed_attendance(course_id=None, start_date=None, end_date=None):
    conn = get_conn()
    try:
        params, where = [], []
        if course_id: where.append("sess.course_id=?"); params.append(course_id)
        if start_date: where.append("sess.session_date>=?"); params.append(start_date)
        if end_date: where.append("sess.session_date<=?"); params.append(end_date)
        w = ("WHERE " + " AND ".join(where)) if where else ""
        rows = conn.execute(f"""
            SELECT st.full_name, st.matric_number, c.course_code,
                   sess.session_date, a.time_in, a.confidence, a.status
            FROM   attendance a
            JOIN   students   st   ON st.student_id  = a.student_id
            JOIN   sessions   sess ON sess.session_id = a.session_id
            JOIN   courses    c    ON c.course_id     = sess.course_id
            {w}
            ORDER BY sess.session_date DESC, a.time_in DESC
        """, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()