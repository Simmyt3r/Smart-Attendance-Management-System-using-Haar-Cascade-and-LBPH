"""
test_system.py – automated tests for all non-GUI modules.
Run:  python test_system.py
"""

import os, sys, shutil, tempfile
import numpy as np, cv2

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import database as db

TMP = tempfile.mkdtemp(prefix="sas_test_")
db.DB_PATH     = os.path.join(TMP, "test.db")
db.DATASET_DIR = os.path.join(TMP, "dataset")
db.TRAINER_DIR = os.path.join(TMP, "trainer")
db.EXPORTS_DIR = os.path.join(TMP, "exports")
db.MODEL_PATH  = os.path.join(TMP, "trainer", "trainer.pkl")
for d in (db.DATASET_DIR, db.TRAINER_DIR, db.EXPORTS_DIR):
    os.makedirs(d, exist_ok=True)

import train_model as tm
tm.DATASET_DIR = db.DATASET_DIR
tm.MODEL_PATH  = db.MODEL_PATH

import face_recognizer as fr

# ── runner ────────────────────────────────────────────────────────────────────
PASSED = FAILED = 0

def test(name, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  ✔  {name}")
        PASSED += 1
    except Exception as e:
        print(f"  ✘  {name}\n       {e}")
        FAILED += 1

def ok(cond, msg=""): 
    if not cond: raise AssertionError(msg or "condition failed")
def eq(a, b, msg=""):
    if a != b: raise AssertionError(f"Expected {b!r}, got {a!r}. {msg}")
def nn(v, msg=""):
    if v is None: raise AssertionError(f"Got None. {msg}")


# ── synthetic face ────────────────────────────────────────────────────────────
def mf(label, noise=0.01):
    """Make a reproducible 100×100 grayscale test face."""
    rng  = np.random.default_rng(label * 100)
    base = np.zeros((100, 100), dtype=np.uint8)
    for r in range(5, 50, 8):
        val = (label * 37 + r * 13) % 255
        cv2.circle(base, (50, 50), r, int(val), 2)
    noise_arr = (rng.random((100, 100)) * noise * 255).astype(np.uint8)
    return cv2.add(base, noise_arr)


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATABASE
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Database ──────────────────────────────────────────────────────────")

test("init_db creates all 4 tables", lambda: (
    db.init_db(),
    ok({r[0] for r in db.get_conn().execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()} >=
       {"students","courses","sessions","attendance"})
))

test("add_student returns positive id", lambda: ok(
    db.add_student("Alice A","MAT/001","SE","300","Female") > 0))

test("add_student duplicate returns None", lambda: eq(
    db.add_student("Alice B","MAT/001"), None))

test("get_all_students returns list", lambda: ok(
    len(db.get_all_students()) >= 1))

def _t_get_by_id():
    sid = db.add_student("Bob B","MAT/002")
    s = db.get_student_by_id(sid)
    nn(s); eq(s["full_name"], "Bob B")
test("get_student_by_id returns correct record", _t_get_by_id)

def _t_name():
    sid = db.add_student("Carol C","MAT/003")
    eq(db.get_student_name(sid), "Carol C")
test("get_student_name returns correct name", _t_name)

test("get_student_name Unknown for missing id", lambda:
    eq(db.get_student_name(99999), "Unknown"))

def _t_course():
    nn(db.add_course("CS101","Computer Science","Dr A"))
test("add_course returns id", _t_course)

test("add_course duplicate returns None", lambda:
    eq(db.add_course("CS101","Dup","Dr B"), None))

test("get_all_courses contains CS101", lambda:
    ok("CS101" in [c["course_code"] for c in db.get_all_courses()]))

def _t_session():
    cid = db.add_course("CS202","Sessioning","Dr S")
    ok(db.create_session(cid) > 0)
test("create_session returns positive id", _t_session)

def _t_att():
    sid = db.add_student("Dave D","MAT/010")
    cid = db.add_course("CS303","Att Course","Dr T")
    ssid = db.create_session(cid)
    ok(db.record_attendance(sid, ssid, 45.0), "first insert should be True")
test("record_attendance returns True on first insert", _t_att)

def _t_dup_att():
    conn = db.get_conn()
    try:
        sess = conn.execute("SELECT session_id FROM sessions ORDER BY session_id DESC LIMIT 1").fetchone()
        stud = conn.execute("SELECT student_id FROM students WHERE matric_number='MAT/010'").fetchone()
    finally:
        conn.close()
    eq(db.record_attendance(stud["student_id"], sess["session_id"], 50.0), False)
test("record_attendance returns False for duplicate", _t_dup_att)

test("count_face_images returns 0 for no images", lambda:
    eq(db.count_face_images(db.add_student("NoFace","MAT/099")), 0))

def _t_delete():
    sid = db.add_student("Del Me","MAT/DEL")
    db.delete_student(sid)
    eq(db.get_student_by_id(sid), None)
test("delete_student removes student", _t_delete)


# ═════════════════════════════════════════════════════════════════════════════
# 2. FACE RECOGNIZER
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Face Recognizer ──────────────────────────────────────────────────")

test("factory returns a recogniser", lambda: nn(fr.LBPHFaceRecognizer_create()))

test("train() completes without error", lambda: (
    lambda rec: rec.train([mf(i) for i in range(3) for _ in range(5)],
                          np.array([i for i in range(3) for _ in range(5)])
    ))(fr.LBPHFaceRecognizer_create()))

def _t_predict_known():
    rec = fr.LBPHFaceRecognizer_create(threshold=85.0)
    rec.train([mf(0, 0.01)]*20, np.array([0]*20))
    lbl, conf = rec.predict(mf(0, 0.01))
    eq(lbl, 0, f"got {lbl} conf={conf:.2f}")
test("predict() returns correct label for known face", _t_predict_known)

def _t_multi():
    rec = fr.LBPHFaceRecognizer_create(threshold=85.0)
    imgs, lbls = [], []
    for l in range(5):
        for _ in range(20):
            imgs.append(mf(l, 0.01)); lbls.append(l)
    rec.train(imgs, np.array(lbls))
    correct = sum(1 for l in range(5) if rec.predict(mf(l, 0.01))[0] == l)
    ok(correct/5 >= 0.8, f"accuracy {correct/5:.0%} < 80%")
test("predict() ≥80% accuracy across 5 faces", _t_multi)

def _t_save_load():
    path = os.path.join(TMP, "m.pkl")
    rec = fr.LBPHFaceRecognizer_create(threshold=85.0)
    rec.train([mf(7,0.01)]*15, np.array([7]*15))
    rec.write(path)
    ok(os.path.isfile(path))
    rec2 = fr.LBPHFaceRecognizer_create()
    rec2.read(path)
    lbl, _ = rec2.predict(mf(7, 0.01))
    eq(lbl, 7)
test("write() and read() persist and restore model", _t_save_load)

test("predict() accepts BGR input", lambda: (
    lambda rec, gray: (
        rec.train([gray]*15, np.array([2]*15)),
        eq(rec.predict(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))[0], 2)
    )
)(fr.LBPHFaceRecognizer_create(threshold=85.0), mf(2, 0.01)))

test("predict() returns -1, 100.0 when empty", lambda: (
    lambda r: (eq(r[0], -1), eq(r[1], 100.0))
)(fr.LBPHFaceRecognizer_create().predict(mf(0))))


# ═════════════════════════════════════════════════════════════════════════════
# 3. TRAINING MODULE
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Training Module ──────────────────────────────────────────────────")

def _t_no_data():
    if os.path.isdir(db.DATASET_DIR): shutil.rmtree(db.DATASET_DIR)
    os.makedirs(db.DATASET_DIR)
    ok2, msg = tm.train_model()
    eq(ok2, False); ok("No face data" in msg or "No valid" in msg)
test("train_model fails gracefully with no data", _t_no_data)

def _write_dataset(n_students=3, n_imgs=15):
    for sid in range(1, n_students+1):
        d = os.path.join(db.DATASET_DIR, str(sid))
        os.makedirs(d, exist_ok=True)
        for i in range(n_imgs):
            cv2.imwrite(os.path.join(d, f"face_{i+1:03d}.png"), mf(sid, 0.02))

def _t_train_data():
    _write_dataset()
    ok2, msg = tm.train_model()
    ok(ok2, f"train failed: {msg}")
    ok(os.path.isfile(db.MODEL_PATH))
test("train_model() succeeds and creates trainer.pkl", _t_train_data)

def _t_loadable():
    rec = fr.LBPHFaceRecognizer_create()
    rec.read(db.MODEL_PATH)
    lbl, _ = rec.predict(mf(1, 0.02))
    ok(lbl in (1, 2, 3, -1))
test("Saved model is loadable and makes valid predictions", _t_loadable)


# ═════════════════════════════════════════════════════════════════════════════
# 4. ATTENDANCE REPORTS
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Attendance Reports ───────────────────────────────────────────────")

test("get_attendance_report() returns list", lambda:
     ok(isinstance(db.get_attendance_report(), list)))

def _t_report_status():
    s1 = db.add_student("Report A","RPT/001")
    s2 = db.add_student("Report B","RPT/002")
    cid = db.add_course("RPT101","Report Course","Dr R")
    nn(cid)
    # s1 attends 4 out of 4 sessions (100% → Eligible)
    for _ in range(4):
        sess = db.create_session(cid)
        db.record_attendance(s1, sess, 40.0)
    # s2 attends 1 out of 5 sessions (20% → Barred)
    sess5 = db.create_session(cid)
    db.record_attendance(s2, sess5, 50.0)

    rows  = db.get_attendance_report(course_id=cid)
    by_matric = {r["matric_number"]: r for r in rows}
    ok("RPT/001" in by_matric)
    ok("RPT/002" in by_matric)
    eq(by_matric["RPT/001"]["status"], "Eligible",
       f"got {by_matric['RPT/001']['status']}")
    eq(by_matric["RPT/002"]["status"], "Barred",
       f"got {by_matric['RPT/002']['status']}")
test("get_attendance_report() calculates status correctly", _t_report_status)

test("get_detailed_attendance() returns list with keys", lambda: (
    lambda rows: (
        ok(isinstance(rows, list)),
        ok(not rows or "full_name" in rows[0])
    )
)(db.get_detailed_attendance()))


# ═════════════════════════════════════════════════════════════════════════════
# 5. FULL PIPELINE INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Integration – Full Pipeline ──────────────────────────────────────")

def _t_pipeline():
    # 1. Register two students
    s1 = db.add_student("Alice Pipeline","INT/P001","SE","400","Female")
    s2 = db.add_student("Bob Pipeline",  "INT/P002","SE","300","Male")
    nn(s1); nn(s2)

    # 2. Write synthetic faces – use student_id as label for the image generator
    #    so each student gets a UNIQUE face pattern
    for sid in (s1, s2):
        d = os.path.join(db.DATASET_DIR, str(sid))
        os.makedirs(d, exist_ok=True)
        for i in range(20):
            cv2.imwrite(os.path.join(d, f"face_{i+1:03d}.png"), mf(sid, 0.01))

    # 3. Retrain on all students (including previous test students)
    ok2, msg = tm.train_model()
    ok(ok2, f"Training failed: {msg}")

    # 4. Load model and predict – same face pattern used in training
    rec = fr.LBPHFaceRecognizer_create(threshold=85.0)
    rec.read(db.MODEL_PATH)
    pred1, conf1 = rec.predict(mf(s1, 0.01))
    pred2, conf2 = rec.predict(mf(s2, 0.01))
    eq(pred1, s1, f"s1: expected {s1}, got {pred1} (conf={conf1:.2f})")
    eq(pred2, s2, f"s2: expected {s2}, got {pred2} (conf={conf2:.2f})")

    # 5. Record attendance
    cid  = db.add_course("INT501","Integration","Dr Int")
    sess = db.create_session(cid)
    ok(db.record_attendance(s1, sess, conf1))
    ok(db.record_attendance(s2, sess, conf2))

    # 6. Verify session summary
    summary = db.get_session_attendance(sess)
    names = {r["full_name"] for r in summary}
    ok("Alice Pipeline" in names)
    ok("Bob Pipeline"   in names)

test("Full pipeline: register→capture→train→predict→attend→report", _t_pipeline)


# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*58}")
print(f"  Results:  {PASSED} passed  |  {FAILED} failed  |  {PASSED+FAILED} total")
print(f"{'─'*58}\n")

shutil.rmtree(TMP, ignore_errors=True)
if FAILED:
    sys.exit(1)
