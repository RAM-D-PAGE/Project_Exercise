"""
Video Review Display (ระบบตรวจสอบความถูกต้องจากคลิปวิดีโอ)
==========================================================
ช่วยให้ผู้ใช้โหลดวิดีโอ แล้วดูผลการวิเคราะห์ของ AI แบบ frame-by-frame
เพื่อเช็คว่า Label / Rep / Phase ถูกต้องหรือไม่ ก่อน/หลังเทรนโมเดล

คีย์ลัด:
  Space   = Play / Pause
  D / →   = Forward 1 frame (ขณะ Pause)
  A / ←   = Back 1 frame (ขณะ Pause)
  ] / +   = เพิ่มความเร็ว
  [ / -   = ลดความเร็ว
  1       = สลับเป็น Mode A (Geometry)
  2       = สลับเป็น Mode C (Bi-LSTM)
  R       = Reset (กลับ frame 0, reset counter)
  Q       = ออก

วิธีใช้: python src/review_video.py
"""

import cv2
import os
import sys
import glob
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog

# --- Add Project Root to Path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Reconfigure stdout for UTF-8 Emojis ---
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# --- Defensive MediaPipe Import ---
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except (ImportError, ModuleNotFoundError, AttributeError):
    try:
        from mediapipe.python.solutions import pose as mp_pose  # type: ignore
        from mediapipe.python.solutions import drawing_utils as mp_drawing  # type: ignore
    except:
        print("ERROR: MediaPipe not found. Please run 'pip install mediapipe'")
        sys.exit(1)

# --- Engine & Utility Imports (รองรับรันจาก root และจาก src/) ---
try:
    from src.engines.geometry_engine import GeometryEngine
    from src.engines.lstm_engine import LSTMEngine
    from src.utils.geometry import calculate_angle
except ImportError:
    from engines.geometry_engine import GeometryEngine
    from engines.lstm_engine import LSTMEngine
    from utils.geometry import calculate_angle

# ============================================================
# CONFIGURATION
# ============================================================

EXERCISE_OPTIONS = {
    '1': {'name': 'Squat_Correct',       'display': 'Squat (Correct)'},
    '2': {'name': 'Squat_Incorrect',     'display': 'Squat (Incorrect)'},
    '3': {'name': 'Jumping_Jack',        'display': 'Jumping Jack'},
    '4': {'name': 'Pushup',              'display': 'Push-up'},
}

MODES = {
    '1': {'name': 'Mode A: Geometry',  'key': 'a'},
    '2': {'name': 'Mode C: Bi-LSTM',   'key': 'c'},
}

MODEL_PATH = "models/exercise_bilstm.tflite"
TIME_STEPS = 15

SPEED_OPTIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
DEFAULT_SPEED_IDX = 3  # 1.0x


# ============================================================
# MENU SYSTEM
# ============================================================

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    clear_screen()
    print("=" * 60)
    print("  🖥️  VIDEO REVIEW DISPLAY")
    print("  ตรวจสอบความถูกต้องของ AI จากคลิปวิดีโอ")
    print("=" * 60)


def menu_select_exercise():
    """เมนูเลือกท่าออกกำลังกาย"""
    print("\n  📋 เลือกท่าออกกำลังกายที่ตรงกับวิดีโอ:\n")
    for key, info in EXERCISE_OPTIONS.items():
        print(f"    [{key}] {info['display']}")
    print()

    while True:
        choice = input("  👉 กดเลข (1-4): ").strip()
        if choice in EXERCISE_OPTIONS:
            selected = EXERCISE_OPTIONS[choice]
            print(f"\n  ✅ เลือก: {selected['display']}")
            return selected
        print("  ❌ กรุณากดเลข 1-4 เท่านั้น")


def menu_select_mode():
    """เมนูเลือก Mode การวิเคราะห์"""
    print("\n  🔧 เลือก Mode เริ่มต้น (สลับได้ระหว่างรีวิว):\n")
    for key, info in MODES.items():
        print(f"    [{key}] {info['name']}")
    print()

    while True:
        choice = input("  👉 กดเลข (1-2): ").strip()
        if choice in MODES:
            selected = MODES[choice]
            print(f"\n  ✅ เลือก: {selected['name']}")
            return selected
        print("  ❌ กรุณากดเลข 1 หรือ 2 เท่านั้น")


def menu_select_input():
    """เมนูเลือกวิธีนำเข้าวิดีโอ"""
    print("\n  📂 เลือกวิธีนำเข้าวิดีโอ:\n")
    print("    [1] เลือกไฟล์วิดีโอ (เปิดหน้าต่างเลือกไฟล์)")
    print("    [2] สแกนจาก data/raw_videos/ อัตโนมัติ")
    print()

    while True:
        choice = input("  👉 กดเลข (1-2): ").strip()
        if choice in ['1', '2']:
            return choice
        print("  ❌ กรุณากดเลข 1 หรือ 2 เท่านั้น")


def get_video_file_dialog():
    """เปิดหน้าต่าง File Dialog ให้เลือกวิดีโอ"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="เลือกไฟล์วิดีโอที่ต้องการรีวิว",
        filetypes=[
            ("Video Files", "*.mp4 *.avi *.mov *.MP4 *.AVI *.MOV"),
            ("All Files", "*.*")
        ]
    )
    root.destroy()
    return file_path


def scan_raw_videos(exercise_name):
    """สแกนวิดีโอจาก data/raw_videos/"""
    search_dirs = [
        os.path.join("data", "raw_videos", exercise_name),
        os.path.join("data", "raw_videos"),
    ]

    video_files = []
    for search_dir in search_dirs:
        if os.path.isdir(search_dir):
            for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']:
                video_files.extend(glob.glob(os.path.join(search_dir, "**", ext), recursive=True))
            if video_files:
                break

    # ขจัดไฟล์ซ้ำและแปลง path ให้มีรูปแบบที่เป็นมาตรฐานเดียวกัน
    video_files = sorted(list(set(os.path.normpath(f) for f in video_files)))

    if not video_files:
        print(f"\n  ❌ ไม่พบวิดีโอใน data/raw_videos/")
        return None

    print(f"\n  🔍 พบวิดีโอ {len(video_files)} ไฟล์:\n")
    for i, f in enumerate(video_files, 1):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"    [{i}] {os.path.relpath(f)} ({size_mb:.1f} MB)")
    print(f"\n    [0] ยกเลิก")
    print()

    while True:
        choice = input("  👉 กดเลข: ").strip()
        if choice == '0':
            return None
        elif choice.isdigit() and 1 <= int(choice) <= len(video_files):
            return video_files[int(choice) - 1]
        print("  ❌ กรุณาเลือกตัวเลขที่ถูกต้อง")


# ============================================================
# ENGINE FACTORY
# ============================================================

def create_engine(mode_key, exercise_name):
    """สร้าง Engine ตาม Mode ที่เลือก"""
    if mode_key == 'a':
        return GeometryEngine(exercise_name)
    elif mode_key == 'c':
        return LSTMEngine(exercise_name, model_path=MODEL_PATH, time_steps=TIME_STEPS)
    else:
        return GeometryEngine(exercise_name)


# ============================================================
# DRAWING / HUD
# ============================================================

def draw_progress_bar(frame, current_frame, total_frames, y_pos):
    """วาด progress bar แสดงตำแหน่ง frame ปัจจุบัน"""
    h, w, _ = frame.shape
    bar_x = 15
    bar_w = w - 30
    bar_h = 24

    # Background
    cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_w, y_pos + bar_h), (50, 50, 50), -1)

    # Fill
    if total_frames > 0:
        fill_w = int(bar_w * (current_frame / total_frames))
        cv2.rectangle(frame, (bar_x, y_pos), (bar_x + fill_w, y_pos + bar_h), (0, 200, 255), -1)

    # Border
    cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_w, y_pos + bar_h), (100, 100, 100), 1)

    # Percentage text
    if total_frames > 0:
        pct = current_frame / total_frames * 100
        cv2.putText(frame, f"{pct:.0f}%", (bar_x + bar_w - 65, y_pos + bar_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_thai_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """
    วาดข้อความภาษาไทย/อังกฤษ ลงบนภาพ OpenCV (numpy array) โดยใช้ Pillow และจัดลำดับการโหลดฟอนต์
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        font_paths = [
            # 1. Local Font ในโครงการ
            os.path.join(os.path.dirname(__file__), "utils", "Sarabun-Regular.ttf"),
            # 2. ค้นหาชื่อฟอนต์โดยตรง (PIL จะค้นหาใน System Font Directory)
            "tahoma",
            "LeelawUI",
            "arial",
            # 3. Path มาตรฐานบน Windows
            "C:\\Windows\\Fonts\\tahoma.ttf",
            "C:\\Windows\\Fonts\\LeelawUI.ttf",
            "C:\\Windows\\Fonts\\arial.ttf"
        ]
        font = None
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except:
                continue
                
        if font is None:
            font = ImageFont.load_default()
            
        rgb_color = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=rgb_color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        import traceback
        print(f"  ⚠️ [draw_thai_text ERROR]: {e}")
        traceback.print_exc()
        # Fallback กรณีดึง PIL ไม่ได้
        cv2.putText(img, text, (position[0], position[1] + font_size), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_size/32.0, color, 2, cv2.LINE_AA)
        return img

def draw_hud(frame, engine_result, mode_name, exercise_display, current_frame,
             total_frames, video_fps, speed, is_paused):
    """วาด HUD (Heads-Up Display) ทั้งหมดบน frame — ขนาดใหญ่อ่านง่าย"""
    h, w, _ = frame.shape
    overlay = frame.copy()

    # ─── TOP PANEL ───
    panel_h = 160
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Row 1: Mode & Exercise
    cv2.putText(frame, f"Mode: {mode_name}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    cv2.putText(frame, f"Exercise: {exercise_display}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)

    # Row 2: Frame info & speed
    status_icon = "II PAUSED" if is_paused else ">> PLAYING"
    status_clr = (0, 180, 255) if is_paused else (0, 255, 100)
    cv2.putText(frame, f"Frame: {current_frame}/{total_frames}", (20, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, f"FPS: {video_fps:.0f}  Speed: {speed:.2f}x", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 2)

    # Status (right side)
    text_size = cv2.getTextSize(status_icon, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    cv2.putText(frame, status_icon, (w - text_size[0] - 20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_clr, 3)

    # Confidence + Predicted class (for LSTM mode, right side)
    if 'confidence' in engine_result:
        conf = engine_result['confidence']
        conf_color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.4 else (0, 0, 255)
        cv2.putText(frame, f"Confidence: {conf:.0%}", (w - 360, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, conf_color, 2)
    if 'predicted_class' in engine_result:
        cv2.putText(frame, f"Predicted: {engine_result['predicted_class']}", (w - 360, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 200, 100), 2)

    # ─── PROGRESS BAR ───
    draw_progress_bar(frame, current_frame, total_frames, panel_h + 5)

    # ─── BOTTOM PANEL ───
    bottom_h = 150
    panel_y = h - bottom_h
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, panel_y), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay2, 0.82, frame, 0.18, 0, frame)

    # Rep Count
    count = engine_result.get('count', 0)
    cv2.putText(frame, f"REPS: {count}", (20, panel_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 3)

    # Feedback (รองรับการแสดงผลภาษาไทยด้วย Pillow)
    feedback = engine_result.get('feedback', '')
    status_color = engine_result.get('status_color', (255, 255, 255))
    frame = draw_thai_text(frame, feedback, (20, panel_y + 70), font_size=32, color=status_color)

    # Debug angle (Geometry mode)
    if 'debug_angle' in engine_result:
        cv2.putText(frame, f"Angle: {engine_result['debug_angle']}", (w - 350, panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    # Phase indicator
    phase = engine_result.get('phase', 0)
    phase_x = w - 350
    cv2.putText(frame, "Phase:", (phase_x, panel_y + 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (180, 180, 180), 2)
    for i in range(4):
        cx = phase_x + 95 + i * 42
        cy = panel_y + 92
        color = (0, 255, 0) if i <= phase else (60, 60, 60)
        cv2.circle(frame, (cx, cy), 15, color, -1)
        cv2.circle(frame, (cx, cy), 15, (100, 100, 100), 1)

    # ─── KEY HINTS (very bottom) ───
    hints = "[Space]=Play/Pause  [A/D]=Frame  [+/-]=Speed  [1]=GeoMode  [2]=LSTM  [R]=Reset  [Q]=Quit"
    cv2.putText(frame, hints, (15, panel_y + 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 2)

    return frame


# ============================================================
# MAIN REVIEW LOOP
# ============================================================

def review_video(video_path, exercise_info, initial_mode):
    """Main review loop — เล่นวิดีโอพร้อม AI analysis overlay"""
    exercise_name = exercise_info['name']
    exercise_display = exercise_info['display']
    current_mode_key = initial_mode['key']
    current_mode_name = initial_mode['name']

    # ท่า Push-up รันแบบเรขาคณิต (Geometry) เท่านั้น ไม่มีอยู่ในโมเดล LSTM 4 คลาส
    if exercise_name == 'Pushup' and current_mode_key == 'c':
        current_mode_key = 'a'
        current_mode_name = 'Mode A: Geometry'
        print("  ⚠️ [WARNING] Push-up only supports Mode A (Geometry). Forcing Mode A.")

    # สร้าง Engine
    engine = create_engine(current_mode_key, exercise_name)

    # MediaPipe Pose
    pose = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # เปิดวิดีโอ
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ ไม่สามารถเปิดวิดีโอ: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0

    print(f"\n  📹 เปิด: {os.path.basename(video_path)}")
    print(f"     Frames: {total_frames}  |  FPS: {video_fps:.0f}")
    print(f"     Mode: {current_mode_name}  |  Exercise: {exercise_display}")
    print(f"\n  ⏯ เริ่มเล่น! (กด Space เพื่อ Pause)\n")

    # State
    is_paused = False
    speed_idx = DEFAULT_SPEED_IDX
    current_speed = SPEED_OPTIONS[speed_idx]
    current_frame_idx = 0

    # สำหรับเก็บ frame cache (ใช้สำหรับ backward seek)
    # เก็บเฉพาะ frame ปัจจุบัน (backward ใช้ cap.set)
    frame_cache = None

    # Stats tracking
    stats = {
        'total_processed': 0,
        'no_pose_frames': 0,
        'max_reps': 0,
        'low_confidence_frames': 0,
        'mode_switches': 0,
    }

    # สร้างหน้าต่าง (ขยายใหญ่ให้ดูง่าย)
    window_name = 'Video Review Display'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while True:
        need_new_frame = False

        if not is_paused:
            need_new_frame = True
        
        if need_new_frame:
            ret, frame = cap.read()
            if not ret:
                # จบวิดีโอ
                print("\n  🏁 จบวิดีโอแล้ว!")
                break
            current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame_cache = frame.copy()
        else:
            # Paused — ใช้ frame เดิม
            if frame_cache is not None:
                frame = frame_cache.copy()
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                frame_cache = frame.copy()

        h_orig, w_orig = frame.shape[:2]

        # --- Pose Detection ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        engine_result = {
            "count": engine.reps_count,
            "phase": engine.current_phase,
            "feedback": "No pose detected",
            "status_color": (100, 100, 100)
        }

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # --- Run Engine ---
            if current_mode_key == 'a':
                lm_points = [[lm.x, lm.y] for lm in landmarks]
                engine_result = engine.process(lm_points, (w_orig, h_orig))
            elif current_mode_key == 'c':
                engine_result = engine.process(landmarks, (w_orig, h_orig))

            stats['total_processed'] += 1
            if 'confidence' in engine_result and engine_result['confidence'] < 0.5:
                stats['low_confidence_frames'] += 1
        else:
            stats['no_pose_frames'] += 1 if need_new_frame else 0
            # หากตรวจไม่พบกระดูก (หลุดขอบกล้อง) ให้เคลียร์สถานะ Active ชั่วคราวป้องกันการนับผิดพลาด
            if hasattr(engine, 'reset_active_state'):
                engine.reset_active_state()

        stats['max_reps'] = max(stats['max_reps'], engine_result.get('count', 0))

        # --- Resize to 1280x720 FIRST (ก่อนวาด HUD และ Skeleton) ---
        DISPLAY_W, DISPLAY_H = 1280, 720
        display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H),
                                   interpolation=cv2.INTER_LINEAR)

        # --- Draw Skeleton on the resized display_frame (for consistent thickness) ---
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                display_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=3)
            )

        # --- Draw HUD (วาดบน display_frame ขนาดคงที่ 1280x720) ---
        display_frame = draw_hud(
            display_frame, engine_result,
            current_mode_name, exercise_display,
            current_frame_idx, total_frames,
            video_fps, current_speed, is_paused
        )

        cv2.imshow(window_name, display_frame)

        # --- Wait key (ควบคุม speed) ---
        if is_paused:
            wait_ms = 0  # Block until key press
        else:
            base_delay = int(1000.0 / video_fps)
            wait_ms = max(1, int(base_delay / current_speed))

        key = cv2.waitKey(wait_ms) & 0xFF

        # --- Handle Keys ---
        if key == ord('q') or key == 27:  # Q or ESC
            print("\n  ⏹ ออกจากรีวิว")
            break

        elif key == ord(' '):  # Space = toggle pause
            is_paused = not is_paused
            state_text = "⏸ Paused" if is_paused else "▶ Playing"
            print(f"  {state_text} (Frame {current_frame_idx}/{total_frames})")

        elif key == ord('d') or key == 83:  # D or → = forward 1 frame
            if is_paused:
                ret, frame = cap.read()
                if ret:
                    current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    frame_cache = frame.copy()

        elif key == ord('a') or key == 81:  # A or ← = back 1 frame
            if is_paused:
                new_pos = max(0, current_frame_idx - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                ret, frame = cap.read()
                if ret:
                    current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    frame_cache = frame.copy()
                    # Reset engine สำหรับ backward (เพราะ LSTM buffer อาจเพี้ยน)
                    if current_mode_key == 'c':
                        engine.reset()

        elif key == ord(']') or key == ord('+') or key == ord('='):  # Speed up
            if speed_idx < len(SPEED_OPTIONS) - 1:
                speed_idx += 1
                current_speed = SPEED_OPTIONS[speed_idx]
                print(f"  ⏩ Speed: {current_speed}x")

        elif key == ord('[') or key == ord('-'):  # Speed down
            if speed_idx > 0:
                speed_idx -= 1
                current_speed = SPEED_OPTIONS[speed_idx]
                print(f"  ⏪ Speed: {current_speed}x")

        elif key == ord('1'):  # Switch to Mode A (Geometry)
            if current_mode_key != 'a':
                current_mode_key = 'a'
                current_mode_name = 'Mode A: Geometry'
                engine = create_engine('a', exercise_name)
                stats['mode_switches'] += 1
                print(f"  🔄 เปลี่ยนเป็น: {current_mode_name}")

        elif key == ord('2'):  # Switch to Mode C (Bi-LSTM)
            if current_mode_key != 'c':
                # ป้องกันการสลับเป็น Mode C (Bi-LSTM) สำหรับท่า Push-up
                if exercise_name == 'Pushup':
                    print("  ⚠️ [WARNING] Push-up does not support Mode C (LSTM). Switch ignored.")
                else:
                    current_mode_key = 'c'
                    current_mode_name = 'Mode C: Bi-LSTM'
                    engine = create_engine('c', exercise_name)
                    stats['mode_switches'] += 1
                    print(f"  🔄 เปลี่ยนเป็น: {current_mode_name}")

        elif key == ord('r'):  # Reset
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_frame_idx = 0
            engine = create_engine(current_mode_key, exercise_name)
            frame_cache = None
            is_paused = True
            stats = {
                'total_processed': 0, 'no_pose_frames': 0,
                'max_reps': 0, 'low_confidence_frames': 0,
                'mode_switches': stats['mode_switches'],
            }
            print("  🔄 Reset! กลับ frame 0 (Paused)")

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    # --- สรุปผล ---
    print_summary(video_path, exercise_display, stats, engine)


def print_summary(video_path, exercise_display, stats, engine):
    """แสดงสรุปผลหลังจบการรีวิว"""
    print()
    print("=" * 60)
    print("  📊 สรุปผลการรีวิว")
    print("=" * 60)
    print(f"  วิดีโอ       : {os.path.basename(video_path)}")
    print(f"  ท่า          : {exercise_display}")
    print(f"  จำนวน Rep    : {engine.reps_count}")
    print(f"  Frames ที่วิเคราะห์ : {stats['total_processed']}")
    print(f"  Frames ไม่พบ Pose   : {stats['no_pose_frames']}")
    if stats['low_confidence_frames'] > 0:
        print(f"  ⚠ Frames Confidence ต่ำ (<50%): {stats['low_confidence_frames']}")
    if stats['mode_switches'] > 0:
        print(f"  สลับ Mode    : {stats['mode_switches']} ครั้ง")
    print("=" * 60)

    if stats['total_processed'] > 0:
        pose_ratio = stats['total_processed'] / (stats['total_processed'] + stats['no_pose_frames']) * 100
        print(f"  Pose Detection Rate: {pose_ratio:.1f}%")

    print()


# ============================================================
# MAIN
# ============================================================

def main():
    print_banner()

    # ขั้นตอนที่ 1: เลือกท่า
    exercise_info = menu_select_exercise()

    # ขั้นตอนที่ 2: เลือก Mode เริ่มต้น
    mode_info = menu_select_mode()

    # ขั้นตอนที่ 3: เลือกวิดีโอ
    input_mode = menu_select_input()

    video_path = None
    if input_mode == '1':
        print("\n  📂 กำลังเปิดหน้าต่างเลือกไฟล์...")
        video_path = get_video_file_dialog()
    elif input_mode == '2':
        video_path = scan_raw_videos(exercise_info['name'])

    if not video_path or not os.path.isfile(video_path):
        print("\n  ❌ ไม่ได้เลือกวิดีโอ ออกจากโปรแกรม")
        return

    # สรุปการตั้งค่า
    print("\n" + "-" * 60)
    print(f"  ท่า        : {exercise_info['display']}")
    print(f"  Mode       : {mode_info['name']}")
    print(f"  วิดีโอ     : {os.path.basename(video_path)}")
    print("-" * 60)
    input("\n  ✅ กด Enter เพื่อเริ่มรีวิว...")

    # เริ่มรีวิว
    review_video(video_path, exercise_info, mode_info)

    # ถามว่าต้องการรีวิวอีกรอบไหม
    print()
    again = input("  🔄 ต้องการรีวิวอีกรอบไหม? (กด Enter = ออก, R = รีวิวอีก): ").strip().upper()
    if again == 'R':
        main()


if __name__ == "__main__":
    main()
