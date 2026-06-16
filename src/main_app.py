import cv2
import numpy as np
import time
import sys
import os

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
except (ImportError, ModuleNotFoundError):
    try:
        from mediapipe.python.solutions import pose as mp_pose  # type: ignore
        from mediapipe.python.solutions import drawing_utils as mp_drawing  # type: ignore
    except:
        print("ERROR: MediaPipe installation appears corrupted. Please run 'pip install mediapipe'")
        sys.exit(1)

from src.engines.geometry_engine import GeometryEngine
from src.engines.lstm_engine import LSTMEngine

# --- Configuration ---
EXERCISES = {
    '1': 'Squat_Correct',
    '2': 'Pushup',
    '3': 'Jumping_Jack',
}

MODES = {
    'a': ('Mode A: Geometry', 'geometry'),
    'c': ('Mode C: Bi-LSTM', 'lstm'),
}

DEFAULT_MODE = 'a'
DEFAULT_EXERCISE = '1'
MODEL_PATH = "models/exercise_bilstm.tflite"
TIME_STEPS = 15


def create_engine(mode_key, exercise_name):
    """Factory function to create the appropriate engine."""
    if mode_key == 'a':
        return GeometryEngine(exercise_name)
    elif mode_key == 'c':
        return LSTMEngine(exercise_name, model_path=MODEL_PATH, time_steps=TIME_STEPS)
    else:
        return GeometryEngine(exercise_name)


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

def draw_ui_panel(frame, engine_result, mode_name, exercise_name, fps):
    """
    Draws a HUD (Heads-Up Display) panel on the frame showing:
    - Current mode and exercise
    - Rep count, phase, feedback
    - FPS and confidence (for LSTM)
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # --- Top Panel (Mode & Exercise Info) ---
    # ขยายกล่องบนให้กว้างและสูงขึ้นเพื่อรองรับตัวหนังสือขนาดใหญ่ HD 720p
    cv2.rectangle(overlay, (0, 0), (480, 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, f"Mode: {mode_name}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Exercise: {exercise_name}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    # Confidence (only for LSTM mode)
    if 'confidence' in engine_result:
        conf = engine_result['confidence']
        conf_color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.4 else (0, 0, 255)
        cv2.putText(frame, f"Confidence: {conf:.0%}", (15, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2, cv2.LINE_AA)
    if 'predicted_class' in engine_result:
        cv2.putText(frame, f"Predicted: {engine_result['predicted_class']}", (240, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2, cv2.LINE_AA)

    # --- Bottom Panel (Rep Counter & Feedback) ---
    # ปรับขยับ HUD ล่างให้อ่านฟีดแบ็กขนาดใหญ่ชัดเจนขึ้น
    panel_y = h - 110
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, panel_y), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

    # Rep Count (large)
    count = engine_result.get('count', 0)
    cv2.putText(frame, f"REPS: {count}", (20, panel_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)

    # Feedback (รองรับการแสดงผลภาษาไทยด้วย Pillow)
    feedback = engine_result.get('feedback', '')
    status_color = engine_result.get('status_color', (255, 255, 255))
    frame = draw_thai_text(frame, feedback, (20, panel_y + 60), font_size=28, color=status_color)

    # Debug angle (for Geometry mode)
    if 'debug_angle' in engine_result:
        cv2.putText(frame, f"Angle: {engine_result['debug_angle']}", (w - 220, panel_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)

    # Phase indicator
    phase = engine_result.get('phase', 0)
    phase_x_start = w - 220
    cv2.putText(frame, "Phase:", (phase_x_start, panel_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2, cv2.LINE_AA)
    for i in range(4):
        cx = phase_x_start + 80 + i * 30
        cy = panel_y + 80
        color = (0, 255, 0) if i <= phase else (80, 80, 80)
        cv2.circle(frame, (cx, cy), 10, color, -1)

    return frame


def print_controls():
    """Prints the control instructions to the console."""
    print("\n" + "=" * 55)
    print("  AI Exercise Tracker - Real-time Analysis")
    print("=" * 55)
    print("\n  MODE SELECTION:")
    for key, (name, _) in MODES.items():
        print(f"    [{key.upper()}] {name}")
    print("\n  EXERCISE SELECTION:")
    for key, name in EXERCISES.items():
        print(f"    [{key}] {name}")
    print("\n  CONTROLS:")
    print("    [R] Reset counter")
    print("    [Q] Quit")
    print("=" * 55 + "\n")


def main():
    print_controls()

    # --- Initialize ---
    current_mode = DEFAULT_MODE
    current_exercise = DEFAULT_EXERCISE
    engine = create_engine(current_mode, EXERCISES[current_exercise])

    # MediaPipe Pose
    pose = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    # ตั้งค่าความละเอียดกล้องเว็บแคมให้เป็น HD 1280x720 เพื่อความชัดเจนตามเป้าหมายของระบบ
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror
        h, w, _ = frame.shape

        # --- FPS Calculation ---
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

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

            # Draw Skeleton
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)
            )

            # --- Run Engine ---
            if current_mode == 'a':
                # Geometry engine expects list of [x, y] per landmark
                lm_points = [[lm.x, lm.y] for lm in landmarks]
                engine_result = engine.process(lm_points, (w, h))
            elif current_mode == 'c':
                # LSTM engine extracts features internally from raw landmarks
                engine_result = engine.process(landmarks, (w, h))
        else:
            # หากตรวจไม่พบกระดูก (หลุดขอบกล้อง) ให้เคลียร์สถานะ Active ชั่วคราวเพื่อกันบั๊กเดินเข้ากล้องแล้วนับ 1
            if hasattr(engine, 'reset_active_state'):
                engine.reset_active_state()

        # --- Draw UI ---
        mode_name = MODES[current_mode][0]
        exercise_name = EXERCISES[current_exercise]
        frame = draw_ui_panel(frame, engine_result, mode_name, exercise_name, fps)

        cv2.imshow('AI Exercise Tracker', frame)

        # --- Handle Keys ---
        key = cv2.waitKey(1) & 0xFF
        key_char = chr(key).lower() if 32 <= key <= 126 else ""

        if key == ord('q') or key_char == 'q':
            break

        elif key == ord('r') or key_char == 'r':
            engine.reset()
            print("[INFO] Counter reset.")

        elif key_char in EXERCISES:
            selected_exercise = EXERCISES[key_char]
            # ท่า Push-up จะทำงานบน Mode A (Geometry) เท่านั้น เนื่องจากไม่มีอยู่ในโมเดล LSTM 4 คลาส
            if selected_exercise == 'Pushup' and current_mode == 'c':
                current_mode = 'a'
                print("⚠️ [WARNING] Push-up only supports Mode A (Geometry). Forcing Mode A.")
            
            current_exercise = key_char
            engine = create_engine(current_mode, selected_exercise)
            print(f"[INFO] Exercise changed to: {selected_exercise}")

        elif key_char in MODES:
            selected_mode = key_char
            # ป้องกันการสลับเป็น Mode C (Bi-LSTM) สำหรับท่า Push-up
            if selected_mode == 'c' and EXERCISES[current_exercise] == 'Pushup':
                print("⚠️ [WARNING] Push-up does not support Mode C (LSTM). Switch ignored.")
            else:
                current_mode = selected_mode
                engine = create_engine(current_mode, EXERCISES[current_exercise])
                print(f"[INFO] Mode changed to: {MODES[current_mode][0]}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()
