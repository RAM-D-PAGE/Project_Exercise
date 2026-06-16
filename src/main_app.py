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
from src.utils.speech_layer import speak_feedback, get_speech_assistant, toggle_speech_mute
from src.utils.analytics import WorkoutAnalytics
from src.utils.ui_utils import draw_live_hud

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
    analytics = WorkoutAnalytics(EXERCISES[current_exercise], fps=30.0)
    frame_idx = 0

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

    # --- Live Recording & Calibration States ---
    is_recording = False
    video_writer = None
    calibration_active = False
    calibration_start_time = 0.0
    calibration_angles = []

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

            # --- Run Engine ---
            if current_mode == 'a':
                # Geometry engine expects list of [x, y] per landmark
                lm_points = [[lm.x, lm.y] for lm in landmarks]
                engine_result = engine.process(lm_points, (w, h))
            elif current_mode == 'c':
                # LSTM engine extracts features internally from raw landmarks
                engine_result = engine.process(landmarks, (w, h))

            # Draw Skeleton (Buffering indicator)
            is_buffering = (engine_result.get('predicted_class') == 'Buffering')
            skeleton_color = (0, 140, 255) if is_buffering else (0, 255, 0)
            connection_color = (0, 100, 200) if is_buffering else (0, 200, 0)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=skeleton_color, thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=connection_color, thickness=2)
            )
        else:
            # หากตรวจไม่พบกระดูก (หลุดขอบกล้อง) ให้เคลียร์สถานะ Active ชั่วคราวเพื่อกันบั๊กเดินเข้ากล้องแล้วนับ 1
            if hasattr(engine, 'reset_active_state'):
                engine.reset_active_state()

        # --- Auto-Calibration Logic ---
        if calibration_active:
            elapsed = time.time() - calibration_start_time
            if elapsed < 5.0:
                angle = engine_result.get('debug_angle', None)
                if angle is not None:
                    calibration_angles.append(angle)
                engine_result['feedback'] = f"🔧 บันทึกมุมข้อต่อ... ทำท่า 1 ครั้ง ({5.0 - elapsed:.1f} วินาที)"
                engine_result['status_color'] = (0, 165, 255)  # Orange
            else:
                calibration_active = False
                if calibration_angles:
                    min_ang = min(calibration_angles)
                    max_ang = max(calibration_angles)
                    ex_name = EXERCISES[current_exercise]
                    
                    if current_mode == 'a' and hasattr(engine, 'processor'):
                        msg = ""
                        if "Squat" in ex_name:
                            new_stand = max(140.0, max_ang - 10.0)
                            new_bottom = min(115.0, min_ang + 10.0)
                            engine.processor.thresholds['stand'] = new_stand
                            engine.processor.thresholds['bottom'] = new_bottom
                            msg = f"ปรับแต่งเกณฑ์สำเร็จ ยืน: {new_stand:.0f}, ย่อ: {new_bottom:.0f}"
                        elif "Pushup" in ex_name:
                            new_plank = max(140.0, max_ang - 10.0)
                            new_bottom = min(110.0, min_ang + 10.0)
                            engine.processor.thresholds['plank'] = new_plank
                            engine.processor.thresholds['bottom'] = new_bottom
                            msg = f"ปรับแต่งเกณฑ์สำเร็จ แพลงก์: {new_plank:.0f}, ย่อ: {new_bottom:.0f}"
                        elif "Jumping" in ex_name:
                            new_down = min(60.0, min_ang + 10.0)
                            new_up = max(130.0, max_ang - 10.0)
                            engine.processor.thresholds['down'] = new_down
                            engine.processor.thresholds['up'] = new_up
                            msg = f"ปรับแต่งเกณฑ์สำเร็จ หุบ: {new_down:.0f}, กาง: {new_up:.0f}"
                        
                        if msg:
                            speak_feedback(msg)
                            print(f"\n[CALIBRATION] {msg}\n")

        # --- Log frame & Voice Feedback ---
        frame_idx += 1
        angle = engine_result.get('debug_angle', 0.0)
        rep_count = engine_result.get('count', 0)
        phase = engine_result.get('phase', 0)
        status_color = engine_result.get('status_color', (255, 255, 255))
        confidence = engine_result.get('confidence', None)
        predicted_class = engine_result.get('predicted_class', None)
        analytics.log_frame(frame_idx, angle, rep_count, phase, status_color, confidence, predicted_class)

        # Asynchronous voice feedback (ไม่พูดซ้ำหากกำลังทำ calibration)
        feedback_text = engine_result.get('feedback', '')
        if not calibration_active and feedback_text and feedback_text not in ["No pose detected", "Ready", "Stand straight - Ready", "High Plank - Ready", "Idle - Ready"] and not feedback_text.startswith("Buffering"):
            speak_feedback(feedback_text)

        # --- Draw UI ---
        mode_name = MODES[current_mode][0]
        exercise_name = EXERCISES[current_exercise]
        frame = draw_live_hud(frame, engine_result, mode_name, exercise_name, fps, is_recording=is_recording)

        # --- Video Writing ---
        if is_recording and video_writer is not None:
            video_writer.write(frame)

        cv2.imshow('AI Exercise Tracker', frame)

        # --- Handle Keys ---
        key = cv2.waitKey(1) & 0xFF
        key_char = chr(key).lower() if 32 <= key <= 126 else ""

        if key == ord('q') or key_char == 'q':
            break

        elif key == ord('r') or key_char == 'r':
            engine.reset()
            print("[INFO] Counter reset.")

        elif key_char == 'm':
            toggle_speech_mute()

        elif key_char == 's':
            if not is_recording:
                os.makedirs("results", exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                record_path = f"results/live_record_{timestamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(record_path, fourcc, 20.0, (w, h))
                if video_writer.isOpened():
                    is_recording = True
                    print(f"[INFO] เริ่มบันทึกวิดีโอผลลัพธ์... ➡️ {record_path}")
                    speak_feedback("เริ่มบันทึกวิดีโอ")
                else:
                    print("⚠️ [ERROR] ไม่สามารถตั้งค่า VideoWriter ได้")
                    video_writer = None
            else:
                is_recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                print("[INFO] สิ้นสุดการบันทึกวิดีโอ")
                speak_feedback("บันทึกวิดีโอสำเร็จแล้วค่ะ")

        elif key_char == 'k':
            if current_mode != 'a':
                print("⚠️ [WARNING] การปรับแต่งเกณฑ์ (Calibration) รองรับเฉพาะโหมดเรขาคณิต (Geometry) เท่านั้น")
                speak_feedback("การปรับแต่งเกณฑ์ รองรับเฉพาะโหมดเรขาคณิตเท่านั้นค่ะ")
            else:
                calibration_active = True
                calibration_start_time = time.time()
                calibration_angles = []
                print("[INFO] เริ่มการปรับแต่งเกณฑ์ (Calibration)...")
                speak_feedback("เริ่มปรับแต่งเกณฑ์ กรุณายืนตรงแล้วย่อตัวทำท่าหนึ่งครั้งค่ะ")

        elif key_char in EXERCISES:
            selected_exercise = EXERCISES[key_char]
            # บันทึกแดชบอร์ดสถิติและ CSV ของท่าเดิมก่อนเปลี่ยน
            analytics.generate_dashboard()
            analytics.export_to_csv()

            # ท่า Push-up จะทำงานบน Mode A (Geometry) เท่านั้น เนื่องจากไม่มีอยู่ในโมเดล LSTM 4 คลาส
            if selected_exercise == 'Pushup' and current_mode == 'c':
                current_mode = 'a'
                print("⚠️ [WARNING] Push-up only supports Mode A (Geometry). Forcing Mode A.")
            
            current_exercise = key_char
            engine = create_engine(current_mode, selected_exercise)
            analytics = WorkoutAnalytics(selected_exercise, fps=fps)
            frame_idx = 0
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
    if video_writer is not None:
        video_writer.release()

    print("\n  ⌛ กำลังคำนวณและแสดงกราฟสรุปสถิติออกกำลังกาย...")
    analytics.generate_dashboard()
    analytics.export_to_csv()

    # หยุดการทำงานของเสียงแจ้งเตือน
    try:
        get_speech_assistant().stop()
    except:
        pass

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()
