import cv2
import numpy as np
import time
import sys
import os

# --- Defensive MediaPipe Import ---
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except (ImportError, ModuleNotFoundError):
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
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
    cv2.rectangle(overlay, (0, 0), (420, 110), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, f"Mode: {mode_name}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(frame, f"Exercise: {exercise_name}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Confidence (only for LSTM mode)
    if 'confidence' in engine_result:
        conf = engine_result['confidence']
        conf_color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.4 else (0, 0, 255)
        cv2.putText(frame, f"Confidence: {conf:.0%}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
    if 'predicted_class' in engine_result:
        cv2.putText(frame, f"Predicted: {engine_result['predicted_class']}", (220, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

    # --- Bottom Panel (Rep Counter & Feedback) ---
    panel_y = h - 100
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, panel_y), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

    # Rep Count (large)
    count = engine_result.get('count', 0)
    cv2.putText(frame, f"REPS: {count}", (15, panel_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # Feedback
    feedback = engine_result.get('feedback', '')
    status_color = engine_result.get('status_color', (255, 255, 255))
    cv2.putText(frame, feedback, (15, panel_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Debug angle (for Geometry mode)
    if 'debug_angle' in engine_result:
        cv2.putText(frame, f"Angle: {engine_result['debug_angle']}", (w - 180, panel_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Phase indicator
    phase = engine_result.get('phase', 0)
    phase_x_start = w - 180
    cv2.putText(frame, "Phase:", (phase_x_start, panel_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    for i in range(4):
        cx = phase_x_start + 60 + i * 25
        cy = panel_y + 70
        color = (0, 255, 0) if i <= phase else (80, 80, 80)
        cv2.circle(frame, (cx, cy), 8, color, -1)

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

        # --- Draw UI ---
        mode_name = MODES[current_mode][0]
        exercise_name = EXERCISES[current_exercise]
        frame = draw_ui_panel(frame, engine_result, mode_name, exercise_name, fps)

        cv2.imshow('AI Exercise Tracker', frame)

        # --- Handle Keys ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            engine.reset()
            print("[INFO] Counter reset.")

        elif chr(key) in EXERCISES:
            current_exercise = chr(key)
            engine = create_engine(current_mode, EXERCISES[current_exercise])
            print(f"[INFO] Exercise changed to: {EXERCISES[current_exercise]}")

        elif chr(key) in MODES:
            current_mode = chr(key)
            engine = create_engine(current_mode, EXERCISES[current_exercise])
            print(f"[INFO] Mode changed to: {MODES[current_mode][0]}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()
