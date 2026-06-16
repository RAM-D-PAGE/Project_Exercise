import cv2
import numpy as np
import pandas as pd
import os
import time
import sys
from tkinter import filedialog
import tkinter as tk

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

from utils.geometry import calculate_angle

# --- Configuration ---
EXERCISE_CLASSES = {
    '0': 'Idle',
    '1': 'Squat_Correct',
    '2': 'Squat_Incorrect',
    '3': 'Jumping_Jack'
}

BASE_DATA_DIR = "data"
LANDMARK_DIR = os.path.join(BASE_DATA_DIR, "landmarks")
IMAGE_DIR = os.path.join(BASE_DATA_DIR, "raw_images")
VIDEO_DIR = os.path.join(BASE_DATA_DIR, "raw_videos")

FPS_TARGET = 20  # Lock to 20 FPS for consistent time-steps

# --- Initialization ---
pose_tracker = mp_pose.Pose(
    model_complexity=1, # Increased to 1 for better balance between accuracy and CPU
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def main():
    # Ensure base directories exist before starting
    os.makedirs(LANDMARK_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    # ตั้งค่าความละเอียดกล้องเว็บแคมให้เป็น HD 1280x720 เพื่อความชัดเจนตามเป้าหมายของระบบ
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    current_class = '0'
    recording = False
    data_buffer = []
    
    # Hidden root for tkinter dialogs
    root = tk.Tk()
    root.withdraw()
    
    # Video Writer & Session vars
    video_writer = None
    session_id = ""
    session_img_dir = ""
    frame_count = 0
    timestamp_str = ""
    
    print("AI Exercise Data Collector (Triple-Mode + Features)")
    print("--------------------------------------------------")
    print(f"Data will be saved to: {BASE_DATA_DIR}")
    for key, label in EXERCISE_CLASSES.items():
        print(f"  {key}: {label}")
    print("  Space: Toggle Recording (Webcam)")
    print("  U: Upload/Select Local Images/Videos")
    print("  Q: Quit")
    
    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Maintain constant FPS
        curr_time = time.time()
        if curr_time - prev_time < 1./FPS_TARGET:
            continue
        prev_time = curr_time

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy() # Frame to show/save
        h, w, _ = frame.shape
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_tracker.process(rgb_frame)
        
        # UI Elements
        status_color = (0, 0, 255) if recording else (0, 255, 0)
        status_text = f"REC: {EXERCISE_CLASSES[current_class]}" if recording else "READY"
        
        # --------------------------------------------------
        # --- Core Frame Processing Logic (Shared) ---
        # --------------------------------------------------
        display_frame, landmarks_row = process_frame_logic(frame, EXERCISE_CLASSES[current_class], timestamp_str, frame_count + 1)
        
        if recording and landmarks_row:
            frame_count += 1
            # 1. Save Image (.jpg) - Raw Frame
            img_name = f"{session_id}_f{frame_count:04d}.jpg"
            cv2.imwrite(os.path.join(session_img_dir, img_name), frame)
            
            # 2. Write to Video (.mp4) - Raw Frame
            video_writer.write(frame)
            
            # 3. Buffer Landmarks + Engineered Features for CSV
            data_buffer.append(landmarks_row)

        # Draw UI

        # Draw UI
        # ขยายกล่อง HUD ให้กว้างและสูงขึ้นเพื่อรองรับฟอนต์ขนาดใหญ่ HD 720p
        cv2.rectangle(display_frame, (0, 0), (480, 120), (0, 0, 0), -1)
        cv2.putText(display_frame, status_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"Frames: {len(data_buffer)}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Triple-Mode Data Collection', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif chr(key) in EXERCISE_CLASSES:
            current_class = chr(key)
        elif key == ord('u'):
            # --- FILE UPLOAD MODE ---
            file_paths = filedialog.askopenfilenames(
                title="Select Exercise Images or Videos",
                filetypes=[("Media Files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov")]
            )
            
            if file_paths:
                print(f"--- Processing {len(file_paths)} files as a new session ---")
                # Start a mock recording session
                data_buffer = []
                frame_count = 0
                timestamp_str = str(int(time.time()))
                session_id = f"upload_{timestamp_str}"
                class_name = EXERCISE_CLASSES[current_class]
                
                session_img_dir = os.path.join(IMAGE_DIR, class_name, session_id)
                os.makedirs(session_img_dir, exist_ok=True)
                os.makedirs(os.path.join(LANDMARK_DIR, class_name), exist_ok=True)

                for f_path in file_paths:
                    ext = os.path.splitext(f_path)[1].lower()
                    
                    if ext in ['.mp4', '.avi', '.mov']:
                        # Process Video File
                        v_cap = cv2.VideoCapture(f_path)
                        while v_cap.isOpened():
                            v_ret, v_frame = v_cap.read()
                            if not v_ret: break
                            
                            frame_count += 1
                            # Reuse logic
                            res_frame, row = process_frame_logic(v_frame, class_name, timestamp_str, frame_count)
                            if row:
                                data_buffer.append(row)
                                # Save image
                                img_name = f"{session_id}_f{frame_count:04d}.jpg"
                                cv2.imwrite(os.path.join(session_img_dir, img_name), v_frame)
                            
                            cv2.imshow('Processing Upload...', res_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'): break
                        v_cap.release()
                    else:
                        # Process Image File
                        i_frame = cv2.imread(f_path)
                        if i_frame is not None:
                            frame_count += 1
                            res_frame, row = process_frame_logic(i_frame, class_name, timestamp_str, frame_count)
                            if row:
                                data_buffer.append(row)
                                img_name = f"{session_id}_f{frame_count:04d}.jpg"
                                cv2.imwrite(os.path.join(session_img_dir, img_name), i_frame)
                            
                            cv2.imshow('Processing Upload...', res_frame)
                            cv2.waitKey(100) # Small delay to see progress

                if len(data_buffer) > 0:
                    save_landmarks(data_buffer, class_name, session_id)
                    print(f"--- Upload Session Saved: {len(data_buffer)} frames recorded. ---")
                
                cv2.destroyWindow('Processing Upload...')
                data_buffer = []

        elif key == ord(' '):
            if not recording:
                # START RECORDING
                recording = True
                data_buffer = []
                frame_count = 0
                timestamp_str = str(int(time.time()))
                session_id = f"sess_{timestamp_str}"
                class_name = EXERCISE_CLASSES[current_class]
                
                # Create session directories
                session_img_dir = os.path.join(IMAGE_DIR, class_name, session_id)
                os.makedirs(session_img_dir, exist_ok=True)
                os.makedirs(os.path.join(VIDEO_DIR, class_name), exist_ok=True)
                os.makedirs(os.path.join(LANDMARK_DIR, class_name), exist_ok=True)
                
                # Init Video Writer
                video_path = os.path.join(VIDEO_DIR, class_name, f"{session_id}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                video_writer = cv2.VideoWriter(video_path, fourcc, FPS_TARGET, (w, h))
                
                print(f"--- Started recording: {class_name} ({session_id}) ---")
            else:
                # STOP RECORDING
                recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
                
                if len(data_buffer) > 0:
                    save_landmarks(data_buffer, EXERCISE_CLASSES[current_class], session_id)
                
                print(f"--- Stopped recording. Saved {len(data_buffer)} frames. ---")
                data_buffer = []

    # Final Cleanup
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

def process_frame_logic(frame, class_label, timestamp_str, frame_idx):
    """
    Processes a single frame for pose landmarks, draws them, and returns the data row.
    """
    display_frame = frame.copy()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_tracker.process(rgb_frame)
    
    landmarks_row = None
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        lm_list = [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]
        
        # Calculate Angles
        l_knee_angle = calculate_angle(
            [lm_list[23][0], lm_list[23][1]], 
            [lm_list[25][0], lm_list[25][1]], 
            [lm_list[27][0], lm_list[27][1]]
        )
        r_knee_angle = calculate_angle(
            [lm_list[24][0], lm_list[24][1]], 
            [lm_list[26][0], lm_list[26][1]], 
            [lm_list[28][0], lm_list[28][1]]
        )
        
        # Draw Skeleton
        mp_drawing.draw_landmarks(
            display_frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )
        
        # Show angles in UI
        # ปรับตำแหน่งตัวหนังสือมุมข้อต่อและขนาดฟอนต์ให้ชัดเจนขึ้น
        cv2.putText(display_frame, f"L-Knee: {int(l_knee_angle)}", (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"R-Knee: {int(r_knee_angle)}", (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Prepare Data Row
        landmarks_row = [class_label, timestamp_str, frame_idx]
        for lm in lm_list:
            landmarks_row.extend(lm)
        landmarks_row.extend([l_knee_angle, r_knee_angle])
        
    return display_frame, landmarks_row

def save_landmarks(data, class_name, session_id):
    directory = os.path.join(LANDMARK_DIR, class_name)
    filepath = os.path.join(directory, f"{session_id}.csv")
    
    # Create Columns
    cols = ['class', 'timestamp', 'frame_idx']
    for i in range(33):
        cols.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
    # Engineered Features
    cols.extend(['l_knee_angle', 'r_knee_angle'])
    
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(filepath, index=False)
    print(f"Landmarks (with features) saved to: {filepath}")

if __name__ == "__main__":
    main()
