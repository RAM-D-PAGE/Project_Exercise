"""
Auto-Label Video Tool (Interactive Mode)
=========================================
ใช้ Geometry Engine (Mode A) เป็น "ครู" ช่วยตัดและแปะป้ายจังหวะ Idle/Active 
จากวิดีโอยาวๆ อัตโนมัติ เพื่อสร้าง Dataset สำหรับเทรน Bi-LSTM (Mode C)

วิธีใช้: python src/auto_label_video.py
(ไม่ต้องพิมพ์อะไรเพิ่ม ระบบจะให้เลือกทุกอย่างผ่านเมนู)
"""

import cv2
import os
import sys
import glob
import numpy as np
import pandas as pd
import time
import tkinter as tk
from tkinter import filedialog

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

# === Class Mapping (ตรงกับ collect_data.py) ===
EXERCISE_OPTIONS = {
    '1': {'name': 'Squat_Correct',        'mode': 'SQUAT',             'class_id': 1},
    '2': {'name': 'Squat_Incorrect',       'mode': 'SQUAT_INCORRECT',   'class_id': 2},
    '3': {'name': 'Jumping_Jack',          'mode': 'JUMPING_JACK',      'class_id': 3},
    '4': {'name': 'Bicep_Curl_Correct',    'mode': 'BICEP_CURL',        'class_id': 4},
    '5': {'name': 'Bicep_Curl_Incorrect',  'mode': 'BICEP_CURL',        'class_id': 5},
    '6': {'name': 'Push-up',              'mode': 'PUSHUP',            'class_id': 1},
}

CLASS_NAMES = {
    0: 'Idle', 1: 'Squat_Correct', 2: 'Squat_Incorrect',
    3: 'Jumping_Jack', 4: 'Bicep_Curl_Correct', 5: 'Bicep_Curl_Incorrect',
}

FPS_TARGET = 20


# ============================================================
# MENU SYSTEM
# ============================================================

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    clear_screen()
    print("=" * 60)
    print("  🏋️  AUTO-LABEL VIDEO TOOL")
    print("  ใช้ Geometry ช่วยแปะป้ายท่าทางจากวิดีโออัตโนมัติ")
    print("=" * 60)


def menu_select_exercise():
    """เมนูเลือกท่าออกกำลังกาย"""
    print("\n  📋 เลือกท่าออกกำลังกายที่ตรงกับวิดีโอ:\n")
    for key, info in EXERCISE_OPTIONS.items():
        print(f"    [{key}] {info['name']}")
    print()

    while True:
        choice = input("  👉 กดเลข (1-6): ").strip()
        if choice in EXERCISE_OPTIONS:
            selected = EXERCISE_OPTIONS[choice]
            print(f"\n  ✅ เลือก: {selected['name']}")
            return selected
        print("  ❌ กรุณากดเลข 1-6 เท่านั้น")


def menu_select_input_mode():
    """เมนูเลือกวิธีนำเข้าวิดีโอ"""
    print("\n  📂 เลือกวิธีนำเข้าวิดีโอ:\n")
    print("    [1] เลือกไฟล์วิดีโอ (เปิดหน้าต่างเลือกไฟล์)")
    print("    [2] เลือกทั้งโฟลเดอร์ (รันทุกวิดีโอในนั้น)")
    print("    [3] สแกนจาก data/raw_videos/ อัตโนมัติ")
    print()

    while True:
        choice = input("  👉 กดเลข (1-3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("  ❌ กรุณากดเลข 1-3 เท่านั้น")


def menu_select_preview():
    """เมนูเลือกเปิด/ปิด Preview"""
    print("\n  🖥️  แสดงหน้าต่าง Preview ระหว่างประมวลผลไหม?\n")
    print("    [1] เปิด Preview (เห็นผลสดๆ แต่ช้ากว่า)")
    print("    [2] ปิด Preview (รันเร็วสุด)")
    print()

    while True:
        choice = input("  👉 กดเลข (1-2): ").strip()
        if choice == '1':
            return True
        elif choice == '2':
            return False
        print("  ❌ กรุณากดเลข 1 หรือ 2 เท่านั้น")


def get_video_files_dialog():
    """เปิดหน้าต่าง File Dialog ให้เลือกวิดีโอ"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_paths = filedialog.askopenfilenames(
        title="เลือกไฟล์วิดีโอที่ต้องการ Auto-Label",
        filetypes=[
            ("Video Files", "*.mp4 *.avi *.mov *.MP4 *.AVI *.MOV"),
            ("All Files", "*.*")
        ]
    )
    root.destroy()
    return list(file_paths)


def get_video_folder_dialog():
    """เปิดหน้าต่าง Folder Dialog ให้เลือกโฟลเดอร์"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    folder = filedialog.askdirectory(title="เลือกโฟลเดอร์ที่มีวิดีโอ")
    root.destroy()

    if not folder:
        return []

    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']:
        video_files.extend(glob.glob(os.path.join(folder, ext)))
    video_files.sort()
    return video_files


def scan_raw_videos(exercise_name):
    """สแกนวิดีโอจาก data/raw_videos/{exercise_name}/"""
    search_dir = os.path.join("data", "raw_videos", exercise_name)
    if not os.path.isdir(search_dir):
        # ลองสแกนทั้ง raw_videos
        search_dir = os.path.join("data", "raw_videos")

    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']:
        video_files.extend(glob.glob(os.path.join(search_dir, "**", ext), recursive=True))
    video_files.sort()

    if video_files:
        print(f"\n  🔍 พบวิดีโอ {len(video_files)} ไฟล์ ใน {search_dir}:\n")
        for i, f in enumerate(video_files, 1):
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"    [{i}] {os.path.relpath(f)} ({size_mb:.1f} MB)")

        print(f"\n    [A] เลือกทั้งหมด")
        print(f"    [0] ยกเลิก")
        print()

        while True:
            choice = input("  👉 กดเลข หรือ A: ").strip().upper()
            if choice == 'A':
                return video_files
            elif choice == '0':
                return []
            elif choice.isdigit() and 1 <= int(choice) <= len(video_files):
                return [video_files[int(choice) - 1]]
            print("  ❌ กรุณาเลือกตัวเลขที่ถูกต้อง")
    else:
        print(f"\n  ❌ ไม่พบวิดีโอใน {search_dir}")
        return []


# ============================================================
# PROCESSING
# ============================================================

def detect_exercise_phase(lm_list, exercise_mode):
    """
    ใช้มุมทางเรขาคณิตเพื่อจำแนกว่าเฟรมนี้เป็น Idle หรือ Active
    """
    if exercise_mode in ('SQUAT', 'SQUAT_INCORRECT'):
        l_knee = calculate_angle(
            [lm_list[23][0], lm_list[23][1]],
            [lm_list[25][0], lm_list[25][1]],
            [lm_list[27][0], lm_list[27][1]]
        )
        r_knee = calculate_angle(
            [lm_list[24][0], lm_list[24][1]],
            [lm_list[26][0], lm_list[26][1]],
            [lm_list[28][0], lm_list[28][1]]
        )
        avg_knee = (l_knee + r_knee) / 2
        return avg_knee < 160, avg_knee, "Knee"

    elif exercise_mode == 'PUSHUP':
        l_elbow = calculate_angle(
            [lm_list[11][0], lm_list[11][1]],
            [lm_list[13][0], lm_list[13][1]],
            [lm_list[15][0], lm_list[15][1]]
        )
        r_elbow = calculate_angle(
            [lm_list[12][0], lm_list[12][1]],
            [lm_list[14][0], lm_list[14][1]],
            [lm_list[16][0], lm_list[16][1]]
        )
        avg_elbow = (l_elbow + r_elbow) / 2
        return avg_elbow < 150, avg_elbow, "Elbow"

    elif exercise_mode == 'JUMPING_JACK':
        l_arm = calculate_angle(
            [lm_list[23][0], lm_list[23][1]],
            [lm_list[11][0], lm_list[11][1]],
            [lm_list[15][0], lm_list[15][1]]
        )
        r_arm = calculate_angle(
            [lm_list[24][0], lm_list[24][1]],
            [lm_list[12][0], lm_list[12][1]],
            [lm_list[16][0], lm_list[16][1]]
        )
        avg_arm = (l_arm + r_arm) / 2
        return avg_arm > 45, avg_arm, "Arm"

    elif exercise_mode == 'BICEP_CURL':
        l_elbow = calculate_angle(
            [lm_list[11][0], lm_list[11][1]],
            [lm_list[13][0], lm_list[13][1]],
            [lm_list[15][0], lm_list[15][1]]
        )
        r_elbow = calculate_angle(
            [lm_list[12][0], lm_list[12][1]],
            [lm_list[14][0], lm_list[14][1]],
            [lm_list[16][0], lm_list[16][1]]
        )
        avg_elbow = (l_elbow + r_elbow) / 2
        return avg_elbow < 120, avg_elbow, "Elbow"

    return False, 0.0, "N/A"


def process_single_video(video_path, exercise_mode, active_class_id, show_preview=True):
    """ประมวลผลวิดีโอ 1 ไฟล์"""
    pose_tracker = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ ไม่สามารถเปิดวิดีโอ: {video_path}")
        return [], {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"  📹 ไฟล์: {os.path.basename(video_path)} ({total_frames} frames, {video_fps:.0f} FPS)")

    data_buffer = []
    frame_count = 0
    idle_count = 0
    active_count = 0
    skip_count = 0
    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        if curr_time - prev_time < 1.0 / FPS_TARGET:
            continue
        prev_time = curr_time

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_tracker.process(rgb_frame)

        if results.pose_landmarks:
            lm_list = [[lm.x, lm.y, lm.z, lm.visibility]
                       for lm in results.pose_landmarks.landmark]

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

            is_active, debug_angle, angle_name = detect_exercise_phase(lm_list, exercise_mode)

            current_class = active_class_id if is_active else 0
            label_name = exercise_mode if is_active else "Idle"
            color = (0, 255, 0) if is_active else (180, 180, 180)

            if is_active:
                active_count += 1
            else:
                idle_count += 1

            frame_count += 1
            row = [current_class, str(int(time.time())), frame_count]
            for lm in lm_list:
                row.extend(lm)
            row.extend([l_knee_angle, r_knee_angle])
            data_buffer.append(row)

            if show_preview:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)
                )
                cv2.rectangle(frame, (0, 0), (520, 80), (0, 0, 0), -1)
                cv2.putText(frame, f"Label: {label_name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"{angle_name}: {int(debug_angle)} | Frame: {frame_count}/{total_frames}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Resize ให้หน้าต่างไม่ใหญ่เกินไป (จำกัดความกว้างไม่เกิน 640px)
                h_orig, w_orig = frame.shape[:2]
                max_width = 640
                if w_orig > max_width:
                    scale = max_width / w_orig
                    display_frame = cv2.resize(frame, (max_width, int(h_orig * scale)))
                else:
                    display_frame = frame
                cv2.imshow('Auto-Labeling (Q = Stop)', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("  ⏹ หยุดโดยผู้ใช้")
                    break
        else:
            skip_count += 1

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()
    pose_tracker.close()

    stats = {
        'total_frames': frame_count,
        'idle_frames': idle_count,
        'active_frames': active_count,
        'skipped_frames': skip_count,
    }
    return data_buffer, stats


def save_to_csv(data_buffer, output_path):
    """บันทึก data_buffer ลงไฟล์ CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cols = ['class', 'timestamp', 'frame_idx']
    for i in range(33):
        cols.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
    cols.extend(['l_knee_angle', 'r_knee_angle'])

    df = pd.DataFrame(data_buffer, columns=cols)
    df.to_csv(output_path, index=False)
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print_banner()

    # ขั้นตอนที่ 1: เลือกท่าออกกำลังกาย
    exercise_info = menu_select_exercise()
    exercise_mode = exercise_info['mode']
    active_class_id = exercise_info['class_id']
    exercise_name = exercise_info['name']

    # ขั้นตอนที่ 2: เลือกวิธีนำเข้าวิดีโอ
    input_mode = menu_select_input_mode()

    video_files = []
    if input_mode == '1':
        print("\n  📂 กำลังเปิดหน้าต่างเลือกไฟล์...")
        video_files = get_video_files_dialog()
    elif input_mode == '2':
        print("\n  📂 กำลังเปิดหน้าต่างเลือกโฟลเดอร์...")
        video_files = get_video_folder_dialog()
    elif input_mode == '3':
        video_files = scan_raw_videos(exercise_name)

    if not video_files:
        print("\n  ❌ ไม่ได้เลือกวิดีโอ ออกจากโปรแกรม")
        return

    print(f"\n  📹 จะประมวลผลวิดีโอ {len(video_files)} ไฟล์")

    # ขั้นตอนที่ 3: เลือกเปิด/ปิด Preview
    show_preview = menu_select_preview()

    # สรุปการตั้งค่าก่อนเริ่ม
    print("\n" + "-" * 60)
    print(f"  ท่า        : {exercise_name}")
    print(f"  Class ID   : {active_class_id} ({CLASS_NAMES.get(active_class_id, '?')})")
    print(f"  วิดีโอ     : {len(video_files)} ไฟล์")
    print(f"  Preview    : {'เปิด' if show_preview else 'ปิด'}")
    print("-" * 60)
    input("\n  ✅ กด Enter เพื่อเริ่มประมวลผล...")

    # ประมวลผล
    print("\n" + "=" * 60)
    print("  🚀 เริ่มประมวลผล...")
    print("=" * 60 + "\n")

    total_stats = {'total_frames': 0, 'idle_frames': 0, 'active_frames': 0, 'skipped_frames': 0}
    saved_files = []

    for i, vpath in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] ---")
        data, stats = process_single_video(vpath, exercise_mode, active_class_id, show_preview)

        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

        if data:
            video_name = os.path.splitext(os.path.basename(vpath))[0]
            csv_path = os.path.join("data", "landmarks", exercise_name, f"auto_{video_name}.csv")
            df = save_to_csv(data, csv_path)
            saved_files.append(csv_path)
            print(f"  💾 บันทึก: {csv_path} ({len(df)} เฟรม)\n")

    # สรุปผล
    print("\n" + "=" * 60)
    print("  📊 สรุปผลการ Auto-Label")
    print("=" * 60)
    print(f"  Total Labeled Frames : {total_stats['total_frames']}")
    print(f"  Idle Frames          : {total_stats['idle_frames']}")
    print(f"  Active Frames        : {total_stats['active_frames']}")
    print(f"  Skipped (No Pose)    : {total_stats['skipped_frames']}")
    if total_stats['total_frames'] > 0:
        ratio = total_stats['active_frames'] / total_stats['total_frames'] * 100
        print(f"  Active Ratio         : {ratio:.1f}%")
    print(f"\n  📁 ไฟล์ที่บันทึก ({len(saved_files)} ไฟล์):")
    for f in saved_files:
        print(f"     • {f}")
    print("=" * 60)
    print("  ✅ เสร็จสิ้น! พร้อมนำไปเทรนด้วย: python -m src.train_model")

    # ถามว่าต้องการรันอีกรอบไหม
    print()
    again = input("  🔄 ต้องการรันอีกรอบไหม? (กด Enter = ออก, กด R = รันอีก): ").strip().upper()
    if again == 'R':
        main()


if __name__ == "__main__":
    main()
