# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import sys

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
            os.path.join(os.path.dirname(__file__), "Sarabun-Regular.ttf"),
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


def draw_live_hud(frame, engine_result, mode_name, exercise_name, fps, is_recording=False):
    """
    วาดหน้าต่างควบคุม HUD (Heads-Up Display) สำหรับกล้องเว็บแคม (Live Tracker)
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # --- Top Panel (Mode & Exercise Info) ---
    cv2.rectangle(overlay, (0, 0), (480, 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, f"Mode: {mode_name}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Exercise: {exercise_name}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    # Sound status
    from src.utils.speech_layer import is_speech_muted
    muted = is_speech_muted()
    sound_text = "🔇 Sound: OFF" if muted else "🔊 Sound: ON"
    sound_color = (100, 100, 255) if muted else (0, 255, 150)
    cv2.putText(frame, sound_text, (240, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, sound_color, 2, cv2.LINE_AA)

    # Confidence (only for LSTM mode)
    if 'confidence' in engine_result:
        conf = engine_result['confidence']
        conf_color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.4 else (0, 0, 255)
        cv2.putText(frame, f"Confidence: {conf:.0%}", (15, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2, cv2.LINE_AA)
    if 'predicted_class' in engine_result:
        cv2.putText(frame, f"Predicted: {engine_result['predicted_class']}", (240, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2, cv2.LINE_AA)

    # Recording indicator
    if is_recording:
        cv2.circle(frame, (w - 120, 30), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 100, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # --- Bottom Panel (Rep Counter & Feedback) ---
    panel_y = h - 120
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, panel_y), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

    # Rep Count (large)
    count = engine_result.get('count', 0)
    cv2.putText(frame, f"REPS: {count}", (20, panel_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)

    # Feedback
    feedback = engine_result.get('feedback', '')
    status_color = engine_result.get('status_color', (255, 255, 255))
    frame = draw_thai_text(frame, feedback, (20, panel_y + 60), font_size=28, color=status_color)

    # Debug angle
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

    # Key hints at the very bottom
    hints = "[R]=Reset  [Q]=Quit  [M]=Mute  [S]=Record  [K]=Calibrate  [A/C]=Modes  [1/2/3]=Exercises"
    cv2.putText(frame, hints, (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)

    return frame


def draw_review_hud(frame, engine_result, mode_name, exercise_display, current_frame,
                    total_frames, video_fps, speed, is_paused, is_recording=False):
    """
    วาดหน้าต่างควบคุม HUD (Heads-Up Display) สำหรับเล่นวิดีโอรีวิว (Video Review)
    """
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

    # Row 2: Frame info & speed & sound status
    from src.utils.speech_layer import is_speech_muted
    muted = is_speech_muted()
    sound_status = "Muted" if muted else "Sound On"
    cv2.putText(frame, f"Frame: {current_frame}/{total_frames}", (20, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, f"FPS: {video_fps:.0f}  Speed: {speed:.2f}x  |  {sound_status}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 2)

    # Status (right side)
    status_icon = "II PAUSED" if is_paused else ">> PLAYING"
    status_clr = (0, 180, 255) if is_paused else (0, 255, 100)
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

    # Recording indicator
    if is_recording:
        cv2.circle(frame, (w - 280, 40), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 260, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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

    # Feedback
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
    hints = "[Space]=Play/Pause  [A/D]=Frame  [+/-]=Speed  [1]=GeoMode  [2]=LSTM  [R]=Reset  [Q]=Quit  [M]=Mute  [S]=Record"
    cv2.putText(frame, hints, (15, panel_y + 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 2)

    return frame

