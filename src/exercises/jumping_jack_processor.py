# -*- coding: utf-8 -*-
from src.exercises.base_processor import BaseExerciseProcessor
from src.utils.geometry import calculate_angle

class JumpingJackProcessor(BaseExerciseProcessor):
    def __init__(self, thresholds=None):
        if thresholds is None:
            thresholds = {'down': 45, 'up': 150}
        super().__init__(thresholds)

    def process(self, landmarks, frame_size):
        # 1. Get Keypoints (Right Side) to measure arm elevation
        hip = landmarks[24]
        shoulder = landmarks[12]
        wrist = landmarks[16]

        # 2. Calculate Angle (Arm relative to torso)
        angle = calculate_angle(hip, shoulder, wrist)
        status_color = (0, 255, 0) # Green (Default)

        # ข้อมูลระยะการแยกขา
        d_ankles = abs(landmarks[27][0] - landmarks[28][0])
        d_shoulders = abs(landmarks[11][0] - landmarks[12][0])
        leg_ratio = d_ankles / d_shoulders if d_shoulders > 0 else 1.0

        # ตรวจจับมุมมองด้านข้าง (Side/Profile View)
        torso_height = (abs(landmarks[11][1] - landmarks[23][1]) + abs(landmarks[12][1] - landmarks[24][1])) / 2
        is_side_view = (d_shoulders / torso_height < 0.35) if torso_height > 0 else False

        # 3. State Machine Logic
        if angle < self.thresholds['down']:
            if self.current_phase == 3:
                self.reps_count += 1
                self.feedback = f"ครั้งที่ {self.reps_count} สำเร็จ!"
            else:
                self.feedback = "Stand straight - Ready"
            self.current_phase = 0
        elif angle > self.thresholds['up']:
            self.current_phase = 2
            if not is_side_view and leg_ratio < 1.2:
                self.feedback = "[!] ยกมือขึ้นสุดแล้ว แต่ยังไม่ได้แยกขา!"
                status_color = (0, 0, 255)
            else:
                self.feedback = "Hands above head!" + ("" if is_side_view else " Leg opened")
                status_color = (255, 255, 0) 
        elif self.thresholds['down'] <= angle <= self.thresholds['up']:
            if self.current_phase == 0 or self.current_phase == 1:
                self.current_phase = 1
                self.feedback = "Jump out! Open legs"
            elif self.current_phase == 2 or self.current_phase == 3:
                self.current_phase = 3
                self.feedback = "Jump in! Close legs"

        return {
            "count": self.reps_count,
            "phase": self.current_phase,
            "feedback": self.feedback,
            "status_color": status_color,
            "debug_angle": round(angle, 2)
        }
