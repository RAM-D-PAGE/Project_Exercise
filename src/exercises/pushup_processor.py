# -*- coding: utf-8 -*-
import time
import numpy as np
from src.exercises.base_processor import BaseExerciseProcessor
from src.utils.geometry import calculate_angle

class PushupProcessor(BaseExerciseProcessor):
    def __init__(self, thresholds=None):
        if thresholds is None:
            thresholds = {'plank': 160, 'bottom': 90}
        super().__init__(thresholds)

    def process(self, landmarks, frame_size):
        # 1. Get Keypoints (Right Arm)
        shoulder = landmarks[12]
        elbow = landmarks[14]
        wrist = landmarks[16]

        # 2. Calculate Angle
        angle = calculate_angle(shoulder, elbow, wrist)
        status_color = (0, 255, 0) # Green (Default)

        # วิเคราะห์สะโพกตก/โด่ง (สะโพก ไหล่ เข่า)
        hip_angle_l = calculate_angle(landmarks[11], landmarks[23], landmarks[25])
        hip_angle_r = calculate_angle(landmarks[12], landmarks[24], landmarks[26])
        avg_hip_angle = (hip_angle_l + hip_angle_r) / 2

        # 3. State Machine Logic
        if angle > self.thresholds['plank']:
            if self.current_phase == 3:
                duration = time.time() - getattr(self, 'active_start_time', time.time())
                self.reps_count += 1
                if duration < 1.4:
                    self.feedback = f"ครั้งที่ {self.reps_count} สำเร็จ! (ย่อขึ้นเร็วไปนิด ช้าลงหน่อย)"
                else:
                    self.feedback = f"ครั้งที่ {self.reps_count} สำเร็จ! จังหวะเยี่ยม"
            else:
                self.feedback = "High Plank - Ready"
            self.current_phase = 0
        elif angle < self.thresholds['bottom']:
            self.current_phase = 2
            self.feedback = "Good depth! Keep body straight"
            status_color = (255, 255, 0) 
        elif self.thresholds['bottom'] <= angle <= self.thresholds['plank']:
            if self.current_phase == 0 or self.current_phase == 1:
                if self.current_phase == 0:
                    self.active_start_time = time.time()
                self.current_phase = 1
                self.feedback = "Going down..."
            elif self.current_phase == 2 or self.current_phase == 3:
                self.current_phase = 3
                self.feedback = "Push up!"

        # ตรวจจับมุมมองด้านข้าง (Side/Profile View) สำหรับท่าวิดพื้น
        d_shoulders = abs(landmarks[11][0] - landmarks[12][0])
        torso_len = (np.hypot(landmarks[11][0] - landmarks[23][0], landmarks[11][1] - landmarks[23][1]) +
                     np.hypot(landmarks[12][0] - landmarks[24][0], landmarks[12][1] - landmarks[24][1])) / 2
        is_side_view = (d_shoulders / torso_len < 0.35) if torso_len > 0 else False

        # ค้นหาข้อผิดพลาดของหลังและสะโพกขณะทำวิดพื้น (Phase 1, 2, 3)
        if self.current_phase in [1, 2, 3]:
            if not is_side_view:
                self.feedback = "[แนะนำ] หันข้างให้กล้อง เพื่อตรวจเช็คหลังและสะโพก"
                status_color = (0, 165, 255) # Orange warning
            else:
                if avg_hip_angle < 150:
                    self.feedback = "[!] สะโพกสูงเกินไป! ลดสะโพกและก้นลงต่ำลง"
                    status_color = (0, 0, 255)
                elif avg_hip_angle > 195:
                    self.feedback = "[!] สะโพกแอ่น/ตก! เกร็งหน้าท้องรักษาแนวตรง"
                    status_color = (0, 0, 255)
                elif self.current_phase == 1 and angle > 120:
                    self.feedback = "[!] ย่อตัวลงลึกอีกนิด ให้หน้าอกใกล้พื้นอีกหน่อย"
                    status_color = (0, 0, 255)

        return {
            "count": self.reps_count,
            "phase": self.current_phase,
            "feedback": self.feedback,
            "status_color": status_color,
            "debug_angle": round(angle, 2)
        }
