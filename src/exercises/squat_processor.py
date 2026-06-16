# -*- coding: utf-8 -*-
import time
from src.exercises.base_processor import BaseExerciseProcessor
from src.utils.geometry import calculate_angle

class SquatProcessor(BaseExerciseProcessor):
    def __init__(self, thresholds=None):
        if thresholds is None:
            thresholds = {'stand': 160, 'bottom': 100}
        super().__init__(thresholds)

    def process(self, landmarks, frame_size):
        # 1. Get Keypoints (Left Leg)
        hip = landmarks[23]
        knee = landmarks[25]
        ankle = landmarks[27]

        # 2. Calculate Angle
        angle = calculate_angle(hip, knee, ankle)
        status_color = (0, 255, 0) # Green (Default)

        # คำนวณคุณสมบัติเชิงเรขาคณิตเพิ่มเติมเพื่อประเมินความผิดพลาด
        d_ankles = abs(landmarks[27][0] - landmarks[28][0])
        d_shoulders = abs(landmarks[11][0] - landmarks[12][0])
        ratio = d_ankles / d_shoulders if d_shoulders > 0 else 1.0
        back_angle = calculate_angle(landmarks[11], landmarks[23], landmarks[25])

        # ตรวจจับมุมมองด้านข้าง (Side/Profile View)
        # ความกว้างไหล่เทียบกับความสูงของลำตัว
        torso_height = (abs(landmarks[11][1] - landmarks[23][1]) + abs(landmarks[12][1] - landmarks[24][1])) / 2
        is_side_view = (d_shoulders / torso_height < 0.35) if torso_height > 0 else False

        if angle > self.thresholds['stand']:
            if self.current_phase == 3:
                duration = time.time() - getattr(self, 'active_start_time', time.time())
                self.reps_count += 1
                if duration < 1.3:
                    self.feedback = f"ครั้งที่ {self.reps_count} สำเร็จ! (เร็วไปนิด ช้าลงหน่อย)"
                else:
                    self.feedback = f"ครั้งที่ {self.reps_count} สำเร็จ! จังหวะดีมาก"
            else:
                self.feedback = "Stand straight - Ready"
            self.current_phase = 0
        elif angle < self.thresholds['bottom']:
            self.current_phase = 2
            self.feedback = "Good depth! Keep chest up"
            status_color = (255, 255, 0) 
        elif self.thresholds['bottom'] <= angle <= self.thresholds['stand']:
            if self.current_phase == 0 or self.current_phase == 1:
                if self.current_phase == 0:
                    self.active_start_time = time.time()
                self.current_phase = 1
                self.feedback = "Going down..."
            elif self.current_phase == 2 or self.current_phase == 3:
                self.current_phase = 3
                self.feedback = "Push up!"
        
        # ค้นหาข้อผิดพลาดที่เกิดขึ้นขณะเคลื่อนไหว (Phase 1 ย่อลง หรือ Phase 2 ย่อสุด)
        if self.current_phase in [1, 2]:
            if not is_side_view and ratio < 0.8:
                self.feedback = "[!] ขาแคบไป! ขยับเท้ากว้างเท่าช่วงไหล่"
                status_color = (0, 0, 255)
            elif not is_side_view and ratio > 1.7:
                self.feedback = "[!] ขากว้างไป! ขยับเท้าชิดเข้ามาเล็กน้อย"
                status_color = (0, 0, 255)
            elif back_angle < 75:
                self.feedback = "[!] โน้มตัว/หลังก้มเกินไป! ยืดอกขึ้น หลังตรง"
                status_color = (0, 0, 255)
            elif self.current_phase == 1 and angle > 130:
                self.feedback = "[!] ย่อตัวลงต่ำอีก! ย่อก้นลงและเกร็งหน้าท้อง"
                status_color = (0, 0, 255)

        return {
            "count": self.reps_count,
            "phase": self.current_phase,
            "feedback": self.feedback,
            "status_color": status_color,
            "debug_angle": round(angle, 2)
        }
