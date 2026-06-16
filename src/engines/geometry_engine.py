import numpy as np
import time
from src.engines.base_engine import ExerciseEngine
from src.utils.geometry import calculate_angle

class GeometryEngine(ExerciseEngine):
    """
    Mode A: Mathematical Geometry Baseline.
    Uses joint angles and thresholds to detect movement phases and count repetitions.
    Supports: Squat, Push-up, Jumping Jack
    """

    def __init__(self, exercise_name: str):
        super().__init__(exercise_name)
        # Thresholds Configuration
        self.thresholds = {
            'SQUAT': {'stand': 160, 'bottom': 100},
            'PUSHUP': {'plank': 160, 'bottom': 90},
            'JUMPING_JACK': {'down': 45, 'up': 150}
        }
        self.active_start_time = 0.0

    def process(self, landmarks, frame_size):
        """
        Process landmarks using simple geometry based on the selected exercise.
        """
        ex_name = self.exercise_name.upper()
        
        if "SQUAT" in ex_name:
            return self._process_squat(landmarks)
        elif "PUSHUP" in ex_name or "PUSH_UP" in ex_name:
            return self._process_pushup(landmarks)
        elif "JUMPING" in ex_name or "JACK" in ex_name:
            return self._process_jumping_jack(landmarks)
        
        # Default fallback if exercise is unknown
        return {
            "count": self.reps_count,
            "phase": self.current_phase,
            "feedback": f"Mode A: {self.exercise_name} not fully supported yet.",
            "status_color": (255, 255, 255)
        }

    def _process_squat(self, landmarks):
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

        if angle > self.thresholds['SQUAT']['stand']:
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
        elif angle < self.thresholds['SQUAT']['bottom']:
            self.current_phase = 2
            self.feedback = "Good depth! Keep chest up"
            status_color = (255, 255, 0) 
        elif self.thresholds['SQUAT']['bottom'] <= angle <= self.thresholds['SQUAT']['stand']:
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

    def _process_pushup(self, landmarks):
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
        if angle > self.thresholds['PUSHUP']['plank']:
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
        elif angle < self.thresholds['PUSHUP']['bottom']:
            self.current_phase = 2
            self.feedback = "Good depth! Keep body straight"
            status_color = (255, 255, 0) 
        elif self.thresholds['PUSHUP']['bottom'] <= angle <= self.thresholds['PUSHUP']['plank']:
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

    def _process_jumping_jack(self, landmarks):
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
        if angle < self.thresholds['JUMPING_JACK']['down']:
            if self.current_phase == 3:
                self.reps_count += 1
                self.feedback = f"ครั้งที่ {self.reps_count} สำเร็จ!"
            else:
                self.feedback = "Stand straight - Ready"
            self.current_phase = 0
        elif angle > self.thresholds['JUMPING_JACK']['up']:
            self.current_phase = 2
            if not is_side_view and leg_ratio < 1.2:
                self.feedback = "[!] ยกมือขึ้นสุดแล้ว แต่ยังไม่ได้แยกขา!"
                status_color = (0, 0, 255)
            else:
                self.feedback = "Hands above head!" + ("" if is_side_view else " Leg opened")
                status_color = (255, 255, 0) 
        elif self.thresholds['JUMPING_JACK']['down'] <= angle <= self.thresholds['JUMPING_JACK']['up']:
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
