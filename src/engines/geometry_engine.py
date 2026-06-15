import numpy as np
from src.engines.base_engine import ExerciseEngine
from src.utils.geometry import calculate_angle

class GeometryEngine(ExerciseEngine):
    """
    Mode A: Mathematical Geometry Baseline.
    Uses joint angles and thresholds to detect movement phases and count repetitions.
    """

    def __init__(self, exercise_name: str):
        super().__init__(exercise_name)
        # Thresholds for Squats (Example)
        self.thresholds = {
            'SQUAT': {
                'descending': 160,
                'bottom': 100,
                'ascending': 160
            }
        }
        self.prev_phase = 0

    def process(self, landmarks, frame_size):
        """
        Process landmarks using simple geometry.
        For Squats: Monitor Knee Angle.
        """
        if self.exercise_name.upper() == "SQUAT":
            return self._process_squat(landmarks)
        
        # Default fallback
        return {
            "count": self.reps_count,
            "phase": self.current_phase,
            "feedback": "Mode A: Only Squat implemented for now",
            "status_color": (255, 255, 255)
        }

    def _process_squat(self, landmarks):
        # 1. Get Keypoints (Left Leg for example)
        hip = landmarks[23]
        knee = landmarks[25]
        ankle = landmarks[27]

        # 2. Calculate Angle
        angle = calculate_angle(hip, knee, ankle)

        # 3. State Machine Logic
        # Phase 0: Stand (Angle ~ 180)
        # Phase 1: Descending (Angle decreasing)
        # Phase 2: Bottom (Angle < 100)
        # Phase 3: Ascending (Angle increasing)

        status_color = (0, 255, 0) # Green (Default)

        if angle > 160:
            if self.current_phase == 3:
                self.reps_count += 1
                self.feedback = f"Rep {self.reps_count} Complete!"
            self.current_phase = 0
            self.feedback = "Stand straight"
        elif angle < 100:
            self.current_phase = 2
            self.feedback = "Good depth!"
            status_color = (255, 255, 0) # Cyan/Blue-ish
        elif 100 <= angle <= 160:
            if self.current_phase == 0 or self.current_phase == 1:
                self.current_phase = 1
                self.feedback = "Going down..."
            elif self.current_phase == 2 or self.current_phase == 3:
                self.current_phase = 3
                self.feedback = "Push up!"
        
        # Skeleton Color logic based on correctness (Simple)
        if self.current_phase == 1 and angle > 130:
            status_color = (0, 0, 255) # Red (Not low enough yet)

        return {
            "count": self.reps_count,
            "phase": self.current_phase,
            "feedback": self.feedback,
            "status_color": status_color,
            "debug_angle": round(angle, 2)
        }
