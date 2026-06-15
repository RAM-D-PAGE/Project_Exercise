import numpy as np
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

    def process(self, landmarks, frame_size):
        """
        Process landmarks using simple geometry based on the selected exercise.
        """
        ex_name = self.exercise_name.upper()
        
        if ex_name == "SQUAT" or ex_name == "SQUAT_CORRECT":
            return self._process_squat(landmarks)
        elif ex_name == "PUSHUP" or ex_name == "PUSH_UPS":
            return self._process_pushup(landmarks)
        elif ex_name == "JUMPING_JACK":
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

        # 3. State Machine Logic
        if angle > self.thresholds['SQUAT']['stand']:
            if self.current_phase == 3:
                self.reps_count += 1
                self.feedback = f"Rep {self.reps_count} Complete!"
            self.current_phase = 0
            self.feedback = "Stand straight"
        elif angle < self.thresholds['SQUAT']['bottom']:
            self.current_phase = 2
            self.feedback = "Good depth!"
            status_color = (255, 255, 0) # Cyan/Blue-ish
        elif self.thresholds['SQUAT']['bottom'] <= angle <= self.thresholds['SQUAT']['stand']:
            if self.current_phase == 0 or self.current_phase == 1:
                self.current_phase = 1
                self.feedback = "Going down..."
            elif self.current_phase == 2 or self.current_phase == 3:
                self.current_phase = 3
                self.feedback = "Push up!"
        
        if self.current_phase == 1 and angle > 130:
            status_color = (0, 0, 255) # Red (Not low enough yet)

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

        # 3. State Machine Logic
        if angle > self.thresholds['PUSHUP']['plank']:
            if self.current_phase == 3:
                self.reps_count += 1
                self.feedback = f"Rep {self.reps_count} Complete!"
            self.current_phase = 0
            self.feedback = "High Plank"
        elif angle < self.thresholds['PUSHUP']['bottom']:
            self.current_phase = 2
            self.feedback = "Good depth!"
            status_color = (255, 255, 0) 
        elif self.thresholds['PUSHUP']['bottom'] <= angle <= self.thresholds['PUSHUP']['plank']:
            if self.current_phase == 0 or self.current_phase == 1:
                self.current_phase = 1
                self.feedback = "Going down..."
            elif self.current_phase == 2 or self.current_phase == 3:
                self.current_phase = 3
                self.feedback = "Push up!"

        # Warning if not deep enough during descent
        if self.current_phase == 1 and angle > 120:
            status_color = (0, 0, 255) # Red

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

        # 3. State Machine Logic
        if angle < self.thresholds['JUMPING_JACK']['down']:
            if self.current_phase == 3:
                self.reps_count += 1
                self.feedback = f"Rep {self.reps_count} Complete!"
            self.current_phase = 0
            self.feedback = "Stand straight"
        elif angle > self.thresholds['JUMPING_JACK']['up']:
            self.current_phase = 2
            self.feedback = "Hands above head!"
            status_color = (255, 255, 0) 
        elif self.thresholds['JUMPING_JACK']['down'] <= angle <= self.thresholds['JUMPING_JACK']['up']:
            if self.current_phase == 0 or self.current_phase == 1:
                self.current_phase = 1
                self.feedback = "Jump out!"
            elif self.current_phase == 2 or self.current_phase == 3:
                self.current_phase = 3
                self.feedback = "Jump in!"

        return {
            "count": self.reps_count,
            "phase": self.current_phase,
            "feedback": self.feedback,
            "status_color": status_color,
            "debug_angle": round(angle, 2)
        }
