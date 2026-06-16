# -*- coding: utf-8 -*-
from src.engines.base_engine import ExerciseEngine
from src.exercises.squat_processor import SquatProcessor
from src.exercises.pushup_processor import PushupProcessor
from src.exercises.jumping_jack_processor import JumpingJackProcessor

class GeometryEngine(ExerciseEngine):
    """
    Mode A: Mathematical Geometry Baseline.
    Delegates to specific exercise processor classes based on SOLID design patterns.
    Supports: Squat, Push-up, Jumping Jack
    """
    def __init__(self, exercise_name: str):
        super().__init__(exercise_name)
        self.processor = self._create_processor(exercise_name)

    def _create_processor(self, name: str):
        name_upper = name.upper()
        if "SQUAT" in name_upper:
            return SquatProcessor()
        elif "PUSHUP" in name_upper or "PUSH_UP" in name_upper:
            return PushupProcessor()
        elif "JUMPING" in name_upper or "JACK" in name_upper:
            return JumpingJackProcessor()
        else:
            raise ValueError(f"Unsupported exercise: {name}")

    def process(self, landmarks, frame_size):
        """Delegates landmarks processing to the dedicated exercise processor."""
        if self.processor is None:
            return {
                "count": self.reps_count,
                "phase": self.current_phase,
                "feedback": f"Mode A: {self.exercise_name} not supported.",
                "status_color": (255, 255, 255)
            }

        res = self.processor.process(landmarks, frame_size)
        
        # Keep engine state in sync with processor state
        self.reps_count = self.processor.reps_count
        self.current_phase = self.processor.current_phase
        self.feedback = self.processor.feedback

        return res

    def reset(self):
        """Resets the engine and the underlying processor state."""
        super().reset()
        if self.processor:
            self.processor.reset()
