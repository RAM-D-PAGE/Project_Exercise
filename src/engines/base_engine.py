from abc import ABC, abstractmethod
import numpy as np

class ExerciseEngine(ABC):
    """
    Abstract Base Class for all Exercise Analysis Engines.
    Follows the Strategy Pattern to allow switching between Geometry, LSTM, and Bi-LSTM.
    """

    def __init__(self, exercise_name: str):
        self.exercise_name = exercise_name
        self.reps_count = 0
        self.current_phase = 0  # Phase 0, 1, 2, 3 as defined in blueprint
        self.feedback = "Ready"
        self.performance_logs = []  # To store latency/FPS for benchmarking

    @abstractmethod
    def process(self, landmarks, frame_size):
        """
        Process a single frame or a sequence of landmarks.
        Returns a dictionary containing:
            - count: int (Total repetitions)
            - phase: int (Current movement phase 0-3)
            - feedback: str (Text advice for the user)
            - status_color: tuple (BGR color for visualization)
        """
        pass

    def reset(self):
        """Resets the counter and state."""
        self.reps_count = 0
        self.current_phase = 0
        self.feedback = "Ready"
        self.performance_logs = []
