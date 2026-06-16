# -*- coding: utf-8 -*-

class BaseExerciseProcessor:
    """
    Base class for all exercise processors in Mode A (Geometry).
    Encapsulates exercise state and defines the abstract process method.
    """
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.reps_count = 0
        self.current_phase = 0
        self.active_start_time = 0.0
        self.feedback = "Ready"

    def process(self, landmarks, frame_size):
        raise NotImplementedError

    def reset(self):
        self.reps_count = 0
        self.current_phase = 0
        self.active_start_time = 0.0
        self.feedback = "Ready"
