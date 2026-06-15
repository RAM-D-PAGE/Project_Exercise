import numpy as np
import os
import sys
from collections import deque
from src.engines.base_engine import ExerciseEngine
from src.utils.geometry import calculate_angle

# --- TFLite Runtime Import ---
try:
    import tensorflow as tf
    TF_LITE_INTERPRETER = tf.lite.Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        TF_LITE_INTERPRETER = tflite.Interpreter
    except ImportError:
        print("ERROR: Neither TensorFlow nor tflite-runtime found.")
        print("Please install one: pip install tensorflow  OR  pip install tflite-runtime")
        sys.exit(1)

# --- Class Label Mapping (ต้องตรงกับ collect_data.py) ---
CLASS_NAMES = {
    0: 'Idle',
    1: 'Squat_Correct',
    2: 'Squat_Incorrect',
    3: 'Jumping_Jack',
    4: 'Bicep_Curl_Correct',
    5: 'Bicep_Curl_Incorrect'
}

# Exercise classes ที่ต้องนับ Rep (ไม่รวม Idle และ Incorrect)
REP_COUNTING_CLASSES = {1, 3, 4}  # Squat_Correct, Jumping_Jack, Bicep_Curl_Correct


class LSTMEngine(ExerciseEngine):
    """
    Mode C: Bi-LSTM Deep Learning Engine.
    Uses a trained TFLite model to classify exercise poses from a sliding window
    of landmark sequences, providing real-time prediction with temporal context.
    """

    # Number of features per frame: 33 landmarks × 4 (x,y,z,vis) + 2 angles = 134
    NUM_FEATURES = 134

    def __init__(self, exercise_name: str, model_path: str = "models/exercise_bilstm.tflite", time_steps: int = 15):
        super().__init__(exercise_name)
        self.time_steps = time_steps
        self.model_path = model_path

        # Sliding window buffer เก็บ Features ย้อนหลัง
        self.frame_buffer = deque(maxlen=time_steps)

        # สถานะสำหรับ State Machine นับ Rep
        self.prev_class = 0  # เริ่มต้นเป็น Idle
        self.in_active_phase = False  # True เมื่ออยู่ในท่าออกกำลังกาย

        # โหลด TFLite Model
        if not os.path.exists(model_path):
            print(f"WARNING: Model file not found at '{model_path}'.")
            print("Please run train_model.py first to generate the model.")
            self.interpreter = None
            return

        self.interpreter = TF_LITE_INTERPRETER(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print(f"[LSTMEngine] Model loaded: {model_path}")
        print(f"[LSTMEngine] Input shape: {self.input_details[0]['shape']}")
        print(f"[LSTMEngine] Output shape: {self.output_details[0]['shape']}")

    def _extract_features(self, landmarks):
        """
        แปลง MediaPipe landmarks เป็น Feature vector ขนาด 134
        ให้ตรงกับรูปแบบที่ train_model.py ใช้เทรน
        """
        lm_list = [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]

        features = []
        for lm in lm_list:
            features.extend(lm)  # x, y, z, visibility × 33 = 132

        # Engineered Features: Knee angles (ตรงกับ collect_data.py)
        l_knee_angle = calculate_angle(
            [lm_list[23][0], lm_list[23][1]],
            [lm_list[25][0], lm_list[25][1]],
            [lm_list[27][0], lm_list[27][1]]
        )
        r_knee_angle = calculate_angle(
            [lm_list[24][0], lm_list[24][1]],
            [lm_list[26][0], lm_list[26][1]],
            [lm_list[28][0], lm_list[28][1]]
        )
        features.extend([l_knee_angle, r_knee_angle])

        return np.array(features, dtype=np.float32)

    def process(self, landmarks, frame_size):
        """
        Process landmarks using the Bi-LSTM model.
        Maintains a sliding window and runs inference when the buffer is full.
        """
        # ถ้าโมเดลยังไม่ได้โหลด
        if self.interpreter is None:
            return {
                "count": self.reps_count,
                "phase": self.current_phase,
                "feedback": "Model not loaded! Run train_model.py first.",
                "status_color": (0, 0, 255),
                "predicted_class": "N/A",
                "confidence": 0.0
            }

        # 1. แปลง Landmarks เป็น Feature vector
        features = self._extract_features(landmarks)
        self.frame_buffer.append(features)

        # 2. ถ้า Buffer ยังไม่เต็ม ให้รอเก็บข้อมูลเพิ่ม
        if len(self.frame_buffer) < self.time_steps:
            return {
                "count": self.reps_count,
                "phase": self.current_phase,
                "feedback": f"Buffering... ({len(self.frame_buffer)}/{self.time_steps})",
                "status_color": (255, 165, 0),
                "predicted_class": "Buffering",
                "confidence": 0.0
            }

        # 3. สร้าง Input Tensor จาก Sliding Window
        input_data = np.array(list(self.frame_buffer), dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)  # Shape: (1, TIME_STEPS, NUM_FEATURES)

        # 4. Run Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        # 5. ตีความผลลัพธ์
        predicted_class_id = np.argmax(output_data[0])
        confidence = float(np.max(output_data[0]))
        predicted_class_name = CLASS_NAMES.get(predicted_class_id, "Unknown")

        # 6. State Machine สำหรับนับ Rep
        status_color = (0, 255, 0)  # Green (default)
        self._update_rep_counter(predicted_class_id, confidence)

        # กำหนดสีตามสถานะ
        if predicted_class_id == 0:
            status_color = (200, 200, 200)  # Gray (Idle)
        elif predicted_class_id in REP_COUNTING_CLASSES:
            status_color = (0, 255, 0)  # Green (Correct form)
        else:
            status_color = (0, 0, 255)  # Red (Incorrect form)

        return {
            "count": self.reps_count,
            "phase": self.current_phase,
            "feedback": self.feedback,
            "status_color": status_color,
            "predicted_class": predicted_class_name,
            "confidence": round(confidence, 3)
        }

    def _update_rep_counter(self, predicted_class_id, confidence):
        """
        State Machine สำหรับนับ Repetitions จากผล Classification
        Logic: Idle → Active (เริ่มท่า) → Idle (จบท่า = 1 Rep)
        ใช้ Confidence threshold เพื่อลด False Positive
        """
        CONFIDENCE_THRESHOLD = 0.6

        if confidence < CONFIDENCE_THRESHOLD:
            self.feedback = f"Low confidence ({confidence:.0%})"
            return

        # ตรวจจับการเปลี่ยนสถานะ
        if predicted_class_id in REP_COUNTING_CLASSES:
            # เข้าสู่ท่าออกกำลังกาย
            if not self.in_active_phase:
                self.in_active_phase = True
                self.current_phase = 1
                self.feedback = f"Performing: {CLASS_NAMES[predicted_class_id]}"
            else:
                self.current_phase = 2
                self.feedback = f"Active: {CLASS_NAMES[predicted_class_id]} ({confidence:.0%})"

        elif predicted_class_id == 0 and self.in_active_phase:
            # กลับมาที่ Idle หลังจากอยู่ในท่า → นับ 1 Rep
            self.reps_count += 1
            self.in_active_phase = False
            self.current_phase = 0
            self.feedback = f"Rep {self.reps_count} Complete!"

        elif predicted_class_id == 0:
            self.current_phase = 0
            self.feedback = "Idle - Ready"

        else:
            # Incorrect form detected
            self.feedback = f"⚠ Form check: {CLASS_NAMES[predicted_class_id]}"
            self.current_phase = -1

        self.prev_class = predicted_class_id

    def reset(self):
        """Override reset to also clear the frame buffer."""
        super().reset()
        self.frame_buffer.clear()
        self.prev_class = 0
        self.in_active_phase = False
