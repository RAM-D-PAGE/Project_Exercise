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
    3: 'Jumping_Jack'
}

# Exercise classes ที่ต้องนับ Rep (ไม่รวม Idle และ Incorrect)
REP_COUNTING_CLASSES = {1, 3}  # Squat_Correct, Jumping_Jack


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
                "confidence": 0.0,
                "debug_angle": 0.0
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
                "confidence": 0.0,
                "debug_angle": 0.0
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
        self._update_rep_counter(predicted_class_id, confidence, landmarks)

        # กำหนดสีตามสถานะ
        if predicted_class_id == 0:
            status_color = (200, 200, 200)  # Gray (Idle)
        elif predicted_class_id in REP_COUNTING_CLASSES:
            if getattr(self, 'is_geom_incorrect', False):
                status_color = (0, 0, 255)  # Red (Incorrect form detected by geometry)
            else:
                status_color = (0, 255, 0)  # Green (Correct form)
        else:
            status_color = (0, 0, 255)  # Red (Incorrect form classified by Bi-LSTM)


        # คำนวณมุมสำหรับการรายงานสถิติ
        lm_list = [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]
        angle = 0.0
        try:
            if "Squat" in self.exercise_name:
                angle = calculate_angle(
                    [lm_list[23][0], lm_list[23][1]],
                    [lm_list[25][0], lm_list[25][1]],
                    [lm_list[27][0], lm_list[27][1]]
                )
            elif "Jumping" in self.exercise_name or "Jack" in self.exercise_name:
                angle = calculate_angle(
                    [lm_list[24][0], lm_list[24][1]],
                    [lm_list[12][0], lm_list[12][1]],
                    [lm_list[16][0], lm_list[16][1]]
                )
            elif "Pushup" in self.exercise_name:
                angle = calculate_angle(
                    [lm_list[12][0], lm_list[12][1]],
                    [lm_list[14][0], lm_list[14][1]],
                    [lm_list[16][0], lm_list[16][1]]
                )
        except Exception:
            pass

        return {
            "count": self.reps_count,
            "phase": self.current_phase,
            "feedback": self.feedback,
            "status_color": status_color,
            "predicted_class": predicted_class_name,
            "confidence": round(confidence, 3),
            "debug_angle": round(angle, 2)
        }

    def _update_rep_counter(self, predicted_class_id, confidence, landmarks):
        """
        State Machine สำหรับนับ Repetitions จากผล Classification
        Logic: Idle → Active (เริ่มท่า) → Idle (จบท่า = 1 Rep)
        ใช้ Confidence threshold เพื่อลด False Positive
        พร้อมเพิ่มระบบวิเคราะห์ฟีดแบ็กแนะนำการปรับปรุงท่าทางเชิงเรขาคณิตแบบสดๆ
        """
        import time
        CONFIDENCE_THRESHOLD = 0.6
        lm_list = [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]
        self.is_geom_incorrect = False

        if confidence < CONFIDENCE_THRESHOLD:
            self.feedback = f"Low confidence ({confidence:.0%})"
            return

        # ตรวจจับการเปลี่ยนสถานะ
        if predicted_class_id in REP_COUNTING_CLASSES:
            # เข้าสู่ท่าออกกำลังกาย
            if not self.in_active_phase:
                self.in_active_phase = True
                self.active_start_time = time.time()
                self.current_phase = 1
                self.feedback = f"Performing: {CLASS_NAMES[predicted_class_id]}"
            else:
                self.current_phase = 2
                
                # ตรวจเช็คคุณลักษณะเชิงเรขาคณิตระหว่างทำท่าทางเพื่อส่งฟีดแบ็กแนะนำเชิงรุก (Proactive Advice)
                if predicted_class_id == 1:  # Squat_Correct
                    # เช็คระยะห่างข้อเท้าซ้าย-ขวา เทียบไหล่
                    d_ankles = abs(lm_list[27][0] - lm_list[28][0])
                    d_shoulders = abs(lm_list[11][0] - lm_list[12][0])
                    if d_shoulders > 0:
                        ratio = d_ankles / d_shoulders
                        if ratio < 0.8:
                            self.feedback = "[แนะนำ] ยืดอกขึ้น และขยับเท้ากว้างขึ้นเล็กน้อย"
                            self.is_geom_incorrect = True
                        elif ratio > 1.6:
                            self.feedback = "[แนะนำ] ยืนเท้ากว้างเกินไป ขยับเท้าชิดขึ้นหน่อย"
                            self.is_geom_incorrect = True
                        else:
                            self.feedback = "ท่าทางและตำแหน่งเท้าดีมาก ย่อตัวลงลึกไว้"
                elif predicted_class_id == 3:  # Jumping_Jack
                    # เช็คระยะแยกขาเทียบไหล่ และความสูงของมือ
                    d_ankles = abs(lm_list[27][0] - lm_list[28][0])
                    d_shoulders = abs(lm_list[11][0] - lm_list[12][0])
                    arm_angle = calculate_angle(lm_list[24], lm_list[12], lm_list[16])
                    
                    if d_shoulders > 0 and d_ankles / d_shoulders < 1.1:
                        self.feedback = "[แนะนำ] กระโดดแยกขาให้กว้างขึ้นสัมพันธ์กับจังหวะยกมือ"
                        self.is_geom_incorrect = True
                    elif arm_angle < 135:
                        self.feedback = "[แนะนำ] ยกมือชูขึ้นให้สูงขึ้นระดับเหนือศีรษะ"
                        self.is_geom_incorrect = True
                    else:
                        self.feedback = "กางแขนและแยกขาได้จังหวะสวยงาม"

        elif predicted_class_id == 0 and self.in_active_phase:
            # กลับมาที่ Idle หลังจากอยู่ในท่า → นับ 1 Rep
            duration = time.time() - getattr(self, 'active_start_time', time.time())
            self.reps_count += 1
            self.in_active_phase = False
            self.current_phase = 0
            
            # แนะนำเรื่องความเร็ว/จังหวะ
            if duration < 1.3:
                self.feedback = f"ครั้งที่ {self.reps_count} สำเร็จ! (ย่อลุกเร็วไปนิด ช้าลงหน่อย)"
            else:
                self.feedback = f"ครั้งที่ {self.reps_count} สำเร็จ! จังหวะดีมาก"

        elif predicted_class_id == 0:
            self.current_phase = 0
            self.feedback = "Idle - Ready"

        else:
            # Incorrect form detected (คลาส 2 หรือคลาสผิดฟอร์มอื่นๆ)
            # ล้างสถานะ Active ทันที ป้องกันปัญหายังทำท่าผิดแต่ดันนับ
            self.in_active_phase = False 
            self.current_phase = -1
            
            # วิเคราะห์เจาะลึกว่าผิดฟอร์มเรื่องอะไร
            if predicted_class_id == 2:  # Squat_Incorrect
                l_knee = calculate_angle([lm_list[23][0], lm_list[23][1]], [lm_list[25][0], lm_list[25][1]], [lm_list[27][0], lm_list[27][1]])
                r_knee = calculate_angle([lm_list[24][0], lm_list[24][1]], [lm_list[26][0], lm_list[26][1]], [lm_list[28][0], lm_list[28][1]])
                avg_knee = (l_knee + r_knee) / 2
                
                # เช็คระยะห่างเท้า
                d_ankles = abs(lm_list[27][0] - lm_list[28][0])
                d_shoulders = abs(lm_list[11][0] - lm_list[12][0])
                ratio = d_ankles / d_shoulders if d_shoulders > 0 else 1.0
                
                if avg_knee > 140:
                    self.feedback = "[!] ย่อตัวไม่สุด! พยายามย่อสะโพกลงให้ลึกขึ้นอีก"
                elif ratio < 0.8:
                    self.feedback = "[!] ขาแคบเกินไป! ลุกขึ้นยืนแล้วขยับเท้ากว้างขึ้น"
                elif ratio > 1.7:
                    self.feedback = "[!] ขากว้างเกินไป! ขยับเท้าชิดเข้ามาเล็กน้อย"
                else:
                    self.feedback = "[!] ท่าทางผิดฟอร์ม! ยืดอก หลังตรง ไม่ก้มหน้า"
            else:
                self.feedback = f"[!] ท่าทางผิดปกติ: {CLASS_NAMES[predicted_class_id]}"

        self.prev_class = predicted_class_id

    def reset_active_state(self):
        """รีเซ็ตเฉพาะสถานะทำท่าทางชั่วคราวและล้างบัฟเฟอร์เฟรม (ไม่รีเซ็ต reps_count) เพื่อกันเดินเข้ากล้องแล้วนับ 1"""
        self.frame_buffer.clear()
        self.in_active_phase = False
        self.prev_class = 0
        self.current_phase = 0

    def reset(self):
        """Override reset to also clear the frame buffer and state."""
        super().reset()
        self.frame_buffer.clear()
        self.prev_class = 0
        self.in_active_phase = False
        self.current_phase = 0
