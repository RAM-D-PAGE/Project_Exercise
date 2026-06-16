# -*- coding: utf-8 -*-
import sys
import os
import numpy as np

# --- Add Project Root to Path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.analytics import WorkoutAnalytics
from src.utils.speech_layer import toggle_speech_mute, is_speech_muted
from src.engines.lstm_engine import LSTMEngine

def test_speech_mute():
    print("\n--- Testing Speech Mute ---")
    initial_mute = is_speech_muted()
    print(f"Initial Speech Mute: {initial_mute}")
    
    toggled_mute = toggle_speech_mute()
    print(f"Toggled Speech Mute: {toggled_mute}")
    assert toggled_mute != initial_mute, "Speech mute status did not change!"
    
    # Restore
    toggle_speech_mute()
    print("Speech mute restored to initial.")

def test_analytics_csv_export():
    print("\n--- Testing WorkoutAnalytics CSV Export ---")
    analytics = WorkoutAnalytics("Squat_Correct", fps=30.0)
    
    # Log some frames
    analytics.log_frame(frame_idx=1, angle=170.0, rep_count=0, phase=0, status_color=(255, 255, 255), confidence=0.9, predicted_class='Idle')
    analytics.log_frame(frame_idx=2, angle=150.0, rep_count=0, phase=1, status_color=(0, 255, 0), confidence=0.95, predicted_class='Squat_Correct')
    analytics.log_frame(frame_idx=3, angle=95.0, rep_count=0, phase=2, status_color=(0, 255, 0), confidence=0.97, predicted_class='Squat_Correct')
    analytics.log_frame(frame_idx=4, angle=165.0, rep_count=1, phase=3, status_color=(0, 255, 0), confidence=0.96, predicted_class='Squat_Correct')
    
    # Log an incorrect posture frame
    analytics.log_frame(frame_idx=5, angle=120.0, rep_count=1, phase=1, status_color=(0, 0, 255), confidence=0.85, predicted_class='Squat_Incorrect')

    csv_path = analytics.export_to_csv(output_dir="results")
    assert csv_path is not None, "CSV export returned None!"
    assert os.path.exists(csv_path), f"CSV file does not exist at {csv_path}"
    print(f"✓ CSV successfully generated at: {csv_path}")
    
    # Verify CSV contents
    import csv
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"Parsed CSV rows: {len(rows)}")
        assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"
        assert rows[0]['frame_idx'] == '1', f"Expected frame_idx 1, got {rows[0]['frame_idx']}"
        assert rows[0]['confidence'] == '0.9', f"Expected confidence 0.9, got {rows[0]['confidence']}"
        assert rows[0]['predicted_class'] == 'Idle', f"Expected predicted_class Idle, got {rows[0]['predicted_class']}"
        assert rows[4]['state'] == 'Incorrect', f"Expected state Incorrect, got {rows[4]['state']}"
    print("✓ CSV verification passed.")

class DummyLandmark:
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

def test_lstm_vectorization():
    print("\n--- Testing LSTM Feature Extraction Vectorization ---")
    
    # Instantiate engine (we pass a dummy/non-existent model since we only test _extract_features)
    engine = LSTMEngine("Squat_Correct", model_path="models/exercise_bilstm.tflite")
    
    # Create 33 dummy landmarks
    landmarks = [DummyLandmark(0.1 * i, 0.2 * i, 0.3 * i, 0.9) for i in range(33)]
    
    features = engine._extract_features(landmarks)
    print(f"Extracted features shape: {features.shape}")
    assert features.shape == (134,), f"Expected feature shape (134,), got {features.shape}"
    assert isinstance(features, np.ndarray), "Extracted features is not a numpy array"
    assert features.dtype == np.float32, f"Expected features dtype float32, got {features.dtype}"
    print("✓ LSTM vectorization test passed.")

if __name__ == "__main__":
    try:
        test_speech_mute()
        test_analytics_csv_export()
        test_lstm_vectorization()
        print("\n🎉 All tests passed successfully!")
    except AssertionError as e:
        print(f"\n❌ Test assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test crashed: {e}")
        sys.exit(1)
