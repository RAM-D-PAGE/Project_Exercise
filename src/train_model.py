import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical

# --- Configuration ---
DATA_DIR = "data/landmarks"
MODEL_DIR = "models"
TIME_STEPS = 15  # Window size (15 frames = ~0.75 seconds at 20 FPS)
# อ้างอิง Classes ตาม collect_data.py
NUM_CLASSES = 6  

def load_and_preprocess_data():
    """Reads all CSVs and creates Time-Series Sequences."""
    X = []
    y = []
    
    # ค้นหาไฟล์ CSV ทั้งหมดในโฟลเดอร์ย่อย
    csv_files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
    if not csv_files:
        print("Error: No CSV files found. Please run collect_data.py first.")
        return None, None

    for file in csv_files:
        df = pd.read_csv(file)
        
        # คอลัมน์ที่ใช้เป็น Features (ข้าม class, timestamp, frame_idx)
        # ตั้งแต่ x0 ไปจนถึง r_knee_angle
        feature_cols = df.columns[3:] 
        
        # แปลงเป็น Numpy array
        features = df[feature_cols].values
        labels = df['class'].values
        
        # สร้าง Sequence ด้วยวิธี Sliding Window
        for i in range(len(features) - TIME_STEPS):
            window = features[i:(i + TIME_STEPS)]
            label = labels[i + TIME_STEPS - 1] # ใช้ Label ของเฟรมสุดท้ายใน Window
            
            X.append(window)
            y.append(int(label))
            
    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=NUM_CLASSES)
    
    return X, y

def build_bilstm_model(input_shape):
    """Builds the Bi-Directional LSTM Architecture."""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, activation='tanh'), input_shape=input_shape),
        Dropout(0.2), # ลดอาการ Overfitting ทำให้เสถียรขึ้น
        Bidirectional(LSTM(32, activation='tanh')),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("1. Loading and Preprocessing Data...")
    X, y = load_and_preprocess_data()
    if X is None: return
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # แบ่งข้อมูล Train 80% / Test 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("2. Building Bi-LSTM Model...")
    model = build_bilstm_model((TIME_STEPS, X.shape[2]))
    model.summary()
    
    print("3. Training Model...")
    # ใช้ EarlyStopping เพื่อหยุดเทรนอัตโนมัติถ้าโมเดลไม่ดีขึ้น (ป้องกัน Overfitting)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[callback]
    )
    
    # ประเมินผล
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    print("4. Exporting Models...")
    # บันทึกโมเดล Keras (.h5)
    h5_path = os.path.join(MODEL_DIR, "exercise_bilstm.h5")
    model.save(h5_path)
    
    # แปลงเป็น TFLite เพื่อให้รัน Real-time ได้เร็วสุดๆ บน CPU
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # เปิดใช้งาน Quantization เพื่อลดขนาดโมเดล
    converter.optimizations = [tf.lite.Optimize.DEFAULT] 
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(MODEL_DIR, "exercise_bilstm.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Success! Models saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
