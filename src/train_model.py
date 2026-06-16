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
NUM_CLASSES = 4  

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

def build_bilstm_model(input_shape, batch_size=None):
    """Builds the Bi-Directional LSTM Architecture."""
    kwargs = {}
    if batch_size is not None:
        kwargs['batch_input_shape'] = (batch_size, input_shape[0], input_shape[1])
    else:
        kwargs['input_shape'] = input_shape

    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, activation='tanh'), **kwargs),
        Dropout(0.2), # ลดอาการ Overfitting ทำให้เสถียรขึ้น
        Bidirectional(LSTM(32, activation='tanh')),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

class RealTimePlotCallback(tf.keras.callbacks.Callback):
    """Callback สำหรับอัปเดตกราฟผลการเทรนแบบ Realtime และบันทึกลงไฟล์ในทุกๆ Epoch"""
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.epochs = []
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
        
        import matplotlib.pyplot as plt
        self.plt = plt
        
        # ตั้งค่า interactive mode สำหรับ matplotlib
        self.plt.ion()
        self.fig, (self.ax1, self.ax2) = self.plt.subplots(1, 2, figsize=(12, 5))
        self.plt.tight_layout()
        
        # แสดงหน้าต่างทันทีตั้งแต่ก่อนเริ่มกระบวนการเทรน (ไม่ใช่รอดูตอนจบอย่างเดียว)
        self.has_gui = True
        try:
            self.plt.show(block=False)
            self.plt.pause(0.1)
        except Exception as e:
            self.has_gui = False
            print(f"💡 (ระบบใช้การอัปเดตไฟล์กราฟเป็นแบบ Realtime บนดิสก์แทน เนื่องจากตรวจไม่พบสภาพแวดล้อมหน้าจอ GUI: {e})")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs.append(epoch + 1)
        self.acc.append(logs.get('accuracy', 0))
        self.val_acc.append(logs.get('val_accuracy', 0))
        self.loss.append(logs.get('loss', 0))
        self.val_loss.append(logs.get('val_loss', 0))

        try:
            self.ax1.clear()
            self.ax2.clear()

            # 1. วาด Accuracy
            self.ax1.plot(self.epochs, self.acc, 'b-o', label='Train Acc')
            self.ax1.plot(self.epochs, self.val_acc, 'r-x', label='Val Acc')
            self.ax1.set_title(f"Accuracy (Epoch {epoch+1})")
            self.ax1.set_xlabel('Epochs')
            self.ax1.set_ylabel('Accuracy')
            self.ax1.grid(True, linestyle='--', alpha=0.6)
            self.ax1.legend()

            # 2. วาด Loss
            self.ax2.plot(self.epochs, self.loss, 'b-o', label='Train Loss')
            self.ax2.plot(self.epochs, self.val_loss, 'r-x', label='Val Loss')
            self.ax2.set_title(f"Loss (Epoch {epoch+1})")
            self.ax2.set_xlabel('Epochs')
            self.ax2.set_ylabel('Loss')
            self.ax2.grid(True, linestyle='--', alpha=0.6)
            self.ax2.legend()

            self.plt.tight_layout()

            # เซฟลงดิสก์แบบ Realtime
            self.fig.savefig(self.save_path, dpi=120)

            # อัปเดตแสดงผลบนหน้าจอ GUI ทันทีในแต่ละ Epoch
            if self.has_gui:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                self.plt.pause(0.1)  # ให้เวลาวาดเสร็จสมบูรณ์
        except Exception:
            self.has_gui = False

    def on_train_end(self, logs=None):
        try:
            self.plt.ioff()
            # ปล่อยให้หน้าต่างสรุปค้างแสดงผลอยู่บนหน้าจอเพื่อให้ผู้ใช้เปิดดู/วิเคราะห์ได้
        except Exception:
            pass


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
    history_img_path = os.path.join(MODEL_DIR, "training_history.png")
    
    # กำหนด Callbacks สำหรับเทรน
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    plot_callback = RealTimePlotCallback(history_img_path)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, plot_callback]
    )
    
    # ประเมินผล
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\n  🎯 Test Loss: {loss:.4f}  |  Test Accuracy: {accuracy*100:.2f}%")
    
    # เปิดรูปภาพสรุปเมื่อเทรนเสร็จสมบูรณ์
    try:
        if os.name == 'nt':
            os.startfile(os.path.abspath(history_img_path))
            print("  🖼️  เปิดรูปภาพสรุปผลกราฟการเทรนอัตโนมัติ...")
    except Exception:
        pass

    # พิมพ์คำแนะนำการอ่านกราฟ
    print("\n" + "=" * 60)
    print("  💡 วิธีการวิเคราะห์ความเสถียรของโมเดลจากกราฟ:")
    print("  1. โมเดลที่ดี (Good Fit):")
    print("     - Loss ทั้งคู่ลดลงและอยู่ระดับต่ำใกล้เคียงกัน (ลู่เข้าหากัน)")
    print("     - Accuracy ทั้งคู่เพิ่มขึ้นและอยู่ระดับสูงใกล้เคียงกัน (ลู่เข้าหากัน)")
    print("  2. โมเดลเกิดการ Overfitting (เรียนรู้จำเจ):")
    print("     - Training Loss ลดฮวบๆ แต่ Validation Loss วิ่งขึ้นหรือส่ายหัวปัก")
    print("     - Training Acc สูงปรี๊ด แต่ Validation Acc ตกหรือย่ำอยู่กับที่")
    print("  3. โมเดลเกิดการ Underfitting (ยังเทรนไม่พอ/ข้อมูลน้อย):")
    print("     - Loss ทั้งคู่ยังสูงมาก หรือมีแนวโน้มดิ่งลงแต่โปรแกรมดันหยุดเสียก่อน")
    print("=" * 60 + "\n")
    
    print("4. Exporting Models...")
    # บันทึกโมเดล Keras (.h5)
    h5_path = os.path.join(MODEL_DIR, "exercise_bilstm.h5")
    model.save(h5_path)
    
    # แปลงเป็น TFLite เพื่อให้รัน Real-time ได้เร็วสุดๆ บน CPU
    # เพื่อหลีกเลี่ยงข้อผิดพลาด 'tf.TensorListReserve' ของ LSTM
    # เราจะสร้างโมเดลตัวใหม่ที่มีอินพุตแบบ Static (batch_size=1) แล้วโหลดน้ำหนักมาใช้แปลง
    try:
        print("   Preparing static model for TFLite conversion (batch_size=1)...")
        num_features = X.shape[2]
        static_model = build_bilstm_model((TIME_STEPS, num_features), batch_size=1)
        static_model.set_weights(model.get_weights())
        
        converter = tf.lite.TFLiteConverter.from_keras_model(static_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT] 
        tflite_model = converter.convert()
        print("   ✅ Standard TFLite conversion successful!")
    except Exception as e:
        print(f"   ⚠️ Standard TFLite conversion failed: {e}")
        print("   Attempting Select TF Ops fallback...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT] 
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        print("   ✅ Select TF Ops TFLite conversion successful!")
    
    tflite_path = os.path.join(MODEL_DIR, "exercise_bilstm.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Success! Models saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
