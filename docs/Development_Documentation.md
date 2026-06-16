# 📖 เอกสารขั้นตอนการพัฒนาระบบ AI Exercise Analysis System
# AI-Powered Exercise Analysis using MediaPipe and Bi-LSTM

> **เอกสารฉบับเต็ม** — เขียนขึ้นเพื่อให้สามารถนำไปอ้างอิงและขยายความเป็น **เอกสารรายงาน 5 บท** (บทนำ, ทฤษฎี, วิธีดำเนินงาน, ผลการทดลอง, สรุป) ได้ทุกเมื่อ

---

## สารบัญ (Table of Contents)

1. [บทที่ 1: บทนำ (Introduction)](#บทที่-1-บทนำ-introduction)
2. [บทที่ 2: ทฤษฎีและงานวิจัยที่เกี่ยวข้อง (Literature Review)](#บทที่-2-ทฤษฎีและงานวิจัยที่เกี่ยวข้อง-literature-review)
3. [บทที่ 3: วิธีดำเนินงาน (Methodology)](#บทที่-3-วิธีดำเนินงาน-methodology)
4. [บทที่ 4: ผลการทดลองและวิเคราะห์ (Results & Analysis)](#บทที่-4-ผลการทดลองและวิเคราะห์-results--analysis)
5. [บทที่ 5: สรุป อภิปราย และข้อเสนอแนะ (Conclusion)](#บทที่-5-สรุป-อภิปราย-และข้อเสนอแนะ-conclusion)
6. [ภาคผนวก (Appendix)](#ภาคผนวก-appendix)

---

# บทที่ 1: บทนำ (Introduction)

## 1.1 ความเป็นมาและความสำคัญของปัญหา

การออกกำลังกายที่ถูกต้องตามหลักสรีรศาสตร์เป็นสิ่งสำคัญต่อสุขภาพ แต่ผู้ออกกำลังกายจำนวนมากมักทำท่าทางผิดรูปแบบ ส่งผลให้เกิดการบาดเจ็บและลดประสิทธิภาพในการฝึกซ้อม ระบบผู้ช่วยฝึกสอนด้วย AI (AI Trainer) ที่สามารถตรวจจับท่าทาง วิเคราะห์ความถูกต้อง และนับจำนวนครั้ง (Repetition Counting) ได้แบบเรียลไทม์ จึงเป็นเครื่องมือที่มีศักยภาพสูงในการยกระดับการออกกำลังกายให้ปลอดภัยและมีประสิทธิภาพ

## 1.2 วัตถุประสงค์ (Objectives)

1. พัฒนาระบบตรวจจับและจำแนกท่าทางการออกกำลังกายแบบเรียลไทม์ โดยใช้เทคนิค Pose Estimation ร่วมกับ Deep Learning
2. เปรียบเทียบประสิทธิภาพระหว่าง 3 วิธีการ: Geometry-based (Baseline), Vanilla LSTM, และ Bi-directional LSTM
3. พัฒนาระบบ Feedback อัตโนมัติเพื่อช่วยปรับปรุงท่าทาง
4. ทดสอบความเป็นไปได้ในการใช้งานจริงบนเครื่องคอมพิวเตอร์ทั่วไป (CPU-based Real-time)

## 1.3 ขอบเขตการดำเนินงาน (Scope)

| หัวข้อ | รายละเอียด |
|---|---|
| ผู้ใช้งาน | รองรับ 1 คนต่อ 1 เฟรมภาพ (Single Person Tracking) |
| ท่าออกกำลังกาย | Squats, Push-ups, Jumping Jacks, Bicep Curls |
| สภาพแวดล้อม | เว็บแคมในแสงที่เหมาะสม (Indoor/Outdoor Ambient Light) |
| แพลตฟอร์ม | CPU-based (i5/Ryzen 5 ขึ้นไป), รองรับ GPU สำหรับเทรน |
| ซอฟต์แวร์ | Python 3.10+, TensorFlow, MediaPipe, OpenCV |

## 1.4 ประโยชน์ที่คาดว่าจะได้รับ

1. ผู้ใช้สามารถออกกำลังกายอย่างถูกต้องโดยไม่ต้องมีเทรนเนอร์ประจำตัว
2. ลดอัตราการบาดเจ็บจากการทำท่าทางผิดพลาด
3. เป็นต้นแบบสำหรับการต่อยอดในด้านกายภาพบำบัดและ Smart Gym
4. เป็นกรณีศึกษาเปรียบเทียบ Geometry vs LSTM vs Bi-LSTM ในงาน Time-Series Classification

---

# บทที่ 2: ทฤษฎีและงานวิจัยที่เกี่ยวข้อง (Literature Review)

## 2.1 การประมาณท่าทาง (Pose Estimation)

### 2.1.1 MediaPipe Pose

MediaPipe Pose เป็นเฟรมเวิร์กที่พัฒนาโดย Google สำหรับตรวจจับจุดสำคัญบนร่างกาย (Body Landmarks) จำนวน **33 จุด** แบบเรียลไทม์ ทำงานผ่าน 2 ขั้นตอน:

1. **Detection Stage:** ตรวจจับตำแหน่งบุคคลในเฟรมภาพ (ใช้ BlazePose Detector)
2. **Tracking Stage:** ติดตามจุดสำคัญ 33 จุด ในเฟรมถัดไปโดยไม่ต้อง Detect ใหม่

แต่ละจุด (Landmark) ให้ข้อมูล 4 ค่า:
- `x`: พิกัดแนวนอน (Normalized 0.0 - 1.0)
- `y`: พิกัดแนวตั้ง (Normalized 0.0 - 1.0)
- `z`: ความลึก (Relative to hip center)
- `visibility`: ระดับความมั่นใจที่จุดนั้นมองเห็นได้ (0.0 - 1.0)

```
จุดสำคัญที่ใช้ในโปรเจกต์นี้:
┌────────────────────────────────────────────┐
│ Index │ Landmark Name    │ ใช้ในท่า        │
│───────│──────────────────│─────────────────│
│  11   │ Left Shoulder    │ Push-up, JJ     │
│  12   │ Right Shoulder   │ Push-up, JJ     │
│  13   │ Left Elbow       │ Bicep Curl      │
│  14   │ Right Elbow      │ Push-up         │
│  15   │ Left Wrist       │ Bicep Curl      │
│  16   │ Right Wrist      │ Push-up, JJ     │
│  23   │ Left Hip         │ Squat           │
│  24   │ Right Hip        │ JJ              │
│  25   │ Left Knee        │ Squat           │
│  27   │ Left Ankle       │ Squat           │
└────────────────────────────────────────────┘
```

### 2.1.2 Model Complexity

โปรเจกต์นี้ใช้ `model_complexity=1` (Medium) ซึ่งให้ความสมดุลระหว่าง:
- **ความแม่นยำ** ดีกว่า complexity 0 (Lite)
- **ความเร็ว** เร็วกว่า complexity 2 (Full) ที่ใช้สำหรับ 3D Pose

## 2.2 การเรียนรู้ลำดับเวลา (Sequence Learning)

### 2.2.1 LSTM (Long Short-Term Memory)

LSTM เป็นสถาปัตยกรรม RNN ที่ออกแบบมาเพื่อแก้ปัญหา Vanishing Gradient ในลำดับข้อมูลยาว (Long-Range Dependencies) ประกอบด้วย Gate 3 ตัว:

1. **Forget Gate (fₜ):** ตัดสินใจว่าจะลืมข้อมูลเก่าส่วนใด
2. **Input Gate (iₜ):** ตัดสินใจว่าจะเก็บข้อมูลใหม่ส่วนใด
3. **Output Gate (oₜ):** ตัดสินใจว่าจะส่งข้อมูลใดเป็น Output

```
สมการ LSTM:
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)         → Forget Gate
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)         → Input Gate
C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)     → Candidate Cell
Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ             → Cell State
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)         → Output Gate
hₜ = oₜ * tanh(Cₜ)                    → Hidden State
```

### 2.2.2 Bi-directional LSTM

Bi-LSTM ประมวลผลลำดับข้อมูลใน **2 ทิศทาง** พร้อมกัน:
- **Forward LSTM:** ประมวลผลจากอดีต → ปัจจุบัน (t₁ → tₙ)
- **Backward LSTM:** ประมวลผลจากอนาคต → ปัจจุบัน (tₙ → t₁)

จากนั้นรวม Hidden State ของทั้ง 2 ทิศทางเข้าด้วยกัน ทำให้โมเดลมี **Look-ahead capability** — เห็นบริบทข้างหน้าได้ภายใน Window

> **ทำไมใช้ได้กับระบบ Real-time?**
> เพราะ Window Size ที่ใช้มีขนาดเล็ก (15 เฟรม ≈ 0.75 วินาที ที่ 20 FPS) ทำให้ความหน่วงแทบไม่ส่งผลต่อประสบการณ์ผู้ใช้ แต่แลกมาด้วยความแม่นยำในการจำแนกจังหวะ "ขึ้น" และ "ลง" ของท่าทางที่ดีกว่า LSTM แบบทิศทางเดียว

### 2.2.3 Sliding Window Technique

เทคนิค Sliding Window แปลงข้อมูล Time-Series ให้เป็น Sequence สำหรับ LSTM:

```
Frame:    F1  F2  F3  F4  F5  F6  F7  F8  F9  F10 ...
           ├───────────────┤
Window 1: [F1, F2, F3, F4, F5] → Label = F5
              ├───────────────┤
Window 2:    [F2, F3, F4, F5, F6] → Label = F6
                 ├───────────────┤
Window 3:       [F3, F4, F5, F6, F7] → Label = F7
```

- **Window Size = 15 เฟรม** (≈ 0.75 วินาที ที่ 20 FPS)
- **Label:** ใช้คลาสของเฟรมสุดท้ายใน Window
- ได้ Input Tensor รูป `(Batch, 15, 134)`

## 2.3 การคำนวณมุมทางเรขาคณิต (Geometric Angle Calculation)

ใช้ Dot Product เพื่อคำนวณมุมระหว่าง 3 จุด (a, b, c) โดย b เป็นจุดยอดมุม:

```
θ = arccos( (BA · BC) / (|BA| × |BC|) )

โดย:
  BA = A - B (เวกเตอร์จาก B ไป A)
  BC = C - B (เวกเตอร์จาก B ไป C)
  · = Dot Product
  | | = Magnitude (ขนาดเวกเตอร์)
```

ผลลัพธ์อยู่ในช่วง **[0°, 180°]** ซึ่งใช้เป็น Threshold ในการตัดสินท่าทาง:

| ท่า | จุดที่ใช้ | มุม (ท่าเตรียม) | มุม (จุดสูงสุด) |
|---|---|---|---|
| Squat | Hip → Knee → Ankle | > 160° (ยืนตรง) | < 100° (ย่อลึก) |
| Push-up | Shoulder → Elbow → Wrist | > 160° (Plank) | < 90° (ลงสุด) |
| Jumping Jack | Hip → Shoulder → Wrist | < 45° (แนบลำตัว) | > 150° (ยกมือ) |

## 2.4 TensorFlow Lite (TFLite)

TensorFlow Lite เป็น Runtime สำหรับรันโมเดล ML บนอุปกรณ์ Edge/Mobile โดย:
- **Quantization:** ลดขนาดโมเดลจาก Float32 → Int8 (ลดขนาด ~4 เท่า)
- **Optimized Kernels:** ใช้ XNNPACK delegate สำหรับ CPU ทำให้ Inference เร็วขึ้น
- **เหมาะกับ Real-time:** ลด Latency ให้ต่ำพอสำหรับ 20+ FPS

---

# บทที่ 3: วิธีดำเนินงาน (Methodology)

## 3.1 สถาปัตยกรรมระบบ (System Architecture)

### 3.1.1 ภาพรวมระบบ

```
┌─────────────┐     ┌───────────────┐     ┌─────────────────┐     ┌──────────┐
│   Webcam    │ ──→ │  MediaPipe    │ ──→ │  Engine Layer   │ ──→ │   UI     │
│  (Input)    │     │  Pose (33pts) │     │ (Strategy Pat.) │     │ (Output) │
└─────────────┘     └───────────────┘     └─────────────────┘     └──────────┘
                                                  │
                                    ┌─────────────┼─────────────┐
                                    ▼             ▼             ▼
                              ┌──────────┐  ┌──────────┐  ┌──────────┐
                              │ Mode A   │  │ Mode B   │  │ Mode C   │
                              │Geometry  │  │ LSTM     │  │ Bi-LSTM  │
                              │(Baseline)│  │(Vanilla) │  │(Proposed)│
                              └──────────┘  └──────────┘  └──────────┘
```

### 3.1.2 Strategy Pattern (Design Pattern)

ใช้รูปแบบ Strategy Pattern เพื่อให้สามารถสลับ Engine ได้ขณะรันโปรแกรม:

```
ExerciseEngine (Abstract Base Class)
├── process(landmarks, frame_size) → dict   [Abstract Method]
├── reset()                                 [Concrete Method]
├── reps_count: int                         [State]
├── current_phase: int                      [State]
└── feedback: str                           [State]

    ┌─── GeometryEngine   (Mode A: คำนวณมุม)
    ├─── LSTMEngine        (Mode C: TFLite Bi-LSTM)
    └─── [Future: VanillaLSTMEngine (Mode B)]
```

**ไฟล์ที่เกี่ยวข้อง:**
- `src/engines/base_engine.py` — Abstract Base Class
- `src/engines/geometry_engine.py` — Mode A
- `src/engines/lstm_engine.py` — Mode C

## 3.2 ขั้นตอนที่ 1: การเก็บข้อมูล (Data Collection)

### 3.2.1 ไฟล์: `src/collect_data.py`

โปรแกรมเก็บข้อมูลรองรับ 3 โหมดการนำเข้า:
1. **Webcam (Real-time):** กด Space เริ่ม/หยุดบันทึก
2. **Upload Video:** กด U แล้วเลือกไฟล์วิดีโอ (.mp4, .avi, .mov)
3. **Upload Image:** กด U แล้วเลือกรูปภาพ (.jpg, .png)

### 3.2.2 คลาสท่าทาง (Exercise Classes)

| Key | Class ID | ชื่อ | คำอธิบาย |
|-----|----------|------|----------|
| 0 | 0 | Idle | ท่าพัก/ไม่มีการเคลื่อนไหว |
| 1 | 1 | Squat_Correct | สควอทท่าถูกต้อง |
| 2 | 2 | Squat_Incorrect | สควอทท่าผิด |
| 3 | 3 | Jumping_Jack | กระโดดตบ |
| 4 | 4 | Bicep_Curl_Correct | ยกน้ำหนักท่าถูก |
| 5 | 5 | Bicep_Curl_Incorrect | ยกน้ำหนักท่าผิด |

### 3.2.3 โครงสร้างข้อมูลที่บันทึก (Data Schema)

ข้อมูลถูกบันทึกเป็นไฟล์ CSV ที่มีโครงสร้าง:

```
Columns (ทั้งหมด 137 คอลัมน์):
├── class          → ชื่อคลาสของท่าทาง (เช่น "Squat_Correct")
├── timestamp      → เวลาที่บันทึก (Unix timestamp)
├── frame_idx      → ลำดับเฟรมใน Session
├── x0, y0, z0, v0 → Landmark 0 (Nose): x, y, z, visibility
├── x1, y1, z1, v1 → Landmark 1 (Left Eye Inner)
│   ...
├── x32, y32, z32, v32 → Landmark 32 (Right Foot Index)
├── l_knee_angle   → มุมเข่าซ้าย (Engineered Feature)
└── r_knee_angle   → มุมเข่าขวา (Engineered Feature)
```

**จำนวน Features ที่ใช้เทรน:** 134 คอลัมน์ (ตั้งแต่ x0 ถึง r_knee_angle)

### 3.2.4 การตั้งค่าพารามิเตอร์สำหรับเก็บข้อมูล

| พารามิเตอร์ | ค่า | เหตุผล |
|---|---|---|
| FPS Target | 20 | ความสมดุลระหว่างข้อมูลที่เพียงพอและประสิทธิภาพ CPU |
| Model Complexity | 1 | Medium accuracy, ไม่หนักเกินไปสำหรับเก็บข้อมูล |
| Min Detection Conf | 0.5 | กรองเฟรมที่ตรวจจับไม่ชัด |
| Min Tracking Conf | 0.5 | ให้ Tracking ทำงานต่อเนื่อง |

### 3.2.5 โครงสร้างไฟล์ข้อมูลที่ได้

```
data/
├── landmarks/                  ← ไฟล์ CSV พิกัด
│   ├── Squat_Correct/
│   │   ├── sess_1718450000.csv
│   │   └── upload_1718451000.csv
│   ├── Squat_Incorrect/
│   ├── Jumping_Jack/
│   ├── Bicep_Curl_Correct/
│   ├── Bicep_Curl_Incorrect/
│   └── Idle/
├── raw_images/                 ← ภาพ JPG ทีละเฟรม
│   └── [class_name]/[session_id]/
└── raw_videos/                 ← วิดีโอ MP4 ทั้ง Session
    └── [class_name]/
```

### 3.2.6 คำแนะนำในการเก็บข้อมูล

- เก็บท่าละ **2-3 นาที** หรือเซตละ 10-20 ครั้งต่อ Session
- เก็บข้อมูล **Idle** ด้วย เพราะโมเดลต้องรู้จักท่าพักด้วย
- ควรเก็บจากหลายมุมกล้อง/ระยะห่าง เพื่อเพิ่มความหลากหลาย
- ถ้าเก็บท่าผิด (Incorrect) ให้จงใจทำท่าผิดที่พบบ่อย เช่น สควอทเข่าเกินปลายเท้า

### 3.2.7 วิธีรัน

```bash
# เปิด Virtual Environment
.\venv\Scripts\activate

# รันโปรแกรมเก็บข้อมูล
python src/collect_data.py

# ควบคุม:
#   กดเลข 0-5 เลือกคลาส → กด Space เริ่มบันทึก → กด Space หยุด
#   กด U เพื่อ Upload ไฟล์
#   กด Q ออก
```

---

## 3.3 ขั้นตอนที่ 2: Feature Engineering & Pre-processing

### 3.3.1 Features ที่สกัดจาก MediaPipe (ทำใน collect_data.py)

| ประเภท | จำนวน | รายละเอียด |
|---|---|---|
| Raw Coordinates | 33 × 4 = 132 | x, y, z, visibility ของทุก Landmark |
| Joint Angles | 2 | มุมเข่าซ้าย (l_knee_angle), มุมเข่าขวา (r_knee_angle) |
| **รวม** | **134** | Features ต่อเฟรม |

### 3.3.2 การสร้าง Sequence ด้วย Sliding Window (ทำใน train_model.py)

```python
TIME_STEPS = 15  # จำนวนเฟรมต่อ 1 Sequence (≈ 0.75 วินาที)

# สร้าง Sliding Window
for i in range(len(features) - TIME_STEPS):
    window = features[i:(i + TIME_STEPS)]          # 15 เฟรม
    label = labels[i + TIME_STEPS - 1]              # Label ของเฟรมสุดท้าย
    X.append(window)
    y.append(label)

# ผลลัพธ์: X.shape = (N, 15, 134), y.shape = (N, 6)
```

### 3.3.3 การแบ่งข้อมูล

- **Train Set:** 80% (Stratified Random Split)
- **Test Set:** 20%
- **Random State:** 42 (สำหรับ Reproducibility)

---

## 3.4 ขั้นตอนที่ 3: การพัฒนาโมเดล (Model Development)

### 3.4.1 ไฟล์: `src/train_model.py`

### 3.4.2 สถาปัตยกรรม Bi-LSTM ที่ใช้

```
Layer (type)                    Output Shape          Param #
─────────────────────────────────────────────────────────────
Bidirectional(LSTM-64)          (None, 15, 128)       102,144
Dropout(0.2)                    (None, 15, 128)       0
Bidirectional(LSTM-32)          (None, 64)            41,216
Dropout(0.2)                    (None, 64)            0
Dense(32, relu)                 (None, 32)            2,080
Dense(6, softmax)               (None, 6)             198
─────────────────────────────────────────────────────────────
Total params: ~145,638
```

**คำอธิบายแต่ละ Layer:**

1. **Bidirectional(LSTM-64, return_sequences=True)**
   - LSTM 64 units ×2 ทิศทาง = 128 outputs per timestep
   - `return_sequences=True` ส่ง Output ทุก Timestep ไปยัง Layer ถัดไป
   - `activation='tanh'` (ค่า Default ของ LSTM)

2. **Dropout(0.2)**
   - สุ่มปิด 20% ของ Neurons ขณะเทรน เพื่อป้องกัน Overfitting
   - ทำให้โมเดลเสถียร (Stability) มากขึ้น

3. **Bidirectional(LSTM-32)**
   - LSTM 32 units ×2 = 64 outputs
   - ไม่มี `return_sequences` → ส่ง Output เฉพาะ Timestep สุดท้าย

4. **Dropout(0.2)**
   - ป้องกัน Overfitting อีกชั้น

5. **Dense(32, relu)**
   - Fully Connected Layer สำหรับเรียนรู้ Non-linear Patterns
   - ReLU activation ช่วยให้เรียนรู้ได้เร็ว

6. **Dense(6, softmax)**
   - Output Layer: 6 classes (Idle + 5 ท่าออกกำลังกาย)
   - Softmax แปลงเป็น Probability Distribution (ผลรวม = 1.0)

### 3.4.3 Hyperparameters

| Parameter | ค่า | เหตุผล |
|---|---|---|
| Optimizer | Adam | Adaptive learning rate, เหมาะกับ Time-series |
| Loss Function | Categorical Crossentropy | Multi-class classification |
| Batch Size | 32 | สมดุลระหว่าง Memory และ Gradient stability |
| Max Epochs | 100 | จำกัดไว้สูงสุด (EarlyStopping จะหยุดก่อน) |
| EarlyStopping | patience=10 | หยุดเทรนถ้า val_loss ไม่ลดลง 10 epochs ติดต่อกัน |
| restore_best_weights | True | กลับไปใช้ weights ที่ดีที่สุดอัตโนมัติ |

### 3.4.4 การ Export โมเดล

เทรนเสร็จจะได้ 2 ไฟล์:

1. **`models/exercise_bilstm.h5`** — Keras format (สำหรับวิเคราะห์/ปรับจูนเพิ่มเติม)
2. **`models/exercise_bilstm.tflite`** — TFLite format (สำหรับ Real-time Inference)
   - ใช้ **Dynamic Range Quantization** (`tf.lite.Optimize.DEFAULT`)
   - ลดขนาดโมเดล ~2-4 เท่า โดยคุณภาพลดลงน้อยมาก

### 3.4.5 วิธีรัน

```bash
# เทรนโมเดล
python -m src.train_model

# Output:
#   1. Loading and Preprocessing Data...
#   Data shape: X=(N, 15, 134), y=(N, 6)
#   2. Building Bi-LSTM Model...
#   3. Training Model...
#   Test Accuracy: XX.XX%
#   4. Exporting Models...
#   Success! Models saved to models/
```

---

## 3.5 ขั้นตอนที่ 4: Engine สำหรับทำนายแบบ Real-time

### 3.5.1 Mode A: Geometry Engine (`src/engines/geometry_engine.py`)

**หลักการ:** ใช้การคำนวณมุมระหว่างข้อต่อ (Joint Angle) เปรียบเทียบกับค่า Threshold เพื่อจำแนกระยะของการเคลื่อนไหว

**State Machine (เหมือนกันทุกท่า):**

```
Phase 0 (Initial/Ready)
    │  มุมข้ามเข้าเขต Transition
    ▼
Phase 1 (Going Down / Transition In)
    │  มุมข้ามเข้าเขต Peak
    ▼
Phase 2 (Peak / Maximum)
    │  มุมเริ่มกลับเข้าเขต Transition
    ▼
Phase 3 (Recovery / Transition Out)
    │  มุมกลับไปเขต Initial
    ▼
Phase 0 → นับ +1 Rep
```

**ตาราง Threshold สำหรับแต่ละท่า:**

| ท่า | จุด A | จุด B (Vertex) | จุด C | Stand/Plank/Down | Bottom/Peak/Up |
|---|---|---|---|---|---|
| Squat | Hip[23] | Knee[25] | Ankle[27] | > 160° | < 100° |
| Push-up | Shoulder[12] | Elbow[14] | Wrist[16] | > 160° | < 90° |
| Jumping Jack | Hip[24] | Shoulder[12] | Wrist[16] | < 45° | > 150° |

### 3.5.2 Mode C: Bi-LSTM Engine (`src/engines/lstm_engine.py`)

**หลักการ:** เก็บ Feature vector 15 เฟรมย้อนหลังด้วย Sliding Window (deque) จากนั้นส่งเข้า TFLite model เพื่อทำนายคลาส

**ขั้นตอนการทำงานในแต่ละเฟรม:**

```
1. รับ Landmarks จาก MediaPipe
2. แปลงเป็น Feature vector ขนาด 134
   ├── 33 landmarks × 4 (x, y, z, visibility) = 132
   └── 2 engineered features (l_knee_angle, r_knee_angle)
3. เพิ่มเข้า Frame Buffer (deque, maxlen=15)
4. ถ้า Buffer ยังไม่เต็ม → แสดง "Buffering..."
5. ถ้า Buffer เต็ม:
   a. สร้าง Input Tensor: shape (1, 15, 134)
   b. รัน TFLite Inference
   c. อ่าน Output: Probability ของ 6 คลาส
   d. เลือกคลาสที่มี Probability สูงสุด
   e. ตรวจสอบ Confidence Threshold (≥ 60%)
   f. อัปเดต State Machine เพื่อนับ Rep
```

**State Machine สำหรับนับ Rep (LSTM):**

```
Idle (Class 0) ──→ Active Class (1, 3, 4) ──→ Idle (Class 0) = +1 Rep
                    │
                    └─→ Incorrect Class (2, 5) → แจ้งเตือน "⚠ Form check"
```

---

## 3.6 ขั้นตอนที่ 5: หน้าจอแอปพลิเคชันหลัก (Main Application)

### 3.6.1 ไฟล์: `src/main_app.py`

### 3.6.2 องค์ประกอบ UI (HUD - Heads-Up Display)

```
┌──────────────────────────────────────────────────────┐
│ ┌────────────────────────────────────────┐            │
│ │ Mode: Mode A: Geometry                │            │
│ │ Exercise: Squat_Correct               │            │
│ │ FPS: 22.3                             │            │
│ │ Confidence: 95% (LSTM only)           │            │
│ └────────────────────────────────────────┘            │
│                                                      │
│              ┌──────────┐                            │
│              │          │                            │
│              │  Webcam  │                            │
│              │  + Pose  │                            │
│              │ Skeleton │                            │
│              │          │                            │
│              └──────────┘                            │
│                                                      │
│ ┌────────────────────────────────────────────────────┐│
│ │ REPS: 5                    Angle: 95.3°           ││
│ │ Good depth!                Phase: ● ● ● ○        ││
│ └────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
```

### 3.6.3 ปุ่มควบคุม

| ปุ่ม | การทำงาน |
|---|---|
| `A` | สลับเป็น Mode A: Geometry (Baseline) |
| `C` | สลับเป็น Mode C: Bi-LSTM |
| `1` | เลือกท่า Squat |
| `2` | เลือกท่า Push-up |
| `3` | เลือกท่า Jumping Jack |
| `R` | รีเซ็ตตัวนับ Rep |
| `Q` | ออกจากโปรแกรม |

### 3.6.4 วิธีรัน

```bash
# เปิด Virtual Environment
.\venv\Scripts\activate

# รันแอปพลิเคชันหลัก
python -m src.main_app
```

---

## 3.7 สรุปโครงสร้างไฟล์ทั้งหมด (Final Project Structure)

```
Project_Exercise/
├── .gitignore                    ← กำหนดไฟล์ที่ไม่ต้อง Track ใน Git
├── Exercise.md                   ← แผนงานและ Blueprint ของโปรเจกต์
├── requirements.txt              ← รายการ Library ที่ต้องติดตั้ง
├── docs/
│   └── Development_Documentation.md  ← เอกสารฉบับนี้
├── data/                         ← ข้อมูลที่เก็บจากโปรแกรม
│   ├── landmarks/                ← CSV ไฟล์พิกัด + Features
│   ├── raw_images/               ← ภาพ JPG ทีละเฟรม
│   └── raw_videos/               ← วิดีโอ MP4
├── models/                       ← โมเดลที่เทรนเสร็จแล้ว
│   ├── exercise_bilstm.h5        ← Keras format
│   └── exercise_bilstm.tflite    ← TFLite format (ใช้ Real-time)
├── notebooks/                    ← Jupyter Notebook สำหรับวิเคราะห์ผล
├── results/                      ← ผลการทดลอง, กราฟ, ตาราง
├── assets/                       ← รูปภาพ/สื่อประกอบเอกสาร
├── src/
│   ├── engines/
│   │   ├── base_engine.py        ← Abstract Base Class (Strategy Pattern)
│   │   ├── geometry_engine.py    ← Mode A: Geometry Baseline
│   │   └── lstm_engine.py        ← Mode C: Bi-LSTM Engine
│   ├── utils/
│   │   └── geometry.py           ← ฟังก์ชันคำนวณมุม + Normalize
│   ├── collect_data.py           ← โปรแกรมเก็บข้อมูล
│   ├── train_model.py            ← สคริปต์เทรนโมเดล Bi-LSTM
│   └── main_app.py               ← แอปพลิเคชันหลัก (UI)
└── venv/                         ← Python Virtual Environment
```

---

## 3.8 ขั้นตอนการติดตั้งและเตรียมสภาพแวดล้อม (Environment Setup)

### 3.8.1 ความต้องการของระบบ (System Requirements)

| องค์ประกอบ | ข้อกำหนดขั้นต่ำ | แนะนำ |
|---|---|---|
| OS | Windows 10 64-bit | Windows 11 |
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / Ryzen 7 |
| RAM | 8 GB | 16-32 GB |
| GPU | - (CPU-only OK) | NVIDIA RTX 3050+ (สำหรับเทรน) |
| Python | 3.9 | 3.10 |
| Webcam | 720p | 1080p |

### 3.8.2 ขั้นตอนการติดตั้ง

```bash
# 1. Clone โปรเจกต์
git clone https://github.com/RAM-D-PAGE/Project_Exercise.git
cd Project_Exercise

# 2. สร้าง Virtual Environment
python -m venv venv

# 3. เปิดใช้งาน Virtual Environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. ติดตั้ง Dependencies
pip install -r requirements.txt

# 5. ตรวจสอบการติดตั้ง
python -c "import cv2; import mediapipe; import tensorflow; print('All OK!')"
```

### 3.8.3 รายการ Dependencies (`requirements.txt`)

| Library | เวอร์ชัน | การใช้งาน |
|---|---|---|
| opencv-python | ล่าสุด | อ่าน/เขียนภาพ, เปิด Webcam, แสดง UI |
| opencv-contrib-python | ล่าสุด | โมดูลเสริมของ OpenCV |
| mediapipe | 0.10.x | Pose Estimation (33 Landmarks) |
| tensorflow | 2.x | สร้าง/เทรน Bi-LSTM, แปลง TFLite |
| pandas | ล่าสุด | อ่าน/เขียน CSV |
| scikit-learn | ล่าสุด | train_test_split, Metrics |
| numpy | ล่าสุด | การคำนวณเชิงตัวเลข |
| matplotlib | ล่าสุด | สร้างกราฟ, Visualize ผลลัพธ์ |

---

# บทที่ 4: ผลการทดลองและวิเคราะห์ (Results & Analysis)

> **หมายเหตุ:** บทนี้จะเติมข้อมูลจริงหลังจากเก็บข้อมูลและเทรนโมเดลเสร็จสิ้น

## 4.1 การทดลองที่ 1: Single Frame Accuracy (Exp 1)

**วัตถุประสงค์:** วัดความสามารถในการจำแนก Phase 0-3 จากภาพเฟรมเดียว

**วิธีการ:** ใช้ Geometry Engine (Mode A) ทำนาย Phase ของแต่ละเฟรม แล้วเทียบกับ Ground Truth

**ตารางผลลัพธ์ (Template):**

| ท่า | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Squat | _% | _% | _% | _ |
| Push-up | _% | _% | _% | _ |
| Jumping Jack | _% | _% | _% | _ |
| **Average** | _% | _% | _% | _ |

## 4.2 การทดลองที่ 2: Sequence Analysis — Bi-LSTM vs Baseline (Exp 2)

**วัตถุประสงค์:** เปรียบเทียบความแม่นยำระหว่าง Mode A (Geometry), Mode B (LSTM), Mode C (Bi-LSTM)

**ตารางเปรียบเทียบ (Template):**

| Metric | Mode A (Geometry) | Mode B (LSTM) | Mode C (Bi-LSTM) |
|---|---|---|---|
| Overall Accuracy | _% | _% | _% |
| Macro F1-Score | _ | _ | _ |
| Weighted F1-Score | _ | _ | _ |
| Training Time | N/A | _ min | _ min |

**Confusion Matrix:** _(จะแนบภาพจาก Notebook)_

## 4.3 การทดลองที่ 3: Practical Application (Exp 3)

**วัตถุประสงค์:** วัดประสิทธิภาพการนับ Rep และ Feedback กับผู้ทดสอบจริง

**ตารางผลลัพธ์ (Template):**

| ท่า | Actual Reps | Mode A Count | Mode C Count | Mode A Error% | Mode C Error% |
|---|---|---|---|---|---|
| Squat × 10 | 10 | _ | _ | _% | _% |
| Push-up × 10 | 10 | _ | _ | _% | _% |
| Jumping Jack × 10 | 10 | _ | _ | _% | _% |

## 4.4 การทดลองที่ 4: Technical Performance (Exp 4)

**วัตถุประสงค์:** วัด Inference Time และ FPS

**ตารางผลลัพธ์ (Template):**

| Metric | Mode A (Geometry) | Mode C (Bi-LSTM) | Mode C (TFLite) |
|---|---|---|---|
| Avg Inference Time | _ ms | _ ms | _ ms |
| Average FPS | _ | _ | _ |
| Model Size | N/A | _ MB | _ MB |

---

# บทที่ 5: สรุป อภิปราย และข้อเสนอแนะ (Conclusion)

> **หมายเหตุ:** บทนี้จะเขียนหลังจากรวบรวมผลการทดลองเรียบร้อย

## 5.1 สรุปผลการดำเนินงาน

_(จะเติมหลังทดลองเสร็จ)_

- ระบบสามารถตรวจจับและจำแนกท่าทางการออกกำลังกายได้ ___ ท่า
- ความแม่นยำโดยรวมของ Bi-LSTM อยู่ที่ ___% เทียบกับ Geometry Baseline ที่ ___%
- ระบบทำงานได้แบบ Real-time ที่ ___ FPS บน CPU

## 5.2 อภิปรายผล

_(จะเติมหลังทดลองเสร็จ)_

ประเด็นที่ควรอภิปราย:
- ทำไม Bi-LSTM ดีกว่า/แย่กว่า Geometry ในบางท่า?
- Window Size ส่งผลต่อ Latency อย่างไร?
- Quantization ส่งผลต่อ Accuracy มากน้อยแค่ไหน?

## 5.3 ข้อจำกัดของงานวิจัย

1. รองรับผู้ใช้งานเพียง 1 คนในเฟรม
2. ต้องการแสงสว่างที่เพียงพอ
3. ท่าออกกำลังกายจำกัดอยู่ที่ 4 ท่า
4. ยังไม่รองรับอุปกรณ์ Mobile โดยตรง

## 5.4 ข้อเสนอแนะสำหรับงานในอนาคต (ข้อเสนอแนะสำหรับการต่อยอด)

จากโครงสร้างและเอกสารที่ออกแบบไว้ ระบบ AI Exercise Analysis ถือว่าวางรากฐานไว้ดีมาก ทั้งการใช้ Strategy Pattern ที่รองรับการสลับ Engine และการใช้ Bi-LSTM เพื่อแก้ปัญหา Time-series
เพื่อการพัฒนาในระยะต่อไป นี่คือแนวทางที่สามารถนำไปต่อยอดได้ครับ:

### 1. ด้านการขยายขีดความสามารถของโมเดล (Model & AI Enhancements)
* **อัปเกรดสถาปัตยกรรม AI:** ทดลองเปลี่ยนจากการใช้ LSTM ไปเป็น Transformer-based Model ซึ่งจะช่วยให้การเรียนรู้ลำดับข้อมูลแบบ Time-series มีประสิทธิภาพมากขึ้น โดยสามารถใช้ทรัพยากรการ์ดจออย่าง RTX 3050 ที่มีอยู่เพื่อเร่งความเร็วในการเทรนโมเดลที่มีความซับซ้อนระดับนี้ได้สบายๆ

### 2. ด้านการพัฒนาแอปพลิเคชันและฐานข้อมูล (Application & Platform)
* **พัฒนาเป็น Web Application เต็มรูปแบบ:** นำระบบไปให้บริการบนเว็บไซต์ โดยสามารถใช้ React และ TypeScript สร้างหน้าแดชบอร์ดสรุปผลการออกกำลังกายที่สวยงาม (ใช้ Framer Motion ช่วยเรื่องแอนิเมชัน) และเชื่อมต่อกับ Supabase เพื่อจัดการระบบสมาชิกและบันทึกข้อมูลแบบเรียลไทม์
* **ระบบอัตโนมัติและการแจ้งเตือน (Automation):** นำเครื่องมืออย่าง n8n มาช่วยผูก Workflow ดึงข้อมูลสถิติจากฐานข้อมูล เพื่อส่งรายงานสรุปผลการออกกำลังกายรายสัปดาห์ผ่านแอปพลิเคชันแชท
* **เพิ่มระบบ Gamification:** สร้างแรงจูงใจให้ผู้ใช้งานด้วยการเพิ่มระบบเก็บแต้ม, ระดับเลเวล, หรือกระดานผู้นำ (Leaderboard)

### 3. ด้านการรองรับการใช้งานจริง (Practical Usability)
* **ระบบตรวจจับหลายบุคคล:** พัฒนาเพิ่มระบบ Multi-person Tracking เพื่อให้สามารถวิเคราะห์ท่าทางของผู้ใช้งานได้มากกว่า 1 คนในเฟรมเดียวกัน
* **ระบบ Voice Feedback:** เพิ่มการแจ้งเตือนด้วยเสียง (Text-to-Speech) เมื่อระบบตรวจพบการทำท่าผิดรูปแบบ (Incorrect Phase) เพื่อให้ผู้ใช้ปรับท่าทางได้ทันทีโดยไม่ต้องละสายตามามองหน้าจอขณะออกกำลังกาย

---

# ภาคผนวก (Appendix)

## ก. ซอร์สโค้ดหลัก (Key Source Code)

### ก.1 Abstract Base Class — `base_engine.py`

```python
from abc import ABC, abstractmethod
import numpy as np

class ExerciseEngine(ABC):
    def __init__(self, exercise_name: str):
        self.exercise_name = exercise_name
        self.reps_count = 0
        self.current_phase = 0
        self.feedback = "Ready"
        self.performance_logs = []

    @abstractmethod
    def process(self, landmarks, frame_size):
        pass

    def reset(self):
        self.reps_count = 0
        self.current_phase = 0
        self.feedback = "Ready"
        self.performance_logs = []
```

### ก.2 ฟังก์ชันคำนวณมุม — `geometry.py`

```python
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)
```

## ข. ดัชนี MediaPipe Landmarks ทั้ง 33 จุด

| Index | Name | Index | Name |
|---|---|---|---|
| 0 | Nose | 17 | Left Pinky |
| 1 | Left Eye Inner | 18 | Right Pinky |
| 2 | Left Eye | 19 | Left Index |
| 3 | Left Eye Outer | 20 | Right Index |
| 4 | Right Eye Inner | 21 | Left Thumb |
| 5 | Right Eye | 22 | Right Thumb |
| 6 | Right Eye Outer | 23 | **Left Hip** |
| 7 | Left Ear | 24 | **Right Hip** |
| 8 | Right Ear | 25 | **Left Knee** |
| 9 | Mouth Left | 26 | Right Knee |
| 10 | Mouth Right | 27 | **Left Ankle** |
| 11 | **Left Shoulder** | 28 | Right Ankle |
| 12 | **Right Shoulder** | 29 | Left Heel |
| 13 | Left Elbow | 30 | Right Heel |
| 14 | **Right Elbow** | 31 | Left Foot Index |
| 15 | Left Wrist | 32 | Right Foot Index |
| 16 | **Right Wrist** | | |

> จุดที่ตัวหนาคือจุดที่ใช้ในการคำนวณมุมสำหรับ Squat, Push-up และ Jumping Jack

## ค. คำสั่ง Git ที่ใช้ในโปรเจกต์

```bash
# เริ่มต้นโปรเจกต์
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/RAM-D-PAGE/Project_Exercise.git
git push -u origin main

# อัปเดตเวอร์ชัน
git add .
git commit -m "V0.X - [description]"
git push
```

## ง. การแก้ไขปัญหาที่พบบ่อย (Troubleshooting)

| ปัญหา | สาเหตุ | วิธีแก้ |
|---|---|---|
| MediaPipe import error | ใช้ Python Global แทน venv | เปิด `.\venv\Scripts\activate` ก่อนรัน |
| No CSV files found | ยังไม่ได้เก็บข้อมูล | รัน `collect_data.py` เก็บข้อมูลก่อน |
| Model not loaded | ยังไม่ได้เทรนโมเดล | รัน `train_model.py` ก่อน |
| Low FPS | Model complexity สูง | ลด `model_complexity` เป็น 0 |
| CRLF warnings | Windows line endings | เพิ่ม `git config core.autocrlf true` |

---

> **📝 วันที่สร้างเอกสาร:** 15 มิถุนายน 2026
> **📌 เวอร์ชัน:** 1.0
> **👤 ผู้พัฒนา:** RAM-D-PAGE
> **🔗 Repository:** https://github.com/RAM-D-PAGE/Project_Exercise
