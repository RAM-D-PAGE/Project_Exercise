# AI-Powered Exercise Analysis System using MediaPipe and Bi-LSTM

ระบบวิเคราะห์และตรวจจับท่าทางการออกกำลังกายอัจฉริยะ โดยใช้การตรวจจับโครงร่างมนุษย์ (MediaPipe Pose) และการประมวลผลลำดับเวลาด้วยโครงข่ายประสาทเทียมแบบ Bi-directional LSTM

## 📌 บทนำ (Introduction)

โปรเจกต์นี้พัฒนาขึ้นเพื่อเพิ่มประสิทธิภาพในการออกกำลังกายและกายภาพบำบัด โดยระบบสามารถจำแนกท่าทางการเคลื่อนไหว ตรวจสอบความถูกต้องของท่าทาง และนับจำนวนครั้ง (Repetition Counting) ได้แบบเรียลไทม์ พร้อมระบบผู้ช่วยฝึกสอน (AI Trainer) เพื่อยกระดับการออกกำลังกายให้ปลอดภัยและถูกต้องตามหลักสรีรศาสตร์

## 🎯 วัตถุประสงค์และขอบเขตการดำเนินงาน (Objectives & Scope)

- **วัตถุประสงค์:** พัฒนาโมเดล Deep Learning ที่สามารถวิเคราะห์ลำดับการเคลื่อนที่ที่มีความซับซ้อนได้แม่นยำกว่าการใช้ภาพนิ่ง
- **ขอบเขต (Scope):**
    - รองรับผู้ใช้งาน 1 ท่านในเฟรมภาพ (Single Person Tracking)
    - รองรับ 4 ท่าออกกำลังกายหลัก: Squats, Push-ups, Jumping Jacks และ Bicep Curls
    - แสดงผลผ่านเว็บแคมในสภาวะแสงที่เหมาะสม (Indoor/Outdoor Ambient Light)
    - ประมวลผลแบบ Real-time บนเครื่องคอมพิวเตอร์ระดับทั่วไป (CPU-based)

## 🏗 กระบวนการพัฒนา (Development Methodology)

แบ่งการดำเนินงานออกเป็น 4 ระยะตามมาตรฐานวิศวกรรมซอฟต์แวร์:

1. **Data Collection:** บันทึกพิกัดโครงร่าง (Landmarks) 33 จุดในรูปแบบ Time-series (Sequence)
2. **Feature Engineering & Pre-processing:** สกัดคุณลักษณะสำคัญเพื่อเพิ่มความแม่นยำ:
    - **Joint Angles:** คำนวณมุมข้อต่อที่เกี่ยวข้องกับท่านั้นๆ
    - **Relative Displacement:** ปรับพิกัดให้สัมพันธ์กับจุดศูนย์กลางร่างกาย (Mid-Hip)
    - **Velocity Analysis:** คำนวณความเร็วของการขยับแต่ละจุดสำคัญ
    - **Normalization:** ปรับสเกลข้อมูลให้อยู่ในระดับมาตรฐาน
3. **Model Development:** พัฒนา Bi-directional LSTM เพื่อประมวลผลลำดับเหตุการณ์แบบล่วงหน้าและย้อนกลับ
4. **Deployment & Feedback:** พัฒนาส่วนแสดงผลพร้อมระบบ State Machine สำหรับนับครั้งและให้คำแนะนำ

## ⚙️ สถาปัตยกรรมระบบ (Multi-Engine Architecture)

ระบบออกแบบตาม **Strategy Pattern** เพื่อรองรับการรันหลายรูปแบบ (Execution Modes):
- **Mode A: Mathematical Geometry** - ใช้การคำนวณมุมคณิตศาสตร์ (Baseline)
- **Mode B: Vanilla LSTM** - ใช้ LSTM ทิศทางเดียว (อดีต → ปัจจุบัน)
- **Mode C: Bi-LSTM** - ใช้ Bi-directional LSTM (Proposed Method)

## 🏃 รายละเอียดระยะการเคลื่อนที่ (Detailed Movement Phases)

| ท่าออกกำลังกาย | Phase 0 (Initial) | Phase 1 (Transition) | Phase 2 (Peak) | Phase 3 (Recovery) |
| :--- | :--- | :--- | :--- | :--- |
| **Squats** | ยืนตรงเตรียม | ช่วงย่อตัวลง | ต้นขาขนานพื้น | ช่วงยืดตัวกลับ |
| **Push-ups** | High Plank | ช่วงลดตัวลง | หน้าอกใกล้พื้น | ช่วงดันตัวกลับ |
| **Jumping Jacks** | ยืนตรงขาชิด | ช่วงกางแขน-ขา | มือเหนือศีรษะ | ช่วงหุบแขน-ขา |
| **Bicep Curls** | แขนเหยียดตรง | ช่วงยกน้ำหนักขึ้น | มือใกล้ไหล่สุด | ช่วงผ่อนแขนลง |

## 🖥 ส่วนแสดงผลผู้ใช้งาน (User Interface Features)

เพื่อให้เกิดการใช้งานที่ง่ายและมีประสิทธิภาพ ระบบจึงมีฟีเจอร์การแสดงผลดังนี้:
- **Skeleton Overlay:** เส้นโครงร่างที่เปลี่ยนสีตามความถูกต้อง (เขียว = ถูกต้อง / แดง = ผิดพลาด)
- **Counter Dashboard:** ตัวเลขแสดงรายการท่าทางและจำนวนครั้งที่ทำสำเร็จแบบเรียลไทม์
- **Real-time Feedback:** ข้อความแนะนำเชิงภาพ เช่น "Lower your hips" หรือ "Straighten your back"

## 🧪 การออกแบบการทดลอง (Experimental Design)

1. **Exp 1: Single Frame Accuracy** - วัดความสามารถในการแยกแยะ Phase 0-3 ในภาพนิ่ง
2. **Exp 2: Sequence Analysis** - เปรียบเทียบ Bi-LSTM vs LSTM vs Baseline (วัด Recall & F1-Score)
3. **Exp 3: Practical Application** - วัด Counting Error (%) และ Feedback Accuracy จากคนทดสอบจริง
4. **Exp 4: Technical Performance** - วัด Inference Time (ms) และ FPS ระหว่าง Standard และ TFLite Mode

## 🛠 เทคโนโลยีที่ใช้ (Tech Stack)

- **Programming Language:** Python
- **Pose Estimation:** MediaPipe Pose (Complexity 0/1)
- **Deep Learning Framework:** TensorFlow / Keras (Training) & TensorFlow Lite (Inference/Quantization)
- **Image Processing:** OpenCV
- **Data Analysis:** NumPy, Pandas, Scikit-learn
- **Hardware Profile:** Optimized for CPU (i5/Ryzen 5+) & Support GPU (RTX 3050 สำหรับการ Training)

## 🚀 วิธีการใช้งานเบื้องต้น (Quick Start)

1. ตั้งค่าสภาพแวดล้อม: `pip install -r requirements.txt`
2. บันทึกข้อมูลเพื่อฝึกสอน: `python src/collect_data.py`
3. เริ่มต้นใช้งานระบบหลัก: `python src/main_app.py --mode bi-lstm --lite true`

## 💡 ข้อมูลทางเทคนิคเชิงลึก (Technical Insight for Q&A)

> [!TIP]
> **ทำไมต้องใช้ Bi-directional LSTM ในเมื่อมันอาจทำให้เกิด Delay?**
> ในระบบนี้ใช้ Window Size ขนาดสั้น (เช่น 0.5 - 1.0 วินาที) ทำให้ความหน่วงที่เกิดขึ้นแทบไม่ส่งผลต่อความรู้สึกของผู้ใช้ แต่แลกมาด้วยความแม่นยำในการแยกแยะจังหวะ "ขึ้น" และ "ลง" ได้ดีกว่า LSTM แบบทิศทางเดียว เนื่องจากโมเดลมีโอกาสเห็นบทสรุปของท่าทางในวินาทีถัดไปก่อนทำการตัดสินใจ (Look-ahead capability)

## 📁 โครงสร้างไฟล์ในโปรเจกต์ (Project Structure)

```text
├── data/               # ไฟล์พิกัดที่สกัดออกมา (CSV)
├── models/             # โมเดลฝึกสอน (.h5, .tflite)
├── src/
│   ├── engines/        # Class สำหรับ Strategy Pattern
│   ├── collect_data.py # เก็บข้อมูลจาก Webcam
│   ├── train_model.py  # ฝึกสอน Bi-LSTM
│   ├── main_app.py     # โปรแกรมหลัก (สมามารถสลับโหมดได้)
│   └── utils/          # geometry.py, visualization.py
├── notebooks/          # การวิเคราะห์ผลการทดลอง (EDA)
├── requirements.txt    # รายการ Library
└── README.md
```