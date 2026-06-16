# AI Exercise Model Reports

โฟลเดอร์สำหรับเก็บรายงานผลการประเมินประสิทธิภาพโมเดล (Model Evaluation Reports) 

ไฟล์ที่จะถูกบันทึกที่นี่หลังจากรันกระบวนการเทรน (`src/train_model.py` หรือตัวเลือก `[4]` ใน Launcher):
1. **`training_history_YYYYMMDD_HHMMSS.png`**: กราฟแสดงค่า Accuracy และ Loss ของโมเดลแบบ Real-time ตลอดการเทรน
2. **`confusion_matrix_YYYYMMDD_HHMMSS.png`**: กราฟ Matrix แสดงผลการทายถูก/ผิดของท่าทางแต่ละประเภท
3. **`roc_curve_YYYYMMDD_HHMMSS.png`**: กราฟ Receiver Operating Characteristic (ROC) แสดงความสามารถในการแยกแยะแต่ละคลาส และค่า AUC ของโมเดล
4. **`classification_report_YYYYMMDD_HHMMSS.txt`**: รายละเอียดสรุปประสิทธิภาพของโมเดล (ค่า Precision, Recall, F1-Score, ค่า Accuracy ภาพรวม และค่า ROC AUC รายคลาส)

*หมายเหตุ: ระบบจะต่อท้ายชื่อไฟล์ด้วยระบบวันเวลาที่กดเทรน (Timestamp) เพื่อป้องกันการเขียนทับรายงานของเว่อร์ชั่นก่อนหน้าทุกครั้ง*
