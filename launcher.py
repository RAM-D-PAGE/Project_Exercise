"""
🏋️ AI Exercise Analysis System — Main Launcher
================================================
ไฟล์หลักสำหรับเข้าถึงทุกโปรแกรมในโปรเจกต์
รัน: python launcher.py
"""

import os
import sys
import subprocess

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    clear_screen()
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║                                                      ║")
    print("  ║   🏋️  AI EXERCISE ANALYSIS SYSTEM                   ║")
    print("  ║   MediaPipe + Bi-LSTM Real-time Trainer              ║")
    print("  ║                                                      ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()

def print_menu():
    print("  ┌──────────────────────────────────────────────────────┐")
    print("  │  📋  MAIN MENU                                      │")
    print("  ├──────────────────────────────────────────────────────┤")
    print("  │                                                      │")
    print("  │  [1]  🎮  Main App (Real-time Exercise Tracker)      │")
    print("  │         เปิดกล้อง วิเคราะห์ท่า นับ Rep แบบสดๆ       │")
    print("  │                                                      │")
    print("  │  [2]  📸  Collect Data (เก็บข้อมูลสำหรับเทรน)       │")
    print("  │         บันทึกพิกัด Landmark จาก Webcam/วิดีโอ       │")
    print("  │                                                      │")
    print("  │  [3]  🤖  Auto-Label Video (แปะป้ายอัตโนมัติ)       │")
    print("  │         ใช้ Geometry ช่วยตัดช่วงวิดีโอเป็น Dataset   │")
    print("  │                                                      │")
    print("  │  [4]  🧠  Train Model (เทรน Bi-LSTM)                │")
    print("  │         เทรนโมเดลและ Export เป็น TFLite              │")
    print("  │                                                      │")
    print("  │  [5]  🖥️  Video Review Display                      │")
    print("  │         ตรวจสอบความถูกต้องของ AI จากคลิปวิดีโอ       │")
    print("  │                                                      │")
    print("  ├──────────────────────────────────────────────────────┤")
    print("  │                                                      │")
    print("  │  [6]  📂  ดูโครงสร้างโปรเจกต์                       │")
    print("  │  [7]  📊  ดูสถานะข้อมูล & โมเดล                     │")
    print("  │  [8]  📖  เปิดเอกสาร (Development Documentation)    │")
    print("  │                                                      │")
    print("  │  [0]  ❌  ออกจากโปรแกรม                              │")
    print("  │                                                      │")
    print("  └──────────────────────────────────────────────────────┘")
    print()

def run_script(script_path):
    """รันสคริปต์ Python แล้วกลับมาที่เมนูหลัก"""
    print(f"\n  🚀 กำลังเปิด: {script_path}\n")
    print("=" * 60)
    try:
        subprocess.run([sys.executable, script_path], cwd=os.getcwd())
    except KeyboardInterrupt:
        print("\n  ⏹ หยุดโดยผู้ใช้")
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    input("  กด Enter เพื่อกลับเมนูหลัก...")

def show_project_structure():
    """แสดงโครงสร้างโปรเจกต์"""
    clear_screen()
    print("\n  📂 โครงสร้างโปรเจกต์\n")
    print("  Project_Exercise/")
    print("  ├── launcher.py                  ← 🚀 ไฟล์นี้ (Main Launcher)")
    print("  ├── Exercise.md                  ← แผนงาน & Blueprint")
    print("  ├── requirements.txt             ← รายการ Library")
    print("  ├── .gitignore")
    print("  │")
    print("  ├── src/")
    print("  │   ├── main_app.py              ← [1] แอป Real-time Tracker")
    print("  │   ├── collect_data.py           ← [2] เก็บข้อมูล Landmark")
    print("  │   ├── auto_label_video.py       ← [3] Auto-Label วิดีโอ")
    print("  │   ├── train_model.py            ← [4] เทรน Bi-LSTM")
    print("  │   ├── review_video.py           ← [5] Video Review Display")
    print("  │   ├── engines/")
    print("  │   │   ├── base_engine.py        ← Abstract Base Class")
    print("  │   │   ├── geometry_engine.py    ← Mode A: Geometry")
    print("  │   │   └── lstm_engine.py        ← Mode C: Bi-LSTM")
    print("  │   └── utils/")
    print("  │       └── geometry.py           ← ฟังก์ชันคำนวณมุม")
    print("  │")
    print("  ├── data/")
    print("  │   ├── landmarks/                ← CSV ไฟล์พิกัด")
    print("  │   ├── raw_images/               ← ภาพ JPG")
    print("  │   └── raw_videos/               ← วิดีโอ MP4")
    print("  │")
    print("  ├── models/                       ← โมเดลที่เทรนแล้ว (.h5, .tflite)")
    print("  ├── docs/                         ← เอกสารโปรเจกต์")
    print("  ├── notebooks/                    ← Jupyter Notebook")
    print("  └── results/                      ← ผลการทดลอง")
    print()
    input("  กด Enter เพื่อกลับเมนูหลัก...")

def show_data_status():
    """แสดงสถานะข้อมูลและโมเดล"""
    clear_screen()
    print("\n  📊 สถานะข้อมูล & โมเดล\n")
    
    # ตรวจสอบ CSV ในแต่ละโฟลเดอร์
    print("  ── Data (Landmarks CSV) ──────────────────────────────")
    landmark_dir = os.path.join("data", "landmarks")
    if os.path.isdir(landmark_dir):
        total_csv = 0
        for root, dirs, files in os.walk(landmark_dir):
            csv_files = [f for f in files if f.endswith('.csv')]
            if csv_files:
                folder_name = os.path.relpath(root, landmark_dir)
                total_size = sum(os.path.getsize(os.path.join(root, f)) for f in csv_files)
                print(f"    📁 {folder_name}: {len(csv_files)} ไฟล์ ({total_size/1024:.1f} KB)")
                total_csv += len(csv_files)
        if total_csv == 0:
            print("    ⚠️  ยังไม่มีข้อมูล — รัน [2] Collect Data หรือ [3] Auto-Label ก่อน")
        else:
            print(f"    ── รวม: {total_csv} ไฟล์ CSV")
    else:
        print("    ⚠️  ไม่พบโฟลเดอร์ data/landmarks/")

    # ตรวจสอบวิดีโอ
    print("\n  ── Raw Videos ────────────────────────────────────────")
    video_dir = os.path.join("data", "raw_videos")
    if os.path.isdir(video_dir):
        total_videos = 0
        for root, dirs, files in os.walk(video_dir):
            vids = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            if vids:
                folder_name = os.path.relpath(root, video_dir)
                total_size = sum(os.path.getsize(os.path.join(root, f)) for f in vids)
                print(f"    📁 {folder_name}: {len(vids)} ไฟล์ ({total_size/(1024*1024):.1f} MB)")
                total_videos += len(vids)
        if total_videos == 0:
            print("    ⚠️  ยังไม่มีวิดีโอ")
        else:
            print(f"    ── รวม: {total_videos} วิดีโอ")
    else:
        print("    ⚠️  ไม่พบโฟลเดอร์ data/raw_videos/")

    # ตรวจสอบโมเดล
    print("\n  ── Models ────────────────────────────────────────────")
    model_dir = "models"
    if os.path.isdir(model_dir):
        model_files = os.listdir(model_dir)
        if model_files:
            for f in model_files:
                fpath = os.path.join(model_dir, f)
                size = os.path.getsize(fpath)
                if size > 1024 * 1024:
                    print(f"    ✅ {f} ({size/(1024*1024):.1f} MB)")
                else:
                    print(f"    ✅ {f} ({size/1024:.1f} KB)")
        else:
            print("    ⚠️  ยังไม่มีโมเดล — รัน [4] Train Model ก่อน")
    else:
        print("    ⚠️  ยังไม่มีโมเดล — รัน [4] Train Model ก่อน")

    # แนะนำ Workflow
    print("\n  ── 🗺️  ลำดับการทำงานที่แนะนำ ─────────────────────")
    print("    ขั้นที่ 1: [2] Collect Data   หรือ  [3] Auto-Label")
    print("    ขั้นที่ 2: [5] Video Review   ← ตรวจสอบข้อมูลก่อนเทรน")
    print("    ขั้นที่ 3: [4] Train Model")
    print("    ขั้นที่ 4: [5] Video Review   ← ตรวจสอบผลหลังเทรน")
    print("    ขั้นที่ 5: [1] Main App → กด C เลือก Mode Bi-LSTM")
    print()
    input("  กด Enter เพื่อกลับเมนูหลัก...")

def open_documentation():
    """เปิดเอกสาร"""
    doc_path = os.path.join("docs", "Development_Documentation.md")
    if os.path.exists(doc_path):
        print(f"\n  📖 กำลังเปิด: {doc_path}")
        if os.name == 'nt':
            os.startfile(doc_path)
        elif sys.platform == 'darwin':
            subprocess.run(['open', doc_path])
        else:
            subprocess.run(['xdg-open', doc_path])
    else:
        print(f"\n  ❌ ไม่พบไฟล์ {doc_path}")
    
    input("\n  กด Enter เพื่อกลับเมนูหลัก...")

def main():
    while True:
        print_banner()
        print_menu()

        choice = input("  👉 กดเลข (0-8): ").strip()

        if choice == '1':
            run_script(os.path.join("src", "main_app.py"))
        elif choice == '2':
            run_script(os.path.join("src", "collect_data.py"))
        elif choice == '3':
            run_script(os.path.join("src", "auto_label_video.py"))
        elif choice == '4':
            run_script(os.path.join("src", "train_model.py"))
        elif choice == '5':
            run_script(os.path.join("src", "review_video.py"))
        elif choice == '6':
            show_project_structure()
        elif choice == '7':
            show_data_status()
        elif choice == '8':
            open_documentation()
        elif choice == '0':
            clear_screen()
            print("\n  👋 ขอบคุณที่ใช้งาน! ออกจากโปรแกรม\n")
            break
        else:
            input("  ❌ กรุณากดเลข 0-8 เท่านั้น (กด Enter เพื่อลองใหม่)")

if __name__ == "__main__":
    main()
