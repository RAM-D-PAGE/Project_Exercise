# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys

class WorkoutAnalytics:
    def __init__(self, exercise_name, fps=30.0):
        self.exercise_name = exercise_name
        self.fps = fps
        self.start_time = time.time()
        
        # ล็อกข้อมูลรายเฟรม
        self.frames = []      # list of dicts: {'frame_idx': int, 'angle': float, 'rep_count': int, 'state': str}
        self.rep_durations = []  # list of floats (วินาทีในแต่ละ rep)
        self.rep_status = []     # list of bools (True = correct, False = incorrect rhythm)
        
        # ตัวแปลสำหรับจับเวลาในแต่ละ rep (นับจำนวนเฟรมที่อยู่นอก Phase 0)
        self.active_frame_count = 0
        self.last_rep_count = 0
        self.rep_was_incorrect = False  # ตัวแปรจำว่าใน rep ปัจจุบัน มีทำท่าผิดฟอร์มไหม

    def log_frame(self, frame_idx, angle, rep_count, phase, status_color, confidence=None, predicted_class=None):
        """บันทึกข้อมูลในเฟรมนั้นๆ เพื่อไปทำกราฟสรุป"""
        # ระบุสถานะความถูกต้องจากสีของ HUD
        # status_color: (B, G, R) ของ OpenCV
        if status_color == (0, 0, 255):      # สีแดง = ท่าผิด
            state = 'Incorrect'
            self.rep_was_incorrect = True
        elif status_color == (0, 255, 0):    # สีเขียว = ท่าถูก
            state = 'Correct'
        else:                                # สีอื่นๆ / เทา / ส้ม = Idle / Buffering
            state = 'Idle'
            
        self.frames.append({
            'frame_idx': frame_idx,
            'angle': angle,
            'rep_count': rep_count,
            'state': state,
            'confidence': confidence if confidence is not None else 0.0,
            'predicted_class': predicted_class if predicted_class is not None else ''
        })

        # คอนโทรลเรื่องการจับเวลาของแต่ละ Rep ด้วยการนับเฟรม
        if phase > 0:
            self.active_frame_count += 1
        
        # หากตรวจพบการเปลี่ยนค่า Rep Count (จบครั้ง)
        if rep_count > self.last_rep_count:
            # คำนวณระยะเวลา (วินาที) จากจำนวนเฟรมที่ใช้ทำท่า
            duration = self.active_frame_count / self.fps
            if duration <= 0:
                duration = 1.5  # ค่าดีฟอลต์กรณีเฟรมผิดพลาด
                
            self.rep_durations.append(duration)
            
            # เช็คจังหวะความเร็ว (Squat/Push-up < 1.3 วิ ถือว่าเร็วไป)
            is_good_rhythm = not (self.exercise_name in ['Squat_Correct', 'Squat_Incorrect', 'Pushup'] and duration < 1.3)
            # ถ้าท่าผิดฟอร์มด้วย ให้ถือว่าเป็น Rep ที่จังหวะ/ฟอร์มไม่สมบูรณ์
            if self.rep_was_incorrect:
                is_good_rhythm = False
                
            self.rep_status.append(is_good_rhythm)
            
            # รีเซ็ตตัวแปรสำหรับ Rep ถัดไป
            self.active_frame_count = 0
            self.rep_was_incorrect = False
            self.last_rep_count = rep_count

    def generate_dashboard(self, output_dir="results"):
        """พล็อตและบันทึก Dashboard วิเคราะห์ผลลัพธ์แบบครบวงจร"""
        if not self.frames:
            print("  ⚠️ [Analytics] No data logged to generate dashboard.")
            return None

        # สร้างโฟลเดอร์สำหรับผลลัพธ์หากยังไม่มี
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # เตรียมข้อมูลสำหรับพล็อต
        frame_idxs = [f['frame_idx'] for f in self.frames]
        angles = [f['angle'] for f in self.frames]
        states = [f['state'] for f in self.frames]
        
        # ตั้งค่าฟอนต์ภาษาไทยสำหรับ matplotlib
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Tahoma', 'Leelawadee UI', 'Arial', 'Microsoft Sans Serif']
        plt.rcParams['axes.unicode_minus'] = False # ป้องกันเครื่องหมายลบแสดงเพี้ยน

        # สร้างรูปภาพ Dashboard ขนาดใหญ่
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f"รายงานวิเคราะห์สถิติการออกกำลังกาย: ท่า {self.exercise_name}", fontsize=18, fontweight='bold', y=0.96)
        
        # ─── 1. TIMELINE OF ANGLE & POSTURE CORRECTNESS (ครึ่งบน) ───
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax1.plot(frame_idxs, angles, color='#2c3e50', linewidth=2, label='Joint Angle (องศา)')
        ax1.set_title('ไทม์ไลน์มุมข้อต่อและสถานะความถูกต้องของท่าทาง', fontsize=13, fontweight='bold')
        ax1.set_xlabel('เฟรม (Frame Index)', fontsize=10)
        ax1.set_ylabel('มุมข้อต่อ (องศา)', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # ระบายสีพื้นหลังตามสถานะเฟรม (เขียว = ถูกต้อง, แดง = ผิดฟอร์ม)
        # จัดกลุ่มเพื่อความเร็วในการวาด
        i = 0
        n = len(self.frames)
        while i < n:
            start_idx = frame_idxs[i]
            curr_state = states[i]
            
            # ค้นหาจุดสิ้นสุดของสถานะปัจจุบัน
            j = i
            while j < n and states[j] == curr_state:
                j += 1
            end_idx = frame_idxs[j-1]
            
            # วาดแถบสี
            if curr_state == 'Correct':
                ax1.axvspan(start_idx, end_idx, color='#2ecc71', alpha=0.15)
            elif curr_state == 'Incorrect':
                ax1.axvspan(start_idx, end_idx, color='#e74c3c', alpha=0.2)
            i = j
            
        # วาดเส้นแบ่งจุดนับ Rep (เมื่อค่าเปลี่ยน)
        rep_changes = []
        for idx in range(1, len(self.frames)):
            if self.frames[idx]['rep_count'] > self.frames[idx-1]['rep_count']:
                rep_changes.append(self.frames[idx]['frame_idx'])
                
        for rx in rep_changes:
            ax1.axvline(x=rx, color='#f39c12', linestyle=':', linewidth=1.5, alpha=0.8)
            
        # สร้าง Custom Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', alpha=0.3, label='ท่าทางถูกต้อง (Correct)'),
            Patch(facecolor='#e74c3c', alpha=0.3, label='ท่าทางผิดฟอร์ม (Incorrect)'),
            Patch(facecolor='none', edgecolor='#f39c12', linestyle=':', label='จังหวะจบ Rep'),
        ]
        ax1.legend(handles=legend_elements + [ax1.get_lines()[0]], loc='upper right')

        # ─── 2. REP RHYTHM & TEMPO ANALYSIS (ล่างซ้าย) ───
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        if self.rep_durations:
            x_reps = [f"Rep {i+1}" for i in range(len(self.rep_durations))]
            colors = ['#2ecc71' if ok else '#e67e22' for ok in self.rep_status]
            bars = ax2.bar(x_reps, self.rep_durations, color=colors, edgecolor='#7f8c8d', width=0.6)
            
            ax2.axhline(y=1.3, color='#c0392b', linestyle='--', linewidth=1.2, label='เกณฑ์ความเร็วขั้นต่ำ (1.3s)')
            ax2.set_title('การวิเคราะห์จังหวะความเร็วในแต่ละครั้ง (Tempo)', fontsize=13, fontweight='bold')
            ax2.set_xlabel('ครั้งที่ทำ (Reps)', fontsize=10)
            ax2.set_ylabel('ระยะเวลา (วินาที)', fontsize=10)
            
            # ใส่ป้ายกำกับค่าวิบนบาร์
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}s',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # offset 3 points vertical
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'ไม่พบข้อมูลครั้งที่ทำเสร็จสมบูรณ์\n(ต้องนับได้อย่างน้อย 1 ครั้ง)', 
                     ha='center', va='center', fontsize=12, color='gray')
            ax2.set_title('การวิเคราะห์จังหวะความเร็วในแต่ละครั้ง (Tempo)', fontsize=13, fontweight='bold')

        # ─── 3. POSTURE QUALITY DISTRIBUTION (ล่างขวา) ───
        ax3 = plt.subplot2grid((2, 2), (1, 1))
        state_counts = {'Correct': 0, 'Incorrect': 0, 'Idle': 0}
        for f in self.frames:
            state_counts[f['state']] += 1
            
        labels = ['ท่าถูกต้อง', 'ท่าผิดฟอร์ม', 'ท่าพัก/Idle']
        sizes = [state_counts['Correct'], state_counts['Incorrect'], state_counts['Idle']]
        colors = ['#2ecc71', '#e74c3c', '#bdc3c7']
        
        # กรองรายการที่ไม่มีข้อมูลเพื่อความสวยงาม
        filtered_labels = []
        filtered_sizes = []
        filtered_colors = []
        for l, s, c in zip(labels, sizes, colors):
            if s > 0:
                filtered_labels.append(l)
                filtered_sizes.append(s)
                filtered_colors.append(c)
                
        if filtered_sizes:
            ax3.pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors,
                    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11},
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True})
            ax3.set_title('สัดส่วนคุณภาพของสรีระตลอดช่วงเวลา (Posture Quality)', fontsize=13, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'ไม่มีข้อมูลสำหรับแสดงสัดส่วน', ha='center', va='center')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # บันทึกรูปภาพสรุป
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"analytics_{self.exercise_name}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        plt.savefig(filepath, dpi=120)
        plt.close()
        
        print(f"\n  📊 [Analytics] เซฟกราฟสรุปสถิติลงแฟ้มผลลัพธ์สำเร็จ:")
        print(f"     ➡️ {os.path.abspath(filepath)}")
        
        # เปิดรูปภาพอัตโนมัติบนระบบปฏิบัติการ
        try:
            if os.name == 'nt':
                os.startfile(filepath)
            elif sys.platform == 'darwin':
                subprocess.run(['open', filepath])
            else:
                subprocess.run(['xdg-open', filepath])
        except Exception as e:
            print(f"  ⚠️ ไม่สามารถเปิดแสดงผลรูปภาพอัตโนมัติ: {e}")
            
        return filepath

    def export_to_csv(self, output_dir="results"):
        """บันทึกข้อมูลดิบรายเฟรมลงไฟล์ CSV ในโฟลเดอร์ที่กำหนด"""
        if not self.frames:
            print("  ⚠️ [Analytics] No data logged to export CSV.")
            return None

        # สร้างโฟลเดอร์สำหรับผลลัพธ์หากยังไม่มี
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        import csv
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"raw_data_{self.exercise_name}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(['frame_idx', 'angle', 'rep_count', 'state', 'confidence', 'predicted_class'])
                # Write rows
                for r in self.frames:
                    writer.writerow([
                        r['frame_idx'],
                        r['angle'],
                        r['rep_count'],
                        r['state'],
                        r['confidence'],
                        r['predicted_class']
                    ])
            
            print(f"  📝 [Analytics] เซฟข้อมูลดิบลง CSV สำเร็จ:")
            print(f"     ➡️ {os.path.abspath(filepath)}")
            return filepath
        except Exception as e:
            print(f"  ⚠️ [Analytics] ไม่สามารถบันทึกข้อมูล CSV: {e}")
            return None

