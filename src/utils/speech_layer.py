# -*- coding: utf-8 -*-
import pyttsx3
import threading
import queue
import time
import os
import sys

# Configure stdout for UTF-8 Emojis on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class SpeechAssistant:
    def __init__(self, cooldown_seconds=4.0):
        self.cooldown = cooldown_seconds
        self.last_spoken_time = 0.0
        self.last_spoken_text = ""
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        self.muted = False
        self.lock = threading.Lock()
        
        # เริ่มต้น worker thread
        self.start()

    def start(self):
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.worker_thread.start()

    def stop(self):
        self.running = False
        self.speech_queue.put(None)  # Sentinel to stop worker
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

    def _speech_worker(self):
        """Worker thread running the TTS engine entirely in the background"""
        try:
            # Initialize engine inside the background thread (pyttsx3 is not thread-safe)
            engine = pyttsx3.init()
            
            # ค้นหาฟอนต์/เสียงภาษาไทย
            voices = engine.getProperty('voices')
            thai_voice_id = None
            for voice in voices:
                # ตรวจสอบว่าเสียงเป็นภาษาไทยหรือไม่ (เช่นมีคำว่า Thai, Apasara, Pattara)
                if any(kw in voice.name.lower() or kw in voice.id.lower() for kw in ['thai', 'apasara', 'pattara', 'th-th']):
                    thai_voice_id = voice.id
                    break
            
            if thai_voice_id:
                engine.setProperty('voice', thai_voice_id)
            
            # ปรับความเร็วในการพูด (ช้าลงนิดนึงเพื่อให้อ่านข้อความยาวๆ ชัดเจน)
            engine.setProperty('rate', 150)
            
        except Exception as e:
            print(f"  ⚠️ [SpeechAssistant Worker Init Error]: {e}")
            return

        while self.running:
            try:
                text = self.speech_queue.get(timeout=0.5)
                if text is None:
                    break
                
                # พูดข้อความแบบบล็อก (เฉพาะใน thread นี้)
                engine.say(text)
                engine.runAndWait()
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # จัดการเผื่อมีการปิดหน้าต่างหรือเสียงขัดข้อง
                print(f"  ⚠️ [SpeechAssistant Speak Error]: {e}")
                # ลอง Re-init engine ใหม่
                try:
                    engine = pyttsx3.init()
                    if thai_voice_id:
                        engine.setProperty('voice', thai_voice_id)
                    engine.setProperty('rate', 150)
                except:
                    pass

    def speak(self, text):
        """ส่งข้อความพูด โดยเช็ค Cooldown ป้องกันเสียงพูดซ้อน"""
        if not text or not self.running or self.muted:
            return
        
        # ตัดแท็กพิเศษ เช่น [!], [แนะนำ] ออก เพื่อให้อ่านออกเสียงเฉพาะคำพูดภาษาไทยสะอาดๆ
        clean_text = text.replace("[!]", "").replace("[แนะนำ]", "").replace("💡", "").replace("⚠", "").strip()
        if not clean_text:
            return

        now = time.time()
        with self.lock:
            # ถ้าเป็นคำเดิม และยังไม่พ้น Cooldown หรือคำใหม่แต่ห่างจากคำพูดก่อนหน้าน้อยกว่า Cooldown
            if (now - self.last_spoken_time < self.cooldown) and (clean_text == self.last_spoken_text):
                return
            
            # เกรด Cooldown สั้นลงสำหรับคำสั่งแนะนำที่เปลี่ยนไป
            if (now - self.last_spoken_time < 2.5) and (clean_text != self.last_spoken_text):
                return
                
            self.last_spoken_time = now
            self.last_spoken_text = clean_text
            
        # ใส่คิวเข้าไปพูด
        self.speech_queue.put(clean_text)

# Global singleton instance
_assistant = None
_assistant_lock = threading.Lock()

def get_speech_assistant(cooldown=4.0):
    global _assistant
    with _assistant_lock:
        if _assistant is None:
            _assistant = SpeechAssistant(cooldown_seconds=cooldown)
        return _assistant

def speak_feedback(text):
    """ฟังก์ชันหลักสำหรับเรียกใช้งานด่วน"""
    try:
        assistant = get_speech_assistant()
        assistant.speak(text)
    except Exception as e:
        try:
            print(f"  ⚠️ [speak_feedback error]: {e}")
        except UnicodeEncodeError:
            print(f"  [speak_feedback error]: {e}")

def toggle_speech_mute():
    """สลับสถานะเปิด/ปิดเสียงพูดเตือน"""
    try:
        assistant = get_speech_assistant()
        assistant.muted = not assistant.muted
        status = "Muted (ปิดเสียงพูด)" if assistant.muted else "Unmuted (เปิดเสียงพูด)"
        try:
            print(f"  🔊 [SpeechAssistant] {status}")
        except UnicodeEncodeError:
            print(f"  [SpeechAssistant] {status}")
        return assistant.muted
    except Exception as e:
        try:
            print(f"  ⚠️ [toggle_speech_mute error]: {e}")
        except UnicodeEncodeError:
            print(f"  [toggle_speech_mute error]: {e}")
        return False

def is_speech_muted():
    """เช็คสถานะว่าเปิด/ปิดเสียงอยู่หรือไม่"""
    try:
        return get_speech_assistant().muted
    except:
        return False
