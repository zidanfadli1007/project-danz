import cv2
import mediapipe as mp
import time
import numpy as np
import winsound
import threading
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def play_wav(filename):
    path = os.path.join(BASE_DIR, filename)
    def _run():
        winsound.PlaySound(path, winsound.SND_FILENAME)
    threading.Thread(target=_run, daemon=True).start()

# ========== MEDIAPIPE ==========
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Buka kamera
CAMERA_INDEX = 1
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(1)  # kasih waktu Iriun stabil

ret = False
for attempt in range(10):
    ret, frame = cap.read()
    if ret:
        break
    time.sleep(0.3)

if not ret:
    print("Error: Kamera tidak terdeteksi! Pastikan Iriun sudah running & HP terkoneksi.")
    exit()

print(f"✅ Kamera Iriun (index {CAMERA_INDEX}) terkoneksi - resolusi {frame.shape}")

# ========== KOTAK ==========
# Kotak dalam (tempat barang)
inner_x, inner_y, inner_w, inner_h = 200, 200, 150, 150

# Kotak luar (zona pendekatan)
margin = 80
outer_x = inner_x - margin
outer_y = inner_y - margin
outer_w = inner_w + (margin * 2)
outer_h = inner_h + (margin * 2)

# Background
background = None

# State
stolen = False
last_alert = 0

# Threshold
STOLEN_PCT = 25

# ========== FITUR BARU ==========
show_boxes = True   # Default: kotak muncul
# ================================

def calculate_change_pct(bg, curr):
    diff = cv2.absdiff(bg, curr)
    changed = np.count_nonzero(diff > 30)
    total = bg.shape[0] * bg.shape[1]
    return (changed / total) * 100

print("\n" + "="*70)
print("   SISTEM 2 KOTAK + MEDIAPIPE (DETEKSI TANGAN)")
print("="*70)
print("\n🎮 KONTROL:")
print("   W/A/S/D : Geser kotak DALAM")
print("   +/-     : Perbesar/kecil kotak DALAM")
print("   SPACE   : Rekam background & MULAI")
print("   h       : Sembunyikan/Tampilkan kotak (HIDE BOX)")
print("   q       : Keluar")
print("="*70)

# Mode setup
setup_mode = True
while setup_mode:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Update kotak luar
    outer_x = max(0, inner_x - margin)
    outer_y = max(0, inner_y - margin)
    outer_w = inner_w + (margin * 2)
    outer_h = inner_h + (margin * 2)
    
    # Gambar kotak (selalu muncul saat setup)
    cv2.rectangle(frame, (outer_x, outer_y), (outer_x+outer_w, outer_y+outer_h), (0, 255, 255), 2)
    cv2.rectangle(frame, (inner_x, inner_y), (inner_x+inner_w, inner_y+inner_h), (0, 255, 0), 3)
    
    cv2.putText(frame, "ZONA PENDEKATAN", (outer_x, outer_y-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(frame, "TEMPAT BARANG", (inner_x, inner_y-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, "POSISIKAN BARANG DI KOTAK HIJAU", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "W/A/S/D=Geser  +/-=Ukuran  SPACE=Mulai  h=Sembunyikan", 
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Setup - MediaPipe 2 Kotak", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        area = frame[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w]
        background = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        setup_mode = False
        print("\n✅ BACKGROUND DIREKAM!")
        print("🚀 MEDIAPIPE AKTIF - DETEKSI TANGAN & BARANG")
        print("💡 Tekan 'h' untuk sembunyikan/tampilkan kotak\n")
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
    elif key == ord('w'): inner_y -= 10
    elif key == ord('s'): inner_y += 10
    elif key == ord('a'): inner_x -= 10
    elif key == ord('d'): inner_x += 10
    elif key == ord('+') or key == ord('='):
        inner_w += 10
        inner_h += 10
    elif key == ord('-') or key == ord('_'):
        inner_w = max(80, inner_w - 10)
        inner_h = max(80, inner_h - 10)
    
    inner_x = max(0, min(inner_x, frame.shape[1] - inner_w))
    inner_y = max(0, min(inner_y, frame.shape[0] - inner_h))

print("🎯 ALARM AKTIF | q: Keluar | r: Reset | b: Rekam BG | h: Hide/Show Box")
print("="*60)

# ========== LOOP UTAMA DENGAN MEDIAPIPE ==========
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Update kotak luar
    outer_x = max(0, inner_x - margin)
    outer_y = max(0, inner_y - margin)
    outer_w = inner_w + (margin * 2)
    outer_h = inner_h + (margin * 2)
    
    # ========== MEDIAPIPE: DETEKSI TANGAN ==========
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil = hands.process(rgb)
    
    finger_tip_pos = None
    hand_detected = False
    finger_in_inner = False
    finger_in_outer = False
    
    if hasil.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in hasil.multi_hand_landmarks:
            # Gambar landmark tangan (21 titik)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Ambil ujung jari telunjuk (landmark 8)
            h, w = frame.shape[:2]
            fx = int(hand_landmarks.landmark[8].x * w)
            fy = int(hand_landmarks.landmark[8].y * h)
            finger_tip_pos = (fx, fy)
            
            # Gambar lingkaran di ujung jari
            cv2.circle(frame, (fx, fy), 12, (0, 255, 255), -1)
            
            # Cek posisi jari
            if (inner_x < fx < inner_x + inner_w and 
                inner_y < fy < inner_y + inner_h):
                finger_in_inner = True
                cv2.circle(frame, (fx, fy), 18, (0, 0, 255), 3)
            elif (outer_x < fx < outer_x + outer_w and 
                  outer_y < fy < outer_y + outer_h):
                finger_in_outer = True
                cv2.circle(frame, (fx, fy), 18, (0, 255, 0), 2)
    
    # ========== DETEKSI PERUBAHAN BARANG ==========
    curr_area = frame[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w]
    curr_gray = cv2.cvtColor(curr_area, cv2.COLOR_BGR2GRAY)
    
    if background is not None:
        change_pct = calculate_change_pct(background, curr_gray)
    else:
        change_pct = 0
    
    # ========== LOGIKA ALARM ==========
    now = time.time()
    
    # Prioritaskan deteksi tangan dulu
    if finger_in_inner:
        inner_color = (0, 0, 255)
        outer_color = (0, 0, 255)
        status = "TANGAN MAU AMBIL BARANG!"
        if (now - last_alert) > 2:
            play_wav("barang telah dicuri.wav")
            last_alert = now
    elif finger_in_outer:
        inner_color = (0, 255, 255)
        outer_color = (0, 255, 255)
        status = "ADA YANG MENDEKAT!"
        if (now - last_alert) > 3:
            play_wav("ada orang yang mendekati barangmu.wav")
            last_alert = now
    elif change_pct > STOLEN_PCT:
        inner_color = (0, 0, 255)
        outer_color = (0, 0, 255)
        status = f"BARANG HILANG! ({change_pct:.0f}%)"
        if not stolen and (now - last_alert) > 2:
            play_wav("barang telah dicuri.wav")
            stolen = True
            last_alert = now
    else:
        inner_color = (0, 255, 0)
        outer_color = (0, 255, 255)
        status = f"✅ AMAN ({change_pct:.0f}%)"
    
    # Reset stolen
    if stolen and change_pct < 20:
        stolen = False
    
    # ========== TAMPILAN ==========
    # HANYA GAMBAR KOTAK JIKA show_boxes = True
    if show_boxes:
        # Kotak luar
        cv2.rectangle(frame, (outer_x, outer_y), (outer_x+outer_w, outer_y+outer_h), outer_color, 2)
        cv2.putText(frame, "ZONA PENDEKATAN", (outer_x, outer_y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, outer_color, 1)
        
        # Kotak dalam
        cv2.rectangle(frame, (inner_x, inner_y), (inner_x+inner_w, inner_y+inner_h), inner_color, 3)
        cv2.putText(frame, "TEMPAT BARANG", (inner_x, inner_y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, inner_color, 1)
    
    # Status (tetap muncul walau kotak disembunyikan)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, inner_color, 2)
    cv2.putText(frame, f"Perubahan: {change_pct:.1f}%", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Info MediaPipe
    if hand_detected:
        cv2.putText(frame, "MEDIAPIPE: TANGAN TERDETEKSI", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "MEDIAPIPE: TIDAK ADA TANGAN", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    if finger_tip_pos:
        cv2.putText(frame, f"Jari: ({finger_tip_pos[0]}, {finger_tip_pos[1]})", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Progress bar
    bar_len = min(200, int(change_pct * 2))
    cv2.rectangle(frame, (10, frame.shape[0]-30), (10+bar_len, frame.shape[0]-10), inner_color, -1)
    
    # Indikator show/hide
    if not show_boxes:
        cv2.putText(frame, "KOTAK: HIDDEN (Tekan 'h' untuk munculkan)", (10, frame.shape[0]-40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    cv2.imshow("MediaPipe - Alarm 2 Kotak", frame)
    
    # Keyboard control
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        stolen = False
        print("🔄 Alarm di-reset")
    elif key == ord('b'):
        area = frame[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w]
        background = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        stolen = False
        print("✅ Background direkam ulang!")
    elif key == ord('h'):
        show_boxes = not show_boxes
        if show_boxes:
            print("📦 Kotak ditampilkan")
        else:
            print("👻 Kotak disembunyikan (real mode)")

cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n👋 Program selesai!")