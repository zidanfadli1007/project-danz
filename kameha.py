import cv2
import time
import os
import math
import random
import threading
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL = "hand_landmarker.task"

# SUMBER = "http://192.168.1.11:8080/video"   # ganti IP sesuai yng ada di HP
SUMBER = 0                                # webcam / Iriun / DroidCam dll

LEBAR, TINGGI = 960, 540
FOLDER = "foto"

TAHAN_FOTO = 3.0 
TAHAN_MODE = 2.0
HALUS = 0.40
MIN_BUKA = 120
GOYANG = 9

MULAI_BUTUH = 2       
HENTI_BUTUH = 4       
HALUS_PENA = 0.5      
LOMPAT_MAKS = 160     

AMBANG_JARI = 1.12
AMBANG_JEMPOL = 1.15

KAME_CHARGE_TIME = 1.3     # detik nahan telapak biar full charge
KAME_DECAY = 0.9           # kecepatan charge turun kalau tangan dilepas
KAME_DURASI_LEDAK = 0.8    # lama animasi ledakan (detik)

FONT = cv2.FONT_HERSHEY_DUPLEX
CYAN = (255, 235, 0)
MAGENTA = (200, 0, 255)
AMBER = (0, 180, 255)
PUTIH = (240, 245, 250)
ABU = (120, 110, 130)
SAMAR = (55, 50, 62)
BIRU_ENERGI = (255, 210, 120)

os.makedirs(FOLDER, exist_ok=True)

def _lut_retro():
    x = np.arange(256, dtype=np.float32)
    b = np.clip(18 + x * 0.92, 0, 255).astype(np.uint8)
    g = np.clip(6 + x * 0.97, 0, 255).astype(np.uint8)
    r = np.clip(x * 1.06 - 4, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r]).reshape(1, 256, 3)


LUT_RETRO = _lut_retro()


def _vignette(w, h):
    kx = cv2.getGaussianKernel(w, w * 0.55)
    ky = cv2.getGaussianKernel(h, h * 0.55)
    m = ky @ kx.T
    m = 0.35 + 0.65 * (m / m.max())
    return np.clip(m * 255, 0, 255).astype(np.uint8)[:, :, None].repeat(3, axis=2)


VIGNETTE = _vignette(LEBAR, TINGGI)


def grade_retro(img):
    out = cv2.LUT(img, LUT_RETRO)
    out = cv2.multiply(out, VIGNETTE, scale=1 / 255)
    out[::3] = (out[::3] * 0.78).astype(np.uint8)
    return out


BAYER8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float32) * (255.0 / 64.0)

CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
LUT_KONTRAS = np.clip(((np.arange(256) / 255.0) ** 1.6) * 300 - 22,
                      0, 255).astype(np.uint8)
_grain = {}


def abu(roi):
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


def ke_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def m_mono(roi, t):
    return ke_bgr(CLAHE.apply(abu(roi)))


def m_kontras(roi, t):
    return ke_bgr(cv2.LUT(CLAHE.apply(abu(roi)), LUT_KONTRAS))


def m_film(roi, t):
    g = CLAHE.apply(abu(roi))
    h, w = g.shape
    if (h, w) not in _grain:
        rng = np.random.default_rng(7)
        _grain[(h, w)] = [rng.normal(0, 11, (h, w)).astype(np.int16)
                          for _ in range(6)]
    g = np.clip(g.astype(np.int16) + _grain[(h, w)][int(t * 18) % 6],
                0, 255).astype(np.uint8)
    bloom = cv2.GaussianBlur(cv2.threshold(g, 195, 255, cv2.THRESH_TOZERO)[1],
                             (0, 0), 6)
    return ke_bgr(cv2.addWeighted(g, 1.0, bloom, 0.35, 0))


def m_garis(roi, t):
    g = cv2.GaussianBlur(abu(roi), (0, 0), 1.2)
    e = cv2.dilate(cv2.Canny(g, 45, 130), np.ones((2, 2), np.uint8))
    return ke_bgr(cv2.add(cv2.multiply(g, 0.14, dtype=cv2.CV_8U), e))


def m_ambang(roi, t):
    g = cv2.GaussianBlur(CLAHE.apply(abu(roi)), (0, 0), 1.0)
    return ke_bgr(cv2.threshold(g, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])


def m_dither(roi, t):
    g = CLAHE.apply(abu(roi))
    h, w = g.shape
    ubin = np.tile(BAYER8, (h // 8 + 1, w // 8 + 1))[:h, :w]
    return ke_bgr(np.where(g.astype(np.float32) > ubin, 255, 0).astype(np.uint8))


def m_negatif(roi, t):
    return ke_bgr(cv2.bitwise_not(CLAHE.apply(abu(roi))))


LENSA = [("MONO", m_mono), ("KONTRAS", m_kontras), ("FILM", m_film),
         ("GARIS", m_garis), ("AMBANG", m_ambang), ("DITHER", m_dither),
         ("NEGATIF", m_negatif)]

def urutkan_quad(titik):
    p = np.array(titik, dtype=np.float32)
    c = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    return np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)


def cocokkan(q_baru, q_lama):
    if q_lama is None:
        return q_baru
    terbaik, skor_min = 0, None
    for r in range(4):
        skor = float(np.sum((np.roll(q_baru, -r, axis=0) - q_lama) ** 2))
        if skor_min is None or skor < skor_min:
            skor_min, terbaik = skor, r
    return np.roll(q_baru, -terbaik, axis=0)


def sisi(q):
    return (np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1]),
            np.linalg.norm(q[2] - q[3]), np.linalg.norm(q[3] - q[0]))


def orientasi(q):
    atas, kanan, bawah, kiri = sisi(q)
    v = q[1] - q[0]
    return (math.degrees(math.atan2(v[1], v[0])),
            math.degrees(math.atan2(bawah - atas, bawah + atas + 1e-6)) * 2,
            math.degrees(math.atan2(kanan - kiri, kanan + kiri + 1e-6)) * 2)


def warp_efek(frame, q, fn, t):
    Hf, Wf = frame.shape[:2]
    atas, kanan, bawah, kiri = sisi(q)
    lw, lh = int(max(atas, bawah)), int(max(kiri, kanan))
    if lw < 16 or lh < 16:
        return None, None, None
    tujuan = np.float32([[0, 0], [lw - 1, 0], [lw - 1, lh - 1], [0, lh - 1]])
    rect = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(q, tujuan),
                               (lw, lh))
    efek = fn(rect, t)

    bx1 = max(0, int(np.floor(q[:, 0].min())))
    by1 = max(0, int(np.floor(q[:, 1].min())))
    bx2 = min(Wf, int(np.ceil(q[:, 0].max())) + 1)
    by2 = min(Hf, int(np.ceil(q[:, 1].max())) + 1)
    if bx2 - bx1 < 4 or by2 - by1 < 4:
        return None, None, None

    q_lokal = q - np.float32([bx1, by1])
    balik = cv2.warpPerspective(efek, cv2.getPerspectiveTransform(tujuan, q_lokal),
                                (bx2 - bx1, by2 - by1))
    return efek, balik, (bx1, by1, bx2, by2, q_lokal)


def komposit(tampil, balik, meta):
    bx1, by1, bx2, by2, q_lokal = meta
    mask = np.zeros((by2 - by1, bx2 - bx1), np.uint8)
    cv2.fillConvexPoly(mask, q_lokal.astype(np.int32), 255)
    cv2.copyTo(balik, mask, tampil[by1:by2, bx1:bx2])
    return tampil


def teks(img, s, org, skala, tebal=2, warna=PUTIH, aberasi=True):
    x, y = org
    if aberasi:
        cv2.putText(img, s, (x - 2, y), FONT, skala, MAGENTA, tebal, cv2.LINE_AA)
        cv2.putText(img, s, (x + 2, y), FONT, skala, CYAN, tebal, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, skala, warna, tebal, cv2.LINE_AA)


def titik_int(p):
    return (int(round(p[0])), int(round(p[1])))


def kurung_quad(img, q, warna, tebal=3):
    for i in range(4):
        p = q[i]
        for j in ((i - 1) % 4, (i + 1) % 4):
            v = q[j] - p
            L = float(np.linalg.norm(v))
            if L < 2:
                continue
            cv2.line(img, titik_int(p), titik_int(p + v / L * min(L * 0.28, 55)),
                     warna, tebal, cv2.LINE_AA)


def cincin(img, pusat, radius, maju, warna, tebal=4):
    cv2.ellipse(img, pusat, (radius, radius), -90, 0, 360, SAMAR, tebal)
    if maju > 0:
        cv2.ellipse(img, pusat, (radius, radius), -90, 0,
                    int(360 * min(1.0, maju)), warna, tebal, cv2.LINE_AA)


def jarak(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def jari_terbuka(hand):
    """[jempol, telunjuk, tengah, manis, kelingking]"""
    w, ref = hand[0], hand[17]
    out = [jarak(hand[4], ref) > jarak(hand[2], ref) * AMBANG_JEMPOL]
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        out.append(jarak(hand[tip], w) > jarak(hand[pip], w) * AMBANG_JARI)
    return out


def is_L(f):
    return f[0] and f[1] and not f[2] and not f[3] and not f[4]


def is_tunjuk(f):
    """Menunjuk - jempol DIABAIKAN. Dipakai saat menggambar."""
    return f[1] and not f[2] and not f[3] and not f[4]


def is_tunjuk_ketat(f):
    """Menunjuk dengan jempol ditekuk. Dipakai agar tidak bentrok dengan L."""
    return is_tunjuk(f) and not f[0]


def is_peace(f):
    return f[1] and f[2] and not f[3] and not f[4]


def is_kepal(f):
    return not any(f)


def is_telapak(f):
    return all(f)


def jarak_px(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def pusat_dan_ukuran_telapak(px):
    """Titik tengah telapak (antara pergelangan & pangkal jari tengah) + ukurannya."""
    pusat = ((px[0][0] + px[9][0]) / 2, (px[0][1] + px[9][1]) / 2)
    ukuran = jarak_px(px[0], px[9])
    return pusat, ukuran


def is_telapak_kame(f):
    """Versi longgar dari is_telapak, khusus buat charge kamehameha.
    Cukup minimal 4 dari 5 jari kebaca terbuka, jadi tetap kedeteksi
    walau tangan agak miring / dari samping / jempol ketutup dikit."""
    return sum(f) >= 4


def gambar_bola_energi(tampil, pusat, progress, siap):
    p = titik_int(pusat)
    warna = PUTIH if siap else BIRU_ENERGI
    r_inti = max(3, int(14 + 46 * progress))

    lapisan = np.zeros_like(tampil)
    cv2.circle(lapisan, p, r_inti, warna, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(lapisan, p, int(r_inti * 1.7), warna, 2, cv2.LINE_AA)
    lapisan = cv2.GaussianBlur(lapisan, (0, 0), 10 + 12 * progress)
    tampil[:] = cv2.add(tampil, lapisan)

    for i in range(8):
        a = time.time() * (3 + 3 * progress) + i * (math.pi / 4)
        rr = r_inti + 14
        tp = (int(p[0] + rr * math.cos(a)), int(p[1] + rr * math.sin(a)))
        cv2.circle(tampil, tp, 2, warna, cv2.FILLED, cv2.LINE_AA)

    if siap:
        r_pulsa = int(r_inti * 1.9 + 6 * math.sin(time.time() * 8))
        cv2.circle(tampil, p, r_pulsa, PUTIH, 2, cv2.LINE_AA)


def gambar_bar_isi(tampil, progress):
    """Bar 'MENGISI XX%' di tengah atas layar, kayak referensi. Selalu keliatan
    jelas walau tangan lagi di posisi/sudut manapun."""
    w = tampil.shape[1]
    bar_w, bar_h = int(w * 0.46), 24
    x0 = (w - bar_w) // 2
    y0 = 66

    warna = BIRU_ENERGI
    label = f"MENGISI {int(progress * 100)}%"
    (tw, _), _ = cv2.getTextSize(label, FONT, 0.65, 2)
    teks(tampil, label, (x0 + bar_w // 2 - tw // 2, y0 - 14), 0.65, 2, warna)

    cv2.rectangle(tampil, (x0, y0), (x0 + bar_w, y0 + bar_h), SAMAR, 2, cv2.LINE_AA)
    isi_w = max(4, int(bar_w * progress))
    cv2.rectangle(tampil, (x0 + 2, y0 + 2), (x0 + isi_w - 2, y0 + bar_h - 2),
                  warna, cv2.FILLED, cv2.LINE_AA)


def terapkan_getaran(img, intensitas):
    """Guncang layar dikit-dikit, makin gede intensitas (0..1) makin kuat guncangannya."""
    if intensitas <= 0.02:
        return img
    mag = 16 * intensitas
    dx = random.uniform(-mag, mag)
    dy = random.uniform(-mag, mag)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          borderMode=cv2.BORDER_REFLECT)


def gambar_ledakan(tampil, pusat, progress):
    p = titik_int(pusat)
    h, w = tampil.shape[:2]
    r_maks = int(math.hypot(w, h))
    r = int(r_maks * min(1.0, progress * 1.3))

    lapisan = np.zeros_like(tampil)
    cv2.circle(lapisan, p, int(60 + 200 * progress), PUTIH, cv2.FILLED, cv2.LINE_AA)
    lapisan = cv2.GaussianBlur(lapisan, (0, 0), 20)
    tampil[:] = cv2.add(tampil, lapisan)

    for i in range(3):
        rr = int(r * (0.5 + i * 0.3)) - i * 40
        if rr > 2:
            cv2.circle(tampil, p, rr, BIRU_ENERGI, 3, cv2.LINE_AA)

    for i in range(14):
        a = i * (2 * math.pi / 14) + progress * 2
        panjang = r * random.uniform(0.6, 1.0)
        ujung = (int(p[0] + panjang * math.cos(a)), int(p[1] + panjang * math.sin(a)))
        cv2.line(tampil, p, ujung, PUTIH, 2, cv2.LINE_AA)


class KameraStream:
    def __init__(self, sumber):
        if isinstance(sumber, int):
            self.cap = cv2.VideoCapture(sumber, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(sumber)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.jalan = True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.jalan:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = f

    def baca(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.jalan = False
        time.sleep(0.15)
        self.cap.release()

opsi = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)
landmarker = vision.HandLandmarker.create_from_options(opsi)

kamera = KameraStream(SUMBER)
time.sleep(1.0)
if not kamera.cap.isOpened():
    print("Kamera gagal dibuka. Cek IP / kabel / WiFi.")
    raise SystemExit

cv2.namedWindow("RETROLENS", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RETROLENS", LEBAR, TINGGI)
KONEKSI = vision.HandLandmarksConnections.HAND_CONNECTIONS

mode = "LENSA"
idx_lensa = 0
quad = None
hilang = 99
mulai_diam = None
mulai_ganti = None
kilat_sampai = 0.0
jml_foto = 0
prev_time = 0.0
ts = 0
debug = False

kanvas = np.zeros((TINGGI, LEBAR, 3), np.uint8)
pena_akhir = None
pena_halus = None
menggambar = False
beruntun_ya = beruntun_tidak = 0
goresan = 0
PENA = [PUTIH, CYAN, MAGENTA, AMBER]
idx_pena = 1

MODE_URUT = ["LENSA", "GAMBAR"]
kame_charge = 0.0
kame_ledak_sampai = 0.0
kame_pusat_ledak = (LEBAR // 2, TINGGI // 2)
kame_waktu_prev = None


def simpan(gambar, tag="LENS"):
    global jml_foto, kilat_sampai
    nama = os.path.join(FOLDER, time.strftime(f"{tag}_%Y%m%d_%H%M%S.png"))
    cv2.imwrite(nama, gambar)
    print("Tersimpan:", nama, f"{gambar.shape[1]}x{gambar.shape[0]}")
    jml_foto += 1
    kilat_sampai = time.time() + 0.30

while True:
    frame = kamera.baca()
    if frame is None:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (LEBAR, TINGGI))
    bersih = frame                       
    w, h = LEBAR, TINGGI
    now = time.time()

    rgb = cv2.cvtColor(bersih, cv2.COLOR_BGR2RGB)
    ts += 1
    hasil = landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts * 33)
    tangan = hasil.hand_landmarks

    info = []
    for hand in tangan:
        f = jari_terbuka(hand)
        px = [(int(l.x * w), int(l.y * h)) for l in hand]
        info.append((hand, f, px))

    nama_lensa, fn_lensa = LENSA[idx_lensa]
    status = "-"

    if mode == "LENSA":
        sudut = []
        for hand, f, px in info:
            if is_L(f):
                sudut.append(px[4])
                sudut.append(px[8])

        mentah = cocokkan(urutkan_quad(sudut[:4]), quad) if len(sudut) >= 4 else None

        geser_maks = 0.0
        if mentah is not None:
            if quad is None:
                quad = mentah
            else:
                geser_maks = float(np.max(np.linalg.norm(mentah - quad, axis=1)))
                quad = quad + HALUS * (mentah - quad)
            hilang = 0
        else:
            hilang += 1

        tampil = grade_retro(bersih)

        aktif = hilang < 6 and quad is not None
        if aktif:
            q = quad.astype(np.float32)
            diag = (np.linalg.norm(q[2] - q[0]) + np.linalg.norm(q[3] - q[1])) / 2

            if diag < MIN_BUKA:
                status = "GENGGAM"
                mulai_diam = None
                c = titik_int(q.mean(axis=0))
                r = int(18 + 6 * math.sin(now * 6))
                cv2.circle(tampil, c, r, CYAN, 2, cv2.LINE_AA)
                cv2.circle(tampil, c, 3, CYAN, cv2.FILLED)
                teks(tampil, "TARIK UNTUK MEMBUKA", (c[0] - 128, c[1] - 40), 0.6, 2)
            else:

                efek, balik, meta = warp_efek(bersih, q, fn_lensa, now)
                if efek is not None:
                    tampil = komposit(tampil, balik, meta)

                    if geser_maks > GOYANG or mulai_diam is None:
                        mulai_diam = now
                    sisa = TAHAN_FOTO - (now - mulai_diam)
                    status = "ATUR" if geser_maks > GOYANG else "DIAM"
                    warna = AMBER if sisa > 1 else MAGENTA

                    u = (now * 0.35) % 1.0
                    cv2.line(tampil, titik_int(q[0] + (q[3] - q[0]) * u),
                             titik_int(q[1] + (q[2] - q[1]) * u), CYAN, 1, cv2.LINE_AA)
                    cv2.polylines(tampil, [q.astype(np.int32)], True, SAMAR, 1,
                                  cv2.LINE_AA)
                    kurung_quad(tampil, q, warna)

                    n = q[0] + (q[1] - q[0]) * 0.02
                    teks(tampil, f"{nama_lensa}  {efek.shape[1]}x{efek.shape[0]}",
                         (int(n[0]), max(18, int(n[1]) - 12)), 0.5, 1, warna)

                    maju = min(1.0, max(0.0, 1 - sisa / TAHAN_FOTO))
                    a, b = q[3], q[2]
                    cv2.line(tampil, titik_int(a), titik_int(b), SAMAR, 4, cv2.LINE_AA)
                    cv2.line(tampil, titik_int(a), titik_int(a + (b - a) * maju),
                             warna, 4, cv2.LINE_AA)

                    if sisa <= 0:
                        simpan(efek, nama_lensa)
                        mulai_diam = None
                        hilang = 99
                        quad = None
                    elif sisa < TAHAN_FOTO - 0.25:
                        ang = str(int(math.ceil(sisa)))
                        sk = 2.4 + 0.4 * abs(math.sin(sisa * math.pi))
                        (tw, th), _ = cv2.getTextSize(ang, FONT, sk, 6)
                        c = q.mean(axis=0)
                        teks(tampil, ang, (int(c[0] - tw / 2), int(c[1] + th / 2)),
                             sk, 6, warna)
        else:
            mulai_diam = None
            if hilang > 20:
                quad = None

        pemicu = None
        for hand, f, px in info:
            if is_tunjuk_ketat(f):
                pemicu = px[8]
                break

        if pemicu and not aktif:
            if mulai_ganti is None:
                mulai_ganti = now
            maju = (now - mulai_ganti) / TAHAN_MODE
            cincin(tampil, pemicu, 26, maju, CYAN)
            teks(tampil, "MODE GAMBAR", (pemicu[0] - 68, pemicu[1] - 42), 0.55, 2)
            if maju >= 1.0:
                mode, mulai_ganti, quad = "GAMBAR", None, None
                pena_akhir = pena_halus = None
                menggambar = False
                beruntun_ya = beruntun_tidak = 0
        else:
            mulai_ganti = None

        for hand, f, px in info:
            ok = is_L(f)
            for c in KONEKSI:
                cv2.line(tampil, px[c.start], px[c.end],
                         (150, 120, 90) if ok else (95, 80, 105), 1, cv2.LINE_AA)
            for i in (4, 8):
                cv2.circle(tampil, px[i], 7, CYAN if ok else ABU,
                           2 if ok else 1, cv2.LINE_AA)

        if not tangan:
            teks(tampil, "BENTUK 'L' DUA TANGAN, SATUKAN, LALU TARIK",
                 (22, h - 58), 0.55, 2, ABU)

    elif mode == "GAMBAR":
        tampil = cv2.multiply(grade_retro(bersih), 0.5, dtype=cv2.CV_8U)
        status = "SIAP"

        keluar = None
        for hand, f, px in info:
            if is_telapak(f):
                keluar = px[9]
                break

        if keluar:
            if mulai_ganti is None:
                mulai_ganti = now
            maju = (now - mulai_ganti) / TAHAN_MODE
            cincin(tampil, keluar, 30, maju, AMBER)
            teks(tampil, "MODE LENSA", (keluar[0] - 62, keluar[1] - 46), 0.55, 2)
            status = "KELUAR"
            menggambar = False
            pena_akhir = pena_halus = None
            if maju >= 1.0:
                mode, mulai_ganti, quad, hilang = "LENSA", None, None, 99
        else:
            mulai_ganti = None

            hapus = False
            gambar_ok = False
            ujung = None
            if info:
                hand, f, px = info[0]
                ujung = px[8]
                if is_kepal(f):
                    hapus = True
                elif is_peace(f):
                    status = "PINDAH"
                elif is_tunjuk(f):
                    gambar_ok = True

            if hapus:
                kanvas[:] = 0
                goresan = 0
                menggambar = False
                pena_akhir = pena_halus = None
                beruntun_ya = beruntun_tidak = 0
                status = "HAPUS"
            else:
                if gambar_ok:
                    beruntun_ya += 1
                    beruntun_tidak = 0
                else:
                    beruntun_tidak += 1
                    beruntun_ya = 0

                if not menggambar and beruntun_ya >= MULAI_BUTUH:
                    menggambar = True
                elif menggambar and beruntun_tidak >= HENTI_BUTUH:
                    menggambar = False
                    pena_akhir = pena_halus = None

                if menggambar and gambar_ok and ujung is not None:
                    p = np.array(ujung, dtype=np.float32)
                    pena_halus = p if pena_halus is None else \
                        pena_halus + HALUS_PENA * (p - pena_halus)
                    titik = titik_int(pena_halus)
                    if pena_akhir is not None:
                        d = abs(titik[0] - pena_akhir[0]) + abs(titik[1] - pena_akhir[1])
                        if d < LOMPAT_MAKS:
                            cv2.line(kanvas, pena_akhir, titik, PENA[idx_pena],
                                     6, cv2.LINE_AA)
                            goresan += 1
                    pena_akhir = titik
                    status = "GAMBAR"
                    cv2.circle(tampil, titik, 12, PENA[idx_pena], 2, cv2.LINE_AA)
                elif ujung is not None:
                    cv2.circle(tampil, ujung, 9, ABU, 1, cv2.LINE_AA)

        for hand, f, px in info:
            for c in KONEKSI:
                cv2.line(tampil, px[c.start], px[c.end], (80, 70, 90), 1, cv2.LINE_AA)
            for i in (4, 8, 12, 16, 20):
                cv2.circle(tampil, px[i], 4, (60, 60, 235), cv2.FILLED)

        kecil = cv2.GaussianBlur(cv2.resize(kanvas, (w // 3, h // 3)), (0, 0), 4)
        tampil = cv2.add(tampil, cv2.resize(kecil, (w, h)))
        tampil = cv2.add(tampil, kanvas)
        teks(tampil, f"{status}   GORESAN {goresan}", (22, h - 58), 0.6, 2, ABU)

    # --- Overlay KAMEHAMEHA: otomatis aktif di semua mode, gak perlu pencet 'm' ---
    dt_k = max(0.0, now - kame_waktu_prev) if kame_waktu_prev is not None else 0.0
    kame_waktu_prev = now

    palm_list = []
    for hand, f, px in info:
        if is_telapak_kame(f):
            palm_list.append(pusat_dan_ukuran_telapak(px))

    if now >= kame_ledak_sampai:
        if palm_list:
            pusat = (sum(p[0][0] for p in palm_list) / len(palm_list),
                     sum(p[0][1] for p in palm_list) / len(palm_list))

            kame_charge = min(1.0, kame_charge + dt_k / KAME_CHARGE_TIME)

            if kame_charge >= 1.0:
                kame_ledak_sampai = now + KAME_DURASI_LEDAK
                kame_pusat_ledak = pusat
                kame_charge = 0.0
                kilat_sampai = now + 0.15
            else:
                gambar_bola_energi(tampil, pusat, kame_charge, False)
                gambar_bar_isi(tampil, kame_charge)
                tampil = terapkan_getaran(tampil, kame_charge)
        else:
            if kame_charge > 0.0:
                kame_charge = max(0.0, kame_charge - dt_k * KAME_DECAY)
    else:
        progress = 1 - (kame_ledak_sampai - now) / KAME_DURASI_LEDAK
        gambar_ledakan(tampil, kame_pusat_ledak, progress)

    if int(now * 2) % 2 == 0:
        cv2.circle(tampil, (28, 30), 8, (60, 60, 235), cv2.FILLED)
    teks(tampil, "REC", (45, 38), 0.7, 2)
    teks(tampil, time.strftime("%d.%m.%Y %H:%M:%S"), (w - 320, 38), 0.6, 2)

    fps = 1 / (now - prev_time) if prev_time else 0
    prev_time = now
    teks(tampil, f"{mode} | {nama_lensa} | FPS {int(fps)} | FOTO {jml_foto}",
         (22, h - 24), 0.6, 2)

    if mode == "LENSA" and quad is not None and hilang < 6:
        r, p, y = orientasi(quad.astype(np.float32))
        teks(tampil, f"ROLL {r:+.0f}  PITCH {p:+.0f}  YAW {y:+.0f}",
             (w - 320, h - 24), 0.55, 2, AMBER)

    if debug:
        b = " ".join("".join("1" if x else "0" for x in f) for _, f, _ in info)
        teks(tampil, f"TANGAN {len(info)}  JARI[{b}]  {status}",
             (22, h - 92), 0.5, 2, AMBER)

    if now < kilat_sampai:
        a = (kilat_sampai - now) / 0.30
        tampil = cv2.addWeighted(tampil, 1 - a, np.full_like(tampil, 255), a, 0)

    cv2.imshow("RETROLENS", tampil)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    elif k == ord(" "):
        idx_lensa = (idx_lensa + 1) % len(LENSA)
    elif k == ord("m"):
        mode = MODE_URUT[(MODE_URUT.index(mode) + 1) % len(MODE_URUT)]
        mulai_diam = mulai_ganti = quad = None
        pena_akhir = pena_halus = None
        menggambar = False
        beruntun_ya = beruntun_tidak = 0
        hilang = 99
    elif k == ord("c"):
        kanvas[:] = 0
        goresan = 0
    elif k == ord("p"):
        idx_pena = (idx_pena + 1) % len(PENA)
    elif k == ord("s"):
        simpan(tampil, "MANUAL")
    elif k == ord("d"):
        debug = not debug

kamera.stop()
cv2.destroyAllWindows()
landmarker.close()
print(f"Selesai. {jml_foto} foto tersimpan di folder '{FOLDER}'.")