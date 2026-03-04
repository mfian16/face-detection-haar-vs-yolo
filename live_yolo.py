import cv2
import time
from ultralytics import YOLO

# Memuat model YOLOv8 hasil pelatihan
model = YOLO("yolo.pt") 

# Membuka webcam (kamera default)
cap = cv2.VideoCapture(0)
width = int(cap.get(3))  # Lebar frame
height = int(cap.get(4)) # Tinggi frame
fps = 30  # FPS awal (nanti diubah berdasarkan total frame dan durasi)

# Inisialisasi statistik
total_frames = 0   # Jumlah semua frame
detected = 0       # Jumlah frame yang berhasil mendeteksi wajah
total_time = 0     # Total waktu inferensi

# File output video
output_filename = "results/realtime/real-time_yolo.mp4"
frames = []  # Menyimpan semua frame yang diproses
start_time = time.time()  # Waktu mulai deteksi

# Loop utama untuk membaca dari kamera
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Hentikan jika gagal membaca frame

    # Membalik tampilan kamera (mirror mode)
    frame = cv2.flip(frame, 1)

    # Tingkatkan kontras + kecerahan (opsional)
    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=40)

    # Hentikan jika sudah lebih dari 10 detik
    elapsed_time = time.time() - start_time
    if elapsed_time > 10:
        break

    total_frames += 1

    # Mulai deteksi
    t0 = time.time()
    results = model.predict(source=frame, imgsz=480, verbose=False)
    t1 = time.time()

    infer_time_ms = (t1 - t0) * 1000  # Waktu deteksi 1 frame dalam ms
    total_time += (t1 - t0)  # Tambahkan ke total waktu

    # Ambil hasil bounding box
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        detected += 1
        # Ambil hanya 1 wajah (pertama / terbesar)
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Simpan frame yang sudah diproses
    frames.append(frame.copy())

    # Tampilkan di jendela OpenCV
    cv2.imshow("YOLO Real-Time (Mirror + Save)", frame)

    # Tekan 'q' untuk keluar lebih awal
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Setelah selesai, tutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()

# Hitung FPS aktual dari durasi 10 detik
fps = total_frames / 10

# Simpan seluruh frame ke dalam file video
out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
for f in frames:
    out.write(f)
out.release()

# Statistik akhir
accuracy = (detected / total_frames) * 100 if total_frames else 0
avg_time = (total_time / total_frames) * 1000 if total_frames else 0

# Cetak statistik
print("\n=== YOLO (Real-Time) ===")
print(f"Total Frame:           {total_frames}")
print(f"Frame Berisi Wajah:    {total_frames}")  
print(f"Wajah Terdeteksi:      {detected}")
print(f"Akurasi Deteksi:       {accuracy:.2f}%")
print(f"Rata-rata Waktu:       {avg_time:.2f} ms")
