import cv2
import time
import torch

# Cek apakah CUDA tersedia
if torch.cuda.is_available():
    print("✅ GPU AKTIF - CUDA tersedia")
    print("🖥️  Nama GPU:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU TIDAK AKTIF - Menggunakan CPU")
# Memuat model Haar Cascade dari file XML hasil training

from ultralytics import YOLO

# Memuat model YOLO hasil pelatihan
model = YOLO("yolo.pt")

# Daftar input/output video
video_list = [
    ("input/videos/60.mp4", "results/yolo_videos/60.mp4"),
    ("input/videos/-60.mp4", "results/yolo_videos/-60.mp4")
]

# Loop untuk memproses setiap video
for idx, (video_path, output_path) in enumerate(video_list, start=1):
    print(f"\n=== MEMPROSES VIDEO {idx} ===")
    print(f"Input:  {video_path}")
    print(f"Output: {output_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Gagal membuka video: {video_path}")
        continue

    # Ambil informasi video: lebar, tinggi, dan fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Buat objek penulis video_output
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Statistik untuk evaluasi
    total_frames = 0     # Total frame yang diproses
    detected = 0         # Jumlah frame yang terdeteksi wajah
    total_time = 0       # Total waktu inferensi YOLO

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Hentikan loop jika frame tidak berhasil dibaca (selesai)

        total_frames += 1  # Tambahkan jumlah frame

        # Mulai pengukuran waktu deteksi
        start = time.time()
        results = model.predict(frame, imgsz=640, verbose=False)  # Prediksi menggunakan YOLO
        end = time.time()

        total_time += (end - start)

        # Ambil hasil kotak prediksi
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            detected += 1  # Jika ada wajah terdeteksi

            # Ambil hanya 1 bounding box terbesar
            boxes = sorted(
                boxes, 
                key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]),
                reverse=True
            )[:1]

            # Gambar kotak hijau di sekitar wajah
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Tambahkan label nama video input pada frame
        label_text = f"Output: {video_path}"
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Tulis frame yang telah diproses ke video_output
        out.write(frame)

    # Selesai proses video ini
    cap.release()
    out.release()

    # Evaluasi akhir untuk video ini
    accuracy = (detected / total_frames) * 100 if total_frames else 0
    avg_time = (total_time / total_frames) * 1000 if total_frames else 0

    # Cetak hasil statistik
    print(f"Total Frame:           {total_frames}")
    print(f"Frame Berisi Wajah:    {total_frames}")
    print(f"Wajah Terdeteksi:      {detected}")
    print(f"Akurasi:               {accuracy:.2f}%")
    print(f"Rata-rata Waktu:       {avg_time:.2f} ms")
    print(f"Video hasil disimpan sebagai: {output_path}")
