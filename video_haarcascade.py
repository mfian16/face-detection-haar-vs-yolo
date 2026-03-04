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
face_cascade = cv2.CascadeClassifier("haar.xml")

# Daftar path video input dan output
video_list = [
    ("input/videos/60.mp4", "results/haar_videos/60.mp4"),
    ("input/videos/-60.mp4", "results/haar_videos/-60.mp4")
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

    # Ambil ukuran frame dan fps dari video asli
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Siapkan objek untuk menyimpan video output
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Inisialisasi statistik
    total_frames = 0    # Jumlah semua frame
    detected = 0        # Jumlah frame yang terdeteksi wajah
    total_time = 0      # Total waktu inferensi

    # Loop frame demi frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Jika frame tidak dapat dibaca, hentikan loop

        total_frames += 1

        # Konversi ke grayscale (komentar karena model kamu mungkin sudah dilatih langsung dengan warna)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        start = time.time()

        # Deteksi wajah menggunakan Haar Cascade langsung dari frame berwarna
        faces = face_cascade.detectMultiScale(
            frame, 
            scaleFactor=1.075,
            minNeighbors=10,
            minSize=(400, 400),  # Disesuaikan dengan ukuran wajah dalam video
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        end = time.time()
        total_time += (end - start)

        # Jika ada wajah terdeteksi
        if len(faces) > 0:
            # Pilih wajah dengan ukuran terbesar (menghindari deteksi ganda)
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:1]
            x, y, w, h = faces[0]
            # Gambar kotak biru di sekitar wajah
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            detected += 1

        # Tambahkan label nama video input pada frame
        label_text = f"Output: {video_path}"
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Simpan frame yang telah diproses ke dalam video output
        out.write(frame)

    # Selesai memproses video
    cap.release()
    out.release()

    # Hitung dan cetak statistik akhir
    accuracy = (detected / total_frames) * 100 if total_frames else 0
    avg_time = (total_time / total_frames) * 1000 if total_frames else 0

    print(f"Total Frame:           {total_frames}")
    print(f"Wajah Terdeteksi:      {detected}")
    print(f"Akurasi Deteksi:       {accuracy:.2f}%")
    print(f"Rata-rata Waktu:       {avg_time:.2f} ms")
    print(f"Video hasil disimpan sebagai: {output_path}")
