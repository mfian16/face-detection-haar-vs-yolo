import cv2
import time
import torch

# Cek apakah CUDA tersedia
if torch.cuda.is_available():
    print("✅ GPU AKTIF - CUDA tersedia")
    print("🖥️  Nama GPU:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU TIDAK AKTIF - Menggunakan CPU")

# Memuat model Haar Cascade hasil training dari file XML
face_cascade = cv2.CascadeClassifier("haar.xml")

# Buka webcam (kamera default, biasanya index 0)
cap = cv2.VideoCapture(0)

# Ambil ukuran frame dari webcam
width = int(cap.get(3))
height = int(cap.get(4))
fps = 30  # FPS awal

# Siapkan penyimpanan video output
out = cv2.VideoWriter("results/realtime/real-time_haar.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps, (width, height))

# Inisialisasi variabel statistik
total_frames = 0   # Total frame yang diproses
detected = 0       # Jumlah frame yang terdeteksi wajah
total_time = 0     # Total waktu inferensi deteksi

# Waktu awal untuk menghitung durasi real-time (maks. 10 detik)
start_time = time.time()

# Mulai loop real-time
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Hentikan jika gagal baca dari kamera

    # Hentikan loop jika sudah lewat 10 detik
    if time.time() - start_time > 10:
        break

    total_frames += 1  # Tambah jumlah frame yang diproses

    # Mirror frame agar pengguna seperti bercermin
    frame = cv2.flip(frame, 1)

    # Tambah kontras dan kecerahan
    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=40)

    # Konversi ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mulai pengukuran waktu deteksi
    start = time.time()

    # Lakukan deteksi wajah
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.075,
        minNeighbors=250,
        minSize=(500, 500),  # Ukuran minimum wajah yang dideteksi
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    end = time.time()
    total_time += (end - start)

    # Jika wajah terdeteksi
    if len(faces) > 0:
        # Ambil hanya wajah terbesar
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[:1]
        detected += 1

        # Gambar kotak biru pada wajah
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Tulis frame ke video output
    out.write(frame)

    # Tampilkan hasil deteksi secara langsung
    cv2.imshow("Haar Real-time (1 box/frame)", frame)

    # Tekan 'q' untuk keluar dari real-time
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Selesai real-time, tutup semua
cap.release()
out.release()
cv2.destroyAllWindows()

# Hitung dan cetak statistik akhir
accuracy = (detected / total_frames) * 100 if total_frames > 0 else 0
avg_time = (total_time / total_frames) * 1000 if total_frames > 0 else 0

print("\n=== HAAR CASCADE (Real-Time) ===")
print(f"Total Frame:\t\t{total_frames}")
print(f"Frame Berisi Wajah:\t{total_frames}") 
print(f"Wajah Terdeteksi:\t{detected}")
print(f"Akurasi:\t\t{accuracy:.2f}%")
print(f"Rata-rata Waktu:\t{avg_time:.2f} ms")
