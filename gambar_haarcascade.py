import time
import os
import cv2
import torch

# Cek apakah CUDA tersedia
if torch.cuda.is_available():
    print("✅ GPU AKTIF - CUDA tersedia")
    print("🖥️  Nama GPU:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU TIDAK AKTIF - Menggunakan CPU")
# Path ke file model Haar Cascade yang telah dilatih
cascade_path = 'haar.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Daftar folder input (gambar asli) dan output (hasil deteksi)
folder_list = [("input/images", "results/haar_images")]

# Proses tiap folder satu per satu
for idx, (image_folder, output_folder) in enumerate(folder_list, start=1):
    print(f"\n=== MEMPROSES FOLDER {idx} ===")
    print(f"Input:  {image_folder}")
    print(f"Output: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)  # Buat folder output jika belum ada

    # Statistik per folder
    total_images = 0    # Total jumlah gambar yang diproses
    detected = 0        # Jumlah gambar yang berhasil dideteksi wajah
    total_time = 0      # Total waktu deteksi semua gambar

    # Proses semua gambar dalam folder
    for filename in os.listdir(image_folder):
        # Proses hanya file gambar dengan ekstensi tertentu
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_folder, filename)
            image = cv2.imread(img_path)  # Membaca gambar
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Konversi ke grayscale (wajib untuk Haar)

            # Mulai hitung waktu deteksi
            start = time.time()
            faces = face_cascade.detectMultiScale(
                image,
                scaleFactor=1.075,      # Ukuran pengurangan gambar setiap iterasi
                minNeighbors=10,       # Semakin tinggi, semakin ketat deteksi
                minSize=(400, 400),     # Ukuran minimum wajah yang akan dideteksi
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            end = time.time()

            if len(faces) > 0:
                # Ambil wajah dengan ukuran terbesar saja (menghindari deteksi ganda)
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:1]
                x, y, w, h = faces[0]
                # Gambar kotak biru pada area wajah
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                detected += 1  # Tambah jumlah wajah yang terdeteksi

            total_time += (end - start)  # Tambahkan waktu proses deteksi
            total_images += 1            # Tambahkan jumlah gambar yang diproses

            # Simpan gambar hasil deteksi ke folder output
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image)

    # Evaluasi hasil per folder
    accuracy = (detected / total_images) * 100 if total_images > 0 else 0
    avg_time_ms = (total_time / total_images) * 1000 if total_images > 0 else 0

    # Cetak statistik akhir
    print(f"Jumlah Gambar:        {total_images}")
    print(f"Wajah Terdeteksi:     {detected}")
    print(f"Akurasi:              {accuracy:.2f}%")
    print(f"Rata-rata Waktu:      {avg_time_ms:.2f} ms")
