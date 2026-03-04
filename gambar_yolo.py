import time
import os
import cv2

from ultralytics import YOLO

# Memuat model YOLO hasil pelatihan (custom model)
model = YOLO("yolo.pt")

# Daftar folder input (gambar asli) dan output (hasil prediksi)
folder_list = [("input/images", "results/yolo_images")]

# Loop untuk memproses setiap folder
for idx, (image_folder, output_folder) in enumerate(folder_list, start=1):
    print(f"\n=== MEMPROSES FOLDER {idx} ===")
    print(f"Input:  {image_folder}")
    print(f"Output: {output_folder}")

    # Membuat folder output jika belum ada
    os.makedirs(output_folder, exist_ok=True)

    # Inisialisasi statistik untuk folder ini
    total_images = 0          # Jumlah total gambar
    detected = 0              # Jumlah gambar yang berhasil terdeteksi wajahnya
    total_time = 0            # Total waktu proses semua gambar

    # Loop untuk membaca dan memproses semua gambar dalam folder
    for filename in os.listdir(image_folder):
        # Mengecek ekstensi file gambar yang valid
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_folder, filename)
            image = cv2.imread(img_path)  # Membaca gambar

            # Mulai hitung waktu prediksi
            start = time.time()
            results = model.predict(image, imgsz=640, verbose=False)  # Prediksi wajah menggunakan YOLO
            end = time.time()

            # Mengambil hasil deteksi kotak (bounding boxes)
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                detected += 1  # Jika ada wajah terdeteksi

                # Ambil hanya 1 kotak terbesar (area paling besar)
                boxes = sorted(
                    boxes, 
                    key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]),
                    reverse=True
                )[:1]

                # Gambar kotak deteksi pada gambar
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Gambar kotak hijau

            # Tambahkan waktu proses ke total_time
            total_time += (end - start)
            total_images += 1  # Tambah jumlah gambar yang diproses

            # Simpan hasil gambar dengan kotak ke folder output
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image)

    # Hitung dan tampilkan statistik folder ini
    accuracy = (detected / total_images) * 100 if total_images else 0
    avg_time_ms = (total_time / total_images) * 1000 if total_images else 0

    print(f"Jumlah Gambar:        {total_images}")
    print(f"Wajah Terdeteksi:     {detected}")
    print(f"Akurasi:              {accuracy:.2f}%")
    print(f"Rata-rata Waktu:      {avg_time_ms:.2f} ms")
