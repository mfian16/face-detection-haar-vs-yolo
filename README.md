# Face Detection Comparison (Haar Cascade vs YOLO)

## Overview
Project ini merupakan eksperimen **Computer Vision** yang bertujuan untuk membandingkan performa dua metode deteksi wajah yaitu **Haar Cascade** dan **YOLO (You Only Look Once)**.

Pengujian dilakukan menggunakan tiga jenis input:

- Image Detection
- Video Detection
- Real-time Webcam Detection

Project ini dibuat untuk memahami perbedaan pendekatan antara metode **klasik (Haar Cascade)** dan metode **Deep Learning modern (YOLO)** dalam mendeteksi wajah.

---

## Technologies Used

- Python
- OpenCV
- YOLO (Ultralytics)
- PyTorch
- NumPy

---

## Features

Project ini memiliki beberapa fitur utama:

- Deteksi wajah menggunakan **Haar Cascade**
- Deteksi wajah menggunakan **YOLO**
- Pengujian deteksi pada **gambar**
- Pengujian deteksi pada **video**
- Deteksi wajah secara **real-time menggunakan webcam**
- Menampilkan **bounding box** pada wajah yang terdeteksi

---

## Project Structure
```
face-detection-haar-vs-yolo
│
├── input
│ ├── images
│ └── videos
│
├── results
│ ├── haar_images
│ ├── yolo_images
│ ├── haar_videos
│ ├── yolo_videos
│ └── realtime
│
├── gambar_haarcascade.py
├── gambar_yolo.py
├── video_haarcascade.py
├── video_yolo.py
├── live_haarcascade.py
├── live_yolo.py
├── haar.xml
├── yolo.pt
```

---

## Dataset

Dataset yang digunakan dalam project ini berupa **gambar dan video yang berisi wajah penulis sendiri**.  
Data tersebut digunakan hanya untuk keperluan eksperimen dalam membandingkan metode deteksi wajah.

Repository ini hanya menyertakan **beberapa sample input dan hasil output** untuk menjaga ukuran repository tetap kecil.

---

## Installation

Clone repository ini terlebih dahulu:
git clone https://github.com/mfian16/face-detection-haar-vs-yolo.git

Masuk ke folder project:
cd face-detection-haar-vs-yolo

Install dependency:
pip install -r requirements.txt

---

## How to Run

### Image Detection

Haar Cascade
python gambar_haarcascade.py

YOLO
python gambar_yolo.py

---

### Video Detection

Haar Cascade
python video_haarcascade.py

YOLO
python video_yolo.py

---

### Real-Time Detection

Haar Cascade
python live_haarcascade.py

YOLO
python live_yolo.py

---

## Example Results

Hasil deteksi wajah akan menampilkan **bounding box pada wajah yang terdeteksi** baik pada gambar, video, maupun real-time.

Beberapa contoh hasil dapat dilihat pada folder:
results/

---

## Purpose of This Project

Project ini dibuat sebagai bagian dari eksplorasi pembelajaran **Computer Vision menggunakan Python** untuk memahami perbedaan performa antara metode **Haar Cascade** dan **YOLO** dalam mendeteksi wajah.

---

## Author

Muhammad Fiqih Irfiansyah
