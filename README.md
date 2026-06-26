# Fraud Detection BPJS JKN - Sains Data Praktis

Proyek ini dibangun sebagai penyelesaian tugas Ujian Akhir Semester (UTS) mata kuliah **Sains Data Praktis**. Proyek ini mendemonstrasikan implementasi *Deep Learning* tingkat lanjut untuk mengidentifikasi dan mengklasifikasikan anomali / potensi tindak kecurangan (Fraud) pada data historis klaim BPJS Kesehatan.

Dataset yang digunakan berasal dari dataset kompetisi **Healthkathon BPJS 2022** yang terdiri atas belasan juta baris data (mencapai lebih dari 1.5 GB).

## 🚀 Fitur dan Optimasi Teknis
Proyek ini mengadopsi struktur *Data Engineering* tingkat tinggi untuk menangani pemrosesan "Big Data" di lingkungan memori (RAM) lokal yang terbatas, meliputi:
1. **Adaptive Memory Loading**: Teknik pembacaan *chunking* 500ribu baris secara iteratif.
2. **Auto-Downcasting Dtypes**: Pengubahan paksa (*downcasting*) tipe data *integer* dan *float* ke ukuran terendah (`int32`, `float32`, `int8`).
3. **Low-Cardinality Categorization**: Kompresi otomatis pada kolom objek repetitif (seperti `jenkel`, `typefaskes`, `cbg`) menjadi tipe data Pandas `category`.
4. **Aggressive Garbage Collection**: Penghancuran (*deletion*) variabel raksasa secara eksplisit sesaat setelah agregasi data (*merge*) dan pembersihan *cache* memori paksa (`gc.collect()`) sebelum proses training *Machine Learning* berjalan.

## 📁 Struktur Direktori
```text
├── app/                  # Source code aplikasi utama (opsional)
├── config/               # Konfigurasi environment dan variabel
├── data/
│   ├── external/         # Data tambahan dari pihak ketiga
│   ├── processed/        # Data yang sudah melalui tahap pembersihan
│   └── raw/              # Berisi dataset mentah (Healthkathon 2022 CSVs)
├── docs/                 # Dokumentasi (Soal UAS & catatan desain)
├── images/
│   └── output/           # Seluruh ekspor visualisasi data & metrik model
├── logs/                 # Catatan log eksekusi
├── models/               # Penyimpanan bobot Deep Learning (ex: .keras / .h5)
├── notebooks/            # Berisi notebook eksperimen utama (bpjs_fraud_detection.ipynb)
├── src/                  # Generator script & source code pipeline utama
├── test/                 # Script pengujian, debug, dan perbaikan sementara
└── README.md             # Dokumentasi utama proyek
```

## 🧠 Arsitektur Model
Model klasifikasi yang dibangun menggunakan arsitektur **Tabular Neural Network** via `TensorFlow`/`Keras` dengan spesifikasi:
* **Embedding Layers**: Mengubah fitur kategorikal bertipe diskrit dengan kardinalitas beragam ke dalam vektor representasi padat (*dense vector*).
* **Dense Layers**: Memproses fitur numerikal dan menggabungkannya dengan fitur *embedding*.
* **Regularisasi**: Penggunaan *Batch Normalization* dan *Dropout* untuk mencegah *overfitting* pada kelas minoritas (karena Fraud memiliki *Class Imbalance* yang ekstrem).

## 📊 Hasil dan Dampak Bisnis (Business Impact)
Selain menampilkan nilai akurasi dan *Area Under Curve (AUC)*, *notebook* diakhiri dengan modul **Simulasi Dampak Finansial**. Modul ini memproyeksikan perbandingan finansial (dalam Rupiah) antara biaya yang berhasil dihemat BPJS dari pencegahan klaim fraud dibandingkan dengan kerugian yang timbul dari kesalahan prediksi audit (False Alarm).

---
*Dikerjakan dengan penuh ketelitian dan keakuratan sesuai arahan instruktur.*
