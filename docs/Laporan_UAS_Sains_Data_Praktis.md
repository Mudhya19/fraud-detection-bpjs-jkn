# Laporan Asesmen Ujian Akhir Semester
**Mata Ujian:** Sains Data Praktis  
**Universitas:** Universitas Islam Indonesia  
**Program Studi:** Informatika Program Magister / Kelas B  
**Dosen Penguji:** Dhomas Hatta Fudholi, ST, M.Eng, Ph.D · Taufiq Hidayat, ST, MCS, PhD  
**Nama Mahasiswa:** Muhammad Dhiauddin (NIM: 25917024)

---

## Jawaban Soal Ujian (Bobot: 100%)

### 1. Penentuan dan Penjelasan Data (Bobot: 35%)

**Sumber Data:**
Sistem ini menggunakan data berskala raksasa (Big Data) yang diperoleh dari **BPJS Kesehatan Republik Indonesia**, dipublikasikan melalui ajang **Healthkathon 2022**. Dataset ini merepresentasikan rekam jejak pengajuan klaim pembiayaan dari Fasilitas Kesehatan Tingkat Pertama (FKTP) dan Fasilitas Kesehatan Rujukan Tingkat Lanjutan (FKRTL) rawat inap maupun rawat jalan di seluruh Indonesia. Total volume data mencapai lebih dari **11.4 Juta baris klaim** dengan ukuran mentah (raw) menembus angka 1.58 Gigabyte.

**Penjelasan Detail Atribut (Fitur):**
Karakteristik utama dari data medis dan administratif BPJS ini sangat kompleks dan heterogen. Atributnya dikelompokkan menjadi:
1.  **Fitur Demografis:** `jenkel` (Jenis Kelamin), `usia` (Umur pasien saat klaim).
2.  **Fitur Administratif & Temporal:** `id` (Nomor kunjungan/klaim), `tgldatang` & `tglpulang` (Rentang waktu perawatan yang kemudian diekstrak menjadi `Length of Stay / durasi_rawat`), `jenispel` (Jenis pelayanan Inap/Jalan), `kelasrawat`, `typefaskes`, `tingkatpelayanan`, `dati2` (Lokasi Kabupaten/Kota).
3.  **Fitur Medis (Kardinalitas Tinggi):** `cbg` (Case Based Group), `diagfktp` (Diagnosis awal FKTP), tabel relasi **Diagnosa Tambahan (ICD-10)**, dan tabel relasi **Prosedur Medis (ICD-9-CM)**. Karena seorang pasien dapat menerima banyak diagnosis dan tindakan secara bersamaan, fitur ini diagregasi menjadi fitur frekuensi (seperti `jumlah_prosedur`, `jumlah_diagnosa_tambahan`).
4.  **Fitur Numerik Utama:** `biaya` (Total nominal yang ditagihkan oleh faskes ke BPJS).
5.  **Target Label (Supervised Learning):** `label` yang mendefinisikan status klaim (`0 = Normal`, `1 = Fraud / Tindakan Kecurangan`).

**Catatan Karakteristik:** Data memiliki sifat *Class Imbalance* yang teramat ekstrem; dari total 11.4 juta klaim, hanya sekitar **1.4%** yang terindikasi sebagai Fraud. 

---

### 2. Problem dan Solusi Operasional / Manajerial (Bobot: 35%)

**Problem Operasional & Manajerial BPJS Kesehatan:**
1.  **Ledakan Volume Klaim (Volume Velocity):** BPJS menerima jutaan lembar tagihan klaim per bulannya. Memverifikasi seluruh tagihan ini secara manual (100% audit) oleh tenaga auditor medis (*verifikator*) adalah suatu kemustahilan operasional.
2.  **Risiko Defisit Anggaran DJS:** Apabila verifikasi dilakukan secara melonggar atau mengandalkan audit *random sampling*, klaim yang di-_mark up_ secara curang (Fraud) seperti *Upcoding* diagnosa, *phantom billing*, dan *readmission* palsu akan lolos, memicu kerugian finansial triliunan rupiah dan defisit Dana Jaminan Sosial (DJS).
3.  **Trade-Off Biaya Audit:** Mengerahkan ribuan auditor untuk menginvestigasi klaim juga memakan biaya gaji (*Cost of Audit*) yang masif. 

**Solusi Sains Data yang Ditawarkan:**
Membangun sebuah **Decision Support System (DSS)** berbasis klasifikasi Machine Learning/Deep Learning. Model ini akan memindai (men-*scoring*) 100% dari data klaim yang masuk dalam hitungan detik. 
*   **Aksi Manajerial:** Sistem memfilter dan memberikan skor probabilitas "Bahaya". Klaim dengan probabilitas Normal akan otomatis disetujui (mempercepat *cash-flow* rumah sakit). Sebaliknya, klaim yang masuk kategori "Fraud (1)" akan ditandai (*flagged*).
*   **Dampak Operasional:** Auditor medis tidak perlu lagi membaca jutaan klaim secara acak, melainkan hanya perlu mengaudit *batch* klaim yang secara prediktif dituding bermasalah oleh model.

**Validasi Keberhasilan (Sesuai Simulasi Hasil Model):**
Berdasarkan log hasil *training* model pada **1,710,283 klaim (Test Set)**:
*   Beban operasional manual turun **97.6%**! Dari 1.7 juta klaim, sistem hanya mengirimkan **40,221 klaim anomali** ke meja auditor.
*   Estimasi potensi kerugian Fraud yang berhasil dicegah mencapai **Rp 37 Miliar**.
*   Setelah dipotong biaya *review* manual (*False Alarm* sebesar Rp 13 Miliar), sistem mendatangkan **Net Benefit sebesar Rp 23.59 Miliar** per masa pengujian. Solusi ini secara revolusioner melindungi ketahanan kas BPJS tanpa mengorbankan kecepatan birokrasi.

---

### 3. Penerapan Model Machine Learning / Deep Learning (Bobot: 30%)

**Model yang Dipilih:**
Pendekatan **Deep Learning** menggunakan **Tabular Neural Network (TabNN)**.

**Mengapa Model Ini Cocok?**
Data klaim kesehatan memiliki tantangan klasik: Fitur kategorikal dengan ratusan atau ribuan ragam unik (*High Cardinality Categorical Features*) seperti kode ICD-10 untuk diagnosis medis atau kode wilayah Dati 2. Jika menggunakan pendekatan tradisional (*Logistic Regression, Random Forest*) dipadukan dengan *One-Hot-Encoding (OHE)*, matriks akan meledak (*Curse of Dimensionality*) menjadi puluhan ribu kolom kosong (Sparse), membuat komputasi RAM meledak dan akurasi hancur. 
Deep Learning menyelesaikan masalah ini dengan elegan melalui lapisan **Categorical Embedding**.

**Bagaimana Model Ini Diterapkan?**
1.  **Arsitektur Input (Embedding):** Seluruh fitur teks/kategorikal (jenis kelamin, kode penyakit) diubah menjadi token integer dan dilewatkan ke dalam *Embedding Layers*. Di sini, dimensi raksasa ditekan (dimampatkan) menjadi vektor ruang padat (dense space), sehingga model AI bisa merekam hubungan logis—misal, kemiripan pola kecurangan antara kode diagnosa A dan diagnosa B.
2.  **Arsitektur Input (Numerikal):** Fitur berkelanjutan seperti `biaya` dan `usia` dinormalisasi dengan *StandardScaler* dan digabungkan (Concatenate) dengan output Embedding.
3.  **Hidden Layers (Deep Representation):** Gabungan vektor tersebut dilewatkan pada susunan *Dense Layers* bertingkat untuk mencari pola non-linear. Untuk mencegah *overfitting* dan mengingat dominasi kelompok Normal, setiap lapisan diberikan *Batch Normalization* serta dihalangi dengan *Dropout Layer* (me-non-aktifkan acak sekian persen neuron saat belajar).
4.  **Imbalance Handling:** Untuk mengatasi jumlah persentase kelas normal vs fraud (98% vs 1.4%), fungsi objektif model (Loss Function) disuntikkan **Class Weights**. Pelanggaran/kesalahan klasifikasi pada sampel Fraud (minoritas) dipenalti dengan bobot hukuman yang jauh lebih berat ketimbang salah mengklasifikasikan data Normal.
5.  **Output & Thresholding:** Terdapat 1 neuron ujung dengan aktivasi Sigmoid untuk probabilitas 0% hingga 100%. Untuk mengatasi sensitivitas di ranah asuransi, titik pemisahan kepastian (*Classification Threshold*) diubah dinamis (pada angka 0.88), di mana sistem lebih mementingkan Precision dan ROC-AUC agar tidak terlalu banyak memberikan tuduhan palsu (*False Alarm*) yang bisa memperburuk hubungan dengan rumah sakit mitranya.

**Hasil Performa Model:**
Berdasarkan metrik klasifikasi `notebooks/bpjs_fraud_detection.ipynb`:
*   **ROC-AUC Score:** Tercipta metrik luar biasa di angka **0.9168**. Artinya, kemampuan model untuk membedakan distribusi klaim Normal dan Fraud sangat superior, jauh di atas acakan acak (0.5).
*   **Recall Fraud:** Mampu mendeteksi **55.7%** dari total seluruh kecurangan riil yang terjadi secara presisi dan terukur.
*   Model berhenti secara optimal dengan sistem kalibrasi *Early Stopping* di **Epoch ke-60** akibat saturasi validasi *loss* dan AUC terbaik.
