# AGENT.MD — Panduan Operasional AI Software Engineering Agent
## Proyek: Voice Sentiment Analysis & STT Error Evaluation (Tugas Akhir)

---

### 1. Peran & Batasan Ruang Lingkup
- **Peran**: AI software engineering agent yang bertugas mengembangkan repositori ini menjadi prototipe Tugas Akhir Teknik Informatika yang terstruktur, andal, dan dapat dipertanggungjawabkan secara ilmiah.
- **Topik Resmi**: “Analisis sentimen dari ucapan Bahasa Indonesia serta evaluasi pengaruh kesalahan Speech-to-Text terhadap hasil klasifikasi.”
- **Batasan Ketat (Scope Guard)**:
  - **DILARANG** memperluas ruang lingkup ke pengenalan emosi akustik, deteksi stres suara, diagnosis medis/psikologis, atau penilaian public speaking kecuali ada permintaan tertulis dari pengguna.
  - **DILARANG** mengarang metrik evaluasi atau mengklaim hasil uji coba tanpa menjalankan kode pengujian secara nyata.
  - **DILARANG** menimpa berkas model asli (`model/model3/*`) atau dataset asli (`model/data_training.csv`). Segala varian baru harus disimpan sebagai artefak terpisah.

---

### 2. Aturan Lingkungan & Kompatibilitas Scikit-Learn
1. **Sistem Operasi**: Windows 64-bit dengan default shell `powershell`.
2. **OpenBLAS Issue pada Windows**:
   - Jika mengeksekusi modul yang memuat `scipy`, `numpy`, atau `scikit-learn`, variabel lingkungan `OPENBLAS_NUM_THREADS=1` **wajib** diatur terlebih dahulu untuk menghindari galat `OpenBLAS error: Memory allocation still failed`.
   - Contoh di PowerShell: `$env:OPENBLAS_NUM_THREADS="1"`
3. **Penamaan & Konfigurasi Classifier yang Konsisten**:
   - Model yang digunakan pada proyek adalah `sklearn.svm.SVC` dengan kernel linear:
     ```python
     SVC(
         kernel="linear",
         C=1.0,
         probability=True,
         decision_function_shape="ovr",
         random_state=42
     )
     ```
   - **Dilarang keras menyebutnya `LinearSVC`**, karena `LinearSVC` tidak mendukung parameter `probability` atau `decision_function_shape`.
   - Menambahkan `random_state=42` pada `SVC(...)` memastikan proses fitting kalibrasi probabilitas libsvm (cross-validation internal) bersifat sepenuhnya deterministik dan dapat direproduksi.
   - Tidak menambahkan monkeypatch internal pada runtime. Prioritaskan environment dengan versi yang cocok (`scikit-learn==1.7.1`).
4. **FFmpeg Bundled Binary**:
   - Sistem memiliki direktori FFmpeg lokal di `ffmpeg/bin/ffmpeg.exe`.
   - Kode Python **harus** memeriksa dan memprioritaskan binary lokal:
     ```python
     local_ffmpeg = os.path.join(base_dir, "ffmpeg", "bin", "ffmpeg.exe")
     ffmpeg_cmd = local_ffmpeg if os.path.exists(local_ffmpeg) else "ffmpeg"
     ```

---

### 3. Integritas Data & Pra-pemrosesan Teks
1. **Penanganan Kamus Bawaan (`model/corpus.txt`)**:
   - Jangan pernah mengklaim `model/corpus.txt` bersumber dari "KBBI" tanpa bukti metadata dan verifikasi independen; sebut secara jujur sebagai `daftar kosakata bawaan proyek (model/corpus.txt)`.
   - Sebelum dipakai, sistem harus memfilter entri menjadi token kata tunggal bersih yang alfabetis (`word.isalpha()`), lowercase, selaras dengan tahap normalisasi teks.
   - **Larangan Keras Bigram LM**: Jangan pernah memasukkan baris daftar kata kamus ke dalam `BigramLM`. Model bigram membutuhkan urutan kata alami dalam kalimat (dilatih dari kalimat `data_training.csv`).
2. **Koreksi Ejaan**:
   - Rumus perbandingan log probabilitas Bigram LM harus konsisten secara matematis (hindari perbandingan log probabilitas positif vs negatif yang membuat seleksi tidak valid).

---

### 4. Metodologi Ilmiah & Evaluasi Model (`evaluate.py`)
1. **Protokol Penanganan Data Sebelum Split**:
   - Catat dan laporkan tahapan deduplikasi secara presisi:
     - Jumlah baris awal: **15.099 baris**.
     - Eksklusi eksplisit dua teks berlabel konflik: **4 baris dikeluarkan**.
     - Deduplikasi teks identik berlabel sama: **73 baris disingkirkan**.
     - Jumlah baris akhir yang dipakai untuk split evaluasi: **15.022 baris teks unik**.
2. **Pencegahan Kebocoran Data Spesifik Teks Identik**:
   - Stratified train/test split (80:20, `random_state=42`) dilakukan langsung pada 15.022 baris data unik tersebut, sehingga teks identik dipastikan tidak tersebar lintas split train dan test.
3. **Isolasi Penuh Data Train untuk Komponen Pembelajaran**:
   - Semua komponen yang belajar dari data—termasuk `TfidfVectorizer(max_features=5000, ngram_range=(1,1), norm="l2")` dan `BigramLM`—**wajib di-fit hanya menggunakan data train**.
   - Data test harus tetap murni tidak terlihat (*unseen*) hingga evaluasi akhir.
4. **Penyelarasan Konfigurasi Baseline Retraining**:
   - Baseline retraining menggunakan model identik: `SVC(kernel="linear", C=1.0, probability=True, decision_function_shape="ovr", random_state=42)` dan `TfidfVectorizer(max_features=5000, ngram_range=(1,1), norm="l2")`.
   - Seluruh konfigurasi classifier dicatat secara lengkap pada laporan evaluasi.
5. **Pemisahan Metrik Uji Utama vs Diagnostik**:
   - Baseline Retraining murni: **Metrik Uji Utama** (Macro-F1, Precision, Recall, Confusion Matrix).
   - Model tersimpan (`svm_model3.pkl`): **Diagnostik Sahaja** (dengan catatan keterbukaan ilmiah mengenai potensi bias riwayat pelatihan).
6. **Terminologi Evaluasi**:
   - Gunakan istilah **"Jarak Edit Kata dari Koreksi Pengguna"** untuk interaksi UI penyuntingan transkrip. Istilah **Word Error Rate (WER)** hanya dipakai jika ada audio dengan transkrip referensi terverifikasi.
   - Distorsi kata sintetis wajib dilabeli sebagai **"Simulasi Sensitivitas Classifier terhadap Perturbasi Teks"**, bukan evaluasi dampak STT nyata.
   - Nilai keyakinan model di antarmuka web wajib dilabeli sebagai **"Skor Model"** (bukan probabilitas terkalibrasi) kecuali kalibrasinya diuji secara formal.

---

### 5. Aturan Perubahan Kode & File
- Tidak ada duplikasi file dokumentasi; gunakan [architecture.md](architecture.md) sebagai dokumen arsitektur resmi tunggal.
- Jangan memulai implementasi kode backend/frontend sebelum rencana final dipresentasikan dan disetujui pengguna.
