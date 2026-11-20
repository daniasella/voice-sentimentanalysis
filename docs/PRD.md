# PRD — Product Requirements Document
## Prototipe Tugas Akhir: Analisis Sentimen dari Ucapan Bahasa Indonesia & Evaluasi Pengaruh Kesalahan Speech-to-Text

---

### 1. Informasi Proyek
- **Judul Prototipe**: Sistem Analisis Sentimen Berbasis Suara Bahasa Indonesia dengan Evaluasi Dampak Kesalahan Transkripsi STT.
- **Fokus Utama**: Analisis sentimen dari ucapan Bahasa Indonesia serta evaluasi terukur mengenai pengaruh kesalahan *Speech-to-Text* (STT) terhadap performa klasifikasi sentimen teks.
- **Konteks**: Prototipe Tugas Akhir / Skripsi Teknik Informatika.
- **Batasan Eksplisit**: Sistem ini **hanya** berfokus pada transkripsi ucapan Bahasa Indonesia dan analisis polaritas sentimen (Positif, Netral, Negatif). Sistem **tidak** mencakup pengenalan emosi akustik, deteksi stres suara, diagnosis medis/psikologis, maupun penilaian kemampuan *public speaking*.

---

### 2. Latar Belakang & Masalah
1. **Rentan Terhadap Noise STT**: Model analisis sentimen teks umumnya dilatih pada teks tertulis. Dalam aplikasi berbasis suara (*voice-driven*), keluaran STT (*Automatic Speech Recognition / ASR*) sering mengalami kesalahan pengenalan kata (substitusi, delesi, insersi) akibat derau lingkungan, intonasi, atau variasi dialek.
2. **Ketiadaan Transparansi Pipeline**: Pengguna dan penguji skripsi perlu melihat rantai pemrosesan secara utuh: dari rekaman suara mentah $\rightarrow$ transkrip STT $\rightarrow$ normalisasi teks $\rightarrow$ koreksi ejaan $\rightarrow$ skor klasifikasi $\rightarrow$ luaran suara (TTS).
3. **Kebutuhan Evaluasi Dampak STT yang Metodologis & Jujur Secara Ilmiah**:
   - Skrip pelatihan dari model tersimpan (`svm_model3.pkl`) tidak tersedia di repositori, dan terdapat kemungkinan model tersebut dilatih menggunakan keseluruhan isi `data_training.csv`. Oleh karena itu, pengujian model tersimpan pada split data yang sama **hanya berstatus sebagai diagnostik**, bukan metrik uji yang sah.
   - **Metrik uji ilmiah utama** wajib diperoleh dari **baseline model yang dilatih ulang secara terdokumentasi**: semua komponen yang belajar dari data (termasuk TF-IDF dan BigramLM) **hanya dipelajari dari split train**. Split test wajib terisolasi penuh hingga tahap evaluasi akhir.
   - **Protokol Penanganan Data Sebelum Pembagian (*Split*)**:
     1. Dataset awal memuat **15.099 baris**.
     2. Empat baris dari dua kelompok teks berlabel konflik dikeluarkan (**4 baris dikecualikan**).
     3. Teks identik berlabel sama dideduplikasi (**73 baris duplikat disingkirkan**).
     4. Data akhir yang digunakan untuk evaluasi adalah **15.022 baris teks unik**.
     5. Dilakukan *stratified train/test split* (80:20, `random_state=42`) pada 15.022 baris data unik tersebut, sehingga teks identik dipastikan tidak tersebar lintas split train dan test.
   - Metrik **Word Error Rate (WER)** hanya valid digunakan pada pengujian ASR yang memiliki transkrip referensi (*ground truth*) akustik yang diverifikasi. Untuk interaksi pengguna di UI web, perbedaannya dilabeli sebagai **Jarak Edit Kata dari Koreksi Pengguna**.
   - Pengujian distorsi kata buatan dilabeli sebagai **Simulasi Sensitivitas Classifier terhadap Perturbasi Teks**, bukan evaluasi dampak STT riil.

---

### 3. Pengguna & Skenario Penggunaan
- **Mahasiswa / Peneliti**: Menjalankan eksperimen, melatih/menguji model baseline secara reproduktif, mengukur metrik ilmiah (Macro-F1, Precision, Recall, Confusion Matrix).
- **Dosen Penguji / Pembimbing**: Menguji reliabilitas prototipe secara interaktif, memvalidasi transparansi langkah inferensi, dan memeriksa keabsahan metodologi evaluasi.
- **Responden / Pengguna Akhir**: Berinteraksi menggunakan suara melalui antarmuka web, melihat hasil transkripsi dan sentimen, serta memiliki opsi mengoreksi teks hasil ucapan jika terjadi salah pengenalan kata.

---

### 4. Kebutuhan Fungsional (Functional Requirements)

#### Modul A: Alur Analisis Sentimen Transparan
- **FR-A1**: Menerima audio ucapan dari browser (format WebM/Opus) dan mengonversinya secara lokal menjadi WAV mono 16 kHz melalui FFmpeg yang dibundel di proyek.
- **FR-A2**: Mengirim audio ke mesin pengenal ucapan (Google Speech Recognition) untuk menghasilkan transkrip mentah (*raw transcript*).
- **FR-A3**: Menjalankan tahapan preprocessing teks terstruktur:
  1. *Case folding*
  2. *Noise & punctuation cleaning* (dengan perlindungan format angka)
  3. *Slang normalization* (kamus kata gaul ke baku)
  4. *Repeated character reduction* (e.g., "bagussss" $\rightarrow$ "bagus")
  5. *Filler word removal* (penghapusan jeda vokal lisan: "eee", "umm", "anu")
  6. *Context-aware spelling correction* (berbasis Levenshtein Distance & Bigram Language Model yang dilatih dari data kalimat nyata, diverifikasi dengan daftar kata bawaan proyek)
- **FR-A4**: Melakukan klasifikasi sentimen multi-kelas (3 kelas: `positif`, `netral`, `negatif`) menggunakan model `sklearn.svm.SVC` dengan parameter `SVC(kernel="linear", C=1.0, probability=True, decision_function_shape="ovr", random_state=42)` dan TF-IDF (5.000 fitur).
- **FR-A5**: Menampilkan keyakinan prediksi sebagai **Skor Model**, dengan catatan transparan bahwa skor tersebut merupakan luaran fungsi keputusan dan belum melalui uji kalibrasi probabilitas formal (seperti Brier score / ECE).
- **FR-A6**: Menampilkan nilai pengaruh fitur terhadap pemisah keputusan berpasangan (OvO) tanpa klaim "kontribusi kata eksak per kelas".
- **FR-A7**: Menghasilkan respon suara berbasis Text-to-Speech (gTTS) dalam Bahasa Indonesia.

#### Modul B: Koreksi Transkrip untuk Evaluasi Interaktif
- **FR-B1**: Antarmuka menyediakan panel perbandingan berdampingan (*dual-pane comparison*):
  - Teks Transkrip Otomatis (STT).
  - Teks Koreksi Pengguna (dapat diedit secara langsung).
- **FR-B2**: Pengguna dapat memicu tombol **"Analisis Ulang Teks Koreksi"** untuk melihat perbedaan prediksi sentimen antara teks STT dan teks yang telah diperbaiki.
- **FR-B3**: Menghitung **Jarak Edit Kata dari Koreksi Pengguna** (*word edit distance / discrepancy rate*) antara teks STT dan teks koreksi pengguna.
- **FR-B4**: Data koreksi bersifat sementara (*session-only*) dan **tidak** disimpan ke basis data permanen secara default.

#### Modul C: Ketahanan Sistem & Penanganan Error
- **FR-C1**: Indikator status siklus hidup audio yang jelas di UI: `Siap Merekam`, `Merekam...`, `Mengonversi Audio`, `Transkripsi STT...`, `Menganalisis Sentimen...`, `Selesai`, atau `Gagal`.
- **FR-C2**: Penanganan aman terhadap audio terlalu singkat, mikrofon hening, atau kegagalan jaringan.
- **FR-C3**: Tidak ada *infinite retry loop*. Batas percobaan ulang maksimal 3 kali dan menyediakan tombol manual untuk mencoba lagi.
- **FR-C4**: Pesan galat ke pengguna diformat secara ramah tanpa membocorkan internal stack trace server.

#### Modul D: Modul Evaluasi Ilmiah & Benchmark (Offline Pipeline)
- **FR-D1**: Skrip evaluasi terpisah dari *web runtime* (`evaluate.py`), mencegah beban komputasi ganda pada startup server.
- **FR-D2**: **Pencatatan Audit & Deduplikasi Data Sebelum Split**:
  - Baris awal: 15.099 baris.
  - Eksklusi eksplisit: 4 baris dari 2 kelompok teks berlabel konflik (`@tanyakanrl...` dan `@chanbaekgey...`).
  - Deduplikasi teks identik berlabel sama: 73 baris disingkirkan.
  - Jumlah baris akhir yang dipakai untuk pembagian data: 15.022 baris unik.
- **FR-D3**: **Pencegahan Kebocoran Data Spesifik Teks Identik**: *Stratified split* (80:20, `random_state=42`) dilakukan langsung pada 15.022 baris data unik tersebut, sehingga teks identik dipastikan tidak tersebar lintas split train dan test.
- **FR-D4**: **Isolasi Penuh Test Set**: Semua komponen yang belajar dari data—termasuk `TfidfVectorizer` (vocabulary & idf) dan `BigramLM` (probabilitas transisi)—hanya dibangun dari data train. Data test tetap murni tidak terlihat (*unseen*) hingga evaluasi akhir.
- **FR-D5**: **Penyelarasan Konfigurasi Baseline & Pencatatan Lengkap**: Baseline model menggunakan konfigurasi identik dengan artefak proyek yang dilengkapi penentu seed acak untuk kalibrasi probabilitas libsvm: `SVC(kernel="linear", C=1.0, probability=True, decision_function_shape="ovr", random_state=42)` dan `TfidfVectorizer(max_features=5000, ngram_range=(1,1), norm="l2")`. Konfigurasi lengkap wajib dicatat di laporan eksperimen.
- **FR-D6**: **Reproduksibilitas**: Aturan pembagian data, rasio split (80:20), seed acak split (`random_state=42`), serta seed classifier (`random_state=42`) disimpan secara eksplisit dan dicatat di laporan keluaran.
- **FR-D7**: **Pemisahan Metrik Utama vs Diagnostik**:
  - Baseline Retraining murni: sumber **metrik uji utama**.
  - Model Tersimpan (`svm_model3.pkl`): **diagnostik sahaja** (dengan catatan potensi bias riwayat pelatihan).
- **FR-D8**: Pelaporan metrik: Macro-averaged Precision, Recall, F1-Score (metrik utama), Accuracy (pelengkap), dan Confusion Matrix.
- **FR-D9**: **Simulasi Sensitivitas Classifier terhadap Perturbasi Teks**: menguji ketahanan model terhadap variasi tingkat perturbasi kata sintetis.

---

### 5. Kebutuhan Non-Fungsional
1. **Keamanan & Privasi**: Audio sementara dihapus seketika setelah STT selesai; koreksi pengguna bersifat sementara (*ephemeral*).
2. **Kinerja & Responsivitas**: Startup server Flask cepat (< 3 detik) tanpa looping dataset 15k pada booting.
3. **Integritas Kamus & Preprocessing**: Daftar kata bawaan proyek (`model/corpus.txt`) diproses sesuai pipeline teks (tokenisasi alfabetis, lowercase) tanpa klaim sumber tak terverifikasi.
