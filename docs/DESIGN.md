# DESIGN.md — System & UI/UX Design Document
## Voice-to-Voice Sentiment Analysis & Speech-to-Text Error Evaluation

---

### 1. Arsitektur Komponen (High-Level Architecture)

```
+-----------------------------------------------------------------+
|                  1. PRESENTATION LAYER (Web Client)             |
|  - HTML5 / CSS3 / Vanilla JS                                    |
|  - Audio Capture (MediaRecorder WebM/Opus) & Visualizer (AudioContext) |
|  - Status Monitor (State Machine) & Transcript Correction Dual-Pane |
|  - Metrik: Jarak Edit Kata dari Koreksi Pengguna                |
|  - Indikator: Skor Model (Bukan Probabilitas Terkalibrasi)      |
+-----------------------------------------------------------------+
                                | HTTP POST (FormData / JSON)
                                v
+-----------------------------------------------------------------+
|                  2. CONTROLLER / API LAYER (Flask)              |
|  - Endpoint /analyze (Audio input)                              |
|  - Endpoint /reanalyze_text (Text correction input)             |
|  - Endpoint /wake_word, /analyze_answer, /tts_*                 |
|  - Robust Error Handling & Standardized JSON Responses          |
+-----------------------------------------------------------------+
          |                                      |
          v                                      v
+-----------------------------+        +---------------------------------+
| 3. AUDIO CONVERSION & STT   |        | 4. NLP & PREPROCESSING PIPELINE |
| - FFmpeg (Bundled binary)   |        | - Normalization (Slang, Noise)  |
|   16kHz mono WAV conversion |        | - Lexicon Lookup (Cleaned Vocab)|
| - Google Speech Recognition |        | - Context-Aware Spelling        |
+-----------------------------+        |   (Bigram LM from Sentences)    |
                                       | - Rule-based Sentiment Guard    |
                                       +---------------------------------+
                                                         |
                                                         v
                                       +---------------------------------+
                                       | 5. INFERENCE & DECISION AUDIT   |
                                       | - TF-IDF Vectorizer (5k vocab)  |
                                       | - SVC(kernel="linear", C=1.0,   |
                                       |     probability=True, ovr,      |
                                       |     random_state=42)            |
                                       | - Skor Model (Uncalibrated)     |
                                       | - Audit Decision Function OvO   |
                                       +---------------------------------+
                                                         |
                                                         v
                                       +---------------------------------+
                                       | 6. SPEECH SYNTHESIS (gTTS)      |
                                       | - Audio feedback generation     |
                                       | - Transient file management     |
                                       +---------------------------------+

                                       +---------------------------------+
                                       | 7. OFFLINE BENCHMARK & EVAL     |
                                       | - evaluate.py (Standalone)      |
                                       | - Awal: 15.099 baris            |
                                       | - Eksklusi 4 baris konflik      |
                                       | - Deduplikasi: 73 disingkirkan  |
                                       | - Akhir: 15.022 baris unik      |
                                       | - Stratified Split (Seed 42)    |
                                       | - TF-IDF & LM Fit on Train Only |
                                       | - Primary: Identical Config SVC |
                                       |   (random_state=42 tercatat)    |
                                       | - Diagnostic: Saved Model3      |
                                       | - Perturbation Sensitivity Sim  |
                                       +---------------------------------+
```

---

### 2. Metodologi Evaluasi Ilmiah ([evaluate.py](evaluate.py))

#### 2.1 Audit Data & Langkah Deduplikasi Sebelum Split
Proses penyiapan dataset dilakukan dengan langkah akuntansi data yang transparan dan terdokumentasi:
1. **Jumlah Baris Awal**: 15.099 baris pada `model/data_training.csv`.
2. **Pengecualian Dua Teks Berlabel Konflik**:
   - Teks 1: `@tanyakanrl Liat aja jule dia selingkuh lagi` (baris 13382: `negatif`, baris 13784: `netral`).
   - Teks 2: `@chanbaekgey selingkuh kak si jule mana udah punya anak 3...` (baris 13592: `negatif`, baris 13894: `positif`).
   - Sebanyak **4 baris dari 2 teks konflik ini dikeluarkan secara eksplisit** dari set data evaluasi.
3. **Deduplikasi Teks Identik Berlabel Sama**:
   - Dari sisa 15.095 baris, ditemukan 73 baris teks yang identik secara persis dengan baris lain dan memiliki label yang sama.
   - Sebanyak **73 baris duplikat ini disingkirkan** untuk menyisakan satu representasi unik per teks.
4. **Jumlah Baris Akhir yang Dipakai**:
   $$15.099 - 4 - 73 = 15.022 \text{ baris data teks unik}$$
5. **Pencegahan Kebocoran Data Spesifik Teks Identik**:
   - Pembagian *stratified train/test split* (80% train = 12.017 baris, 20% test = 3.005 baris) dilakukan langsung pada 15.022 data unik tersebut dengan `random_state=42`.
   - Hal ini menjamin bahwa **teks identik dipastikan tidak tersebar lintas split train dan test**.

#### 2.2 Isolasi Ketat Test Set (No Data Snooping)
- Semua komponen yang belajar dari data **wajib dibangun hanya menggunakan split train**:
  - `TfidfVectorizer(max_features=5000, ngram_range=(1, 1), norm="l2")`: `fit` hanya pada data train; `transform` pada test.
  - `BigramLM`: dihitung murni dari kalimat train yang dinormalisasi.
- Data test tetap murni tidak terlihat (*unseen*) hingga evaluasi akhir.

#### 2.3 Kesesuaian Konfigurasi Baseline Retraining & Reproduksibilitas
- Untuk memastikan komparasi yang adil dan konsisten dengan model tersimpan serta menjamin reproduksibilitas hasil kalibrasi probabilitas libsvm:
  - Menggunakan model `sklearn.svm.SVC` dengan parameter identik:
    ```python
    SVC(
        kernel="linear",
        C=1.0,
        probability=True,
        decision_function_shape="ovr",
        random_state=42
    )
    ```
  - Vektorisasi identik:
    ```python
    TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 1),
        norm="l2"
    )
    ```
  - **Pencatatan Konfigurasi**: Konfigurasi lengkap classifier dan vectorizer dicatat secara utuh di header laporan eksperimen.
  - (Catatan: Tidak menggunakan `LinearSVC` karena pustaka `LinearSVC` berbasis liblinear dan tidak mendukung parameter `probability` atau `decision_function_shape`).

#### 2.4 Pemisahan Peran Model: Baseline vs Diagnostik
1. **Baseline Model Retraining (Metrik Uji Utama)**:
   - Sumber resmi metrik pengujian ilmiah: Macro-averaged F1-Score, Precision, Recall, Confusion Matrix.
2. **Model Tersimpan `svm_model3.pkl` (Diagnostik Sahaja)**:
   - Dilaporkan sebagai pembanding diagnostik dengan keterbukaan bahwa data latih aslinya berpotensi tumpang-tindih dengan CSV saat ini.

#### 2.5 Simulasi Sensitivitas Classifier terhadap Perturbasi Teks
- Pengujian toleransi classifier terhadap distorsi teks buatan (penggantian/penghapusan karakter dan kata acak pada tingkat 5%, 10%, 20%).
- Diberi judul dan label eksplisit: **"Simulasi Sensitivitas Classifier terhadap Perturbasi Teks"** (bukan evaluasi dampak STT nyata yang memerlukan korpus ucapan akustik dan transkrip referensi terverifikasi).

---

### 3. Spesifikasi Alur Data & Logika Pra-pemrosesan Runtime

#### 3.1 Penanganan Kamus Bawaan Proyek (`model/corpus.txt`)
- Berkas `model/corpus.txt` adalah daftar kosakata bawaan proyek (sumber belum terverifikasi secara independen sebagai KBBI, sehingga tidak diklaim sebagai KBBI).
- **Normalisasi Kamus Sesuai Preprocessing**:
  - Untuk setiap baris, ekstrak token alfabetis murni melalui pola `re.findall(r'[a-zA-Z]+', line.lower())`.
  - Simpan hanya token kata tunggal dengan panjang minimal 2 karakter ke dalam himpunan kosakata `INDONESIAN_DICTIONARY`.
  - **Larangan Keras Bigram LM**: Jangan pernah memasukkan entri daftar kata tersebut ke dalam `BigramLM`. Model bigram membutuhkan urutan kata alami kalimat (dibangun murni dari kalimat-kalimat `data_training.csv` yang dinormalisasi).

#### 3.2 Alur Inferensi Suara Utama (`/analyze`)
1. **Audio Capture**: Browser merekam audio ucapan (`audio/webm;codecs=opus`).
2. **Audio Conversion**: Server memanggil executable FFmpeg lokal (`ffmpeg/bin/ffmpeg.exe`) untuk mengubah ke PCM WAV 16.000 Hz, mono.
3. **STT Transcription**: Google Speech Recognition menghasilkan `text_raw`.
4. **Text Preprocessing**:
   - `text_normalization`: case folding, noise cleaning dengan proteksi angka, normalisasi kata gaul (`SLANG_DICT`), reduksi karakter berulang, penghapusan kata jeda vokal lisan.
   - `context_aware_spelling`: pencarian kandidat ejaan dengan jarak Levenshtein yang divalidasi skor probabilitas log Bigram LM yang konsisten.
5. **Inference & Scoring**:
   - Vektorisasi TF-IDF $x \in \mathbb{R}^{5000}$.
   - Pengecekan aturan khusus (*rule-based guard*).
   - Inferensi `SVC(kernel="linear", C=1.0, probability=True, decision_function_shape="ovr")`.
   - Pelabelan skor keyakinan sebagai **Skor Model** (bukan probabilitas terkalibrasi).
6. **TTS Feedback**: gTTS menghasilkan audio respon ringkas.
7. **Pembersihan Berkas**: Berkas WebM dan WAV sementara langsung dihapus dari disk.

#### 3.3 Alur Analisis Ulang Teks Koreksi (`/reanalyze_text`)
1. Pengguna menyunting teks pada panel koreksi jika transkrip STT mengalami salah dengar.
2. Teks koreksi diproses ulang oleh pipeline NLP dan model sentimen.
3. Sistem menghitung:
   - **Jarak Edit Kata dari Koreksi Pengguna**: rasio jarak Levenshtein kata terhadap panjang token maksimum.
   - Status konsistensi prediksi: apakah label sentimen konsisten atau bergeser (*sentiment shift*).
4. Data koreksi hanya berada pada memori sesi aktif klien, **tidak disimpan secara permanen**.

---

### 4. UI/UX Design

#### 4.1 Status State Machine (Frontend)
- `IDLE`: Tombol "Mulai Bicara" aktif, visualizer hening.
- `RECORDING`: Animasi morphing blob aktif, tombol berubah menjadi "Mendengarkan...", batas durasi otomatis via silence detection (VAD).
- `PROCESSING`: Status "Memproses audio dan transkripsi...".
- `COMPLETED`: Menampilkan panel hasil lengkap.
- `ERROR`: Menampilkan kotak pesan ramah dengan tombol eksplisit "Coba Lagi" (tanpa auto-retry tanpa batas).

#### 4.2 Dual-Pane Comparison
- **Panel Kiri (Transkripsi Otomatis STT)**:
  - Teks Mentah STT.
  - Teks Setelah Normalisasi & Ejaan.
  - Label Sentimen & Skor Model.
- **Panel Kanan (Koreksi Pengguna)**:
  - Area penyuntingan teks interaktif.
  - Tombol aksi "Evaluasi Teks yang Diperbaiki".
  - Hasil sentimen teks koreksi + status pergeseran sentimen.
  - Nilai **Jarak Edit Kata dari Koreksi Pengguna** (% disparitas kata).
