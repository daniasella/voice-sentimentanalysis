import os
import sys
# Prevent OpenBLAS memory allocation crash on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Ensure local venv site-packages can be loaded for pure-python packages (speech_recognition, gtts, fuzzywuzzy)
local_site_packages = os.path.join(os.path.abspath("."), "venv", "Lib", "site-packages")
if os.path.exists(local_site_packages) and local_site_packages not in sys.path:
    sys.path.append(local_site_packages)


import re
import csv
import time
import wave
import joblib
import logging
import tempfile
import subprocess
import shutil
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import speech_recognition as sr
from gtts import gTTS
from fuzzywuzzy import fuzz
from datetime import datetime

import db
import exporter

from model.rulebased import detect_umpatan, is_literal_animal_context, rule_based_sentiment
from preprocessing.normalization import text_normalization
from preprocessing.context_spelling import context_aware_spelling
from preprocessing.ngram_model import BigramLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==============================
#   PATHS SETUP
# ==============================
if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

template_dir = os.path.join(base_dir, "templates")
static_dir = os.path.join(base_dir, "static")
model_dir = os.path.join(base_dir, "model")
preprocessing_dir = os.path.join(base_dir, "preprocessing")

# Local FFmpeg binary fallback
local_ffmpeg = os.path.join(base_dir, "ffmpeg", "bin", "ffmpeg.exe")
FFMPEG_CMD = local_ffmpeg if os.path.exists(local_ffmpeg) else "ffmpeg"
logging.info(f"Using FFmpeg command: {FFMPEG_CMD}")

def process_uploaded_audio_to_wav(audio_file):
    """
    Memproses file audio yang diunggah.
    1. Jika file sudah berformat WAV PCM yang valid, langsung gunakan tanpa FFmpeg (kompatibel penuh Vercel Serverless).
    2. Jika bukan WAV dan FFmpeg tersedia (misal di PC lokal), konversi ke WAV mono 16kHz via FFmpeg.
    3. Jika bukan WAV dan FFmpeg TIDAK tersedia, kembalikan pesan error yang informatif.
    Mengembalikan: (wav_path, cleanup_list, error_msg)
    """
    orig_name = getattr(audio_file, 'filename', '') or 'audio'
    ext = os.path.splitext(orig_name)[1].lower()
    if not ext:
        ext = '.wav'

    temp_input = tempfile.mktemp(suffix=ext)
    audio_file.save(temp_input)
    cleanup_list = [temp_input]

    # Cek apakah file sudah berformat WAV PCM yang valid
    try:
        with wave.open(temp_input, 'rb') as wf:
            if wf.getnframes() > 0:
                return temp_input, cleanup_list, None
    except Exception:
        pass

    # Bukan file WAV valid, periksa ketersediaan FFmpeg
    ffmpeg_available = False
    if os.path.isabs(FFMPEG_CMD) and os.path.exists(FFMPEG_CMD):
        ffmpeg_available = True
    elif shutil.which(FFMPEG_CMD):
        ffmpeg_available = True

    if not ffmpeg_available:
        return None, cleanup_list, "Format audio bukan WAV dan FFmpeg tidak tersedia di server (Vercel). Pastikan browser mengirimkan rekaman dalam format WAV."

    temp_wav = tempfile.mktemp(suffix=".wav")
    cleanup_list.append(temp_wav)

    try:
        proc = subprocess.run(
            [FFMPEG_CMD, "-y", "-i", temp_input, "-ac", "1", "-ar", "16000", temp_wav],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if proc.returncode == 0 and os.path.exists(temp_wav):
            return temp_wav, cleanup_list, None
        return None, cleanup_list, "Gagal mengonversi file audio dengan FFmpeg"
    except Exception as e:
        return None, cleanup_list, f"Gagal menjalankan FFmpeg: {e}"

# ==============================
#   LOAD DICTIONARY (CORPUS.TXT)
# ==============================
INDONESIAN_DICTIONARY = set()
corpus_file = os.path.join(model_dir, "corpus.txt")
if os.path.exists(corpus_file):
    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                for word in re.findall(r'[a-zA-Z]+', line.lower()):
                    if len(word) >= 2:
                        INDONESIAN_DICTIONARY.add(word)
        logging.info(f"Loaded {len(INDONESIAN_DICTIONARY)} words into INDONESIAN_DICTIONARY from corpus.txt")
    except Exception as e:
        logging.warning(f"Error loading corpus.txt: {e}")

if not INDONESIAN_DICTIONARY:
    logging.warning("Using fallback default dictionary words.")
    INDONESIAN_DICTIONARY = {"saya", "kamu", "dia", "mereka", "kita", "bagus", "buruk", "senang", "sedih", "terasi"}

# ==============================
#   INIT FLASK
# ==============================
app = Flask(__name__,
            template_folder=template_dir,
            static_folder=static_dir)

# Initialize Neon PostgreSQL database
try:
    db.init_db()
except Exception as e:
    logging.warning(f"Could not connect to Neon DB at startup: {e}")

# ==============================
#   LOAD MODEL ARTIFACTS
# ==============================
model_path = os.path.join(model_dir, "model3", "svm_model3.pkl")
vec_path = os.path.join(model_dir, "model3", "tfidf_vectorizer3.pkl")
le_path = os.path.join(model_dir, "model3", "label_encoder.pkl")

model = joblib.load(model_path)
# Compatibility guard for scikit-learn >= 1.9
if not hasattr(model, "_effective_probability"):
    model._effective_probability = getattr(model, "probability", True)

vectorizer = joblib.load(vec_path)
label_encoder = joblib.load(le_path)
VOCAB = vectorizer.get_feature_names_out()
logging.info("Model, Vectorizer, and Label Encoder successfully loaded.")

# ==============================
#   BUILD BIGRAM LM (ONCE AT STARTUP)
# ==============================
corpus_sentences = []
csv_path = os.path.join(model_dir, "data_training.csv")
if os.path.exists(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txt = row.get("text", "")
            if txt:
                corpus_sentences.append(text_normalization(txt))

bigram_lm = BigramLM(corpus_sentences)
logging.info(f"Bigram LM built from {len(corpus_sentences)} normalized training sentences.")

# ==============================
#   TTS FOLDER
# ==============================
try:
    TTS_FOLDER = os.path.join(app.static_folder, "tts")
    os.makedirs(TTS_FOLDER, exist_ok=True)
    _test_probe = os.path.join(TTS_FOLDER, ".probe_test")
    with open(_test_probe, "w") as _f:
        _f.write("1")
    os.remove(_test_probe)
except Exception:
    TTS_FOLDER = os.path.join(tempfile.gettempdir(), "flask_tts")
    os.makedirs(TTS_FOLDER, exist_ok=True)

def text_to_speech(text, prefix="tts"):
    tts = gTTS(text, lang="id")
    filename = f"{prefix}_{next(tempfile._get_candidate_names())}.mp3"
    path = os.path.join(TTS_FOLDER, filename)
    tts.save(path)
    return f"tts/{filename}"

# ==============================
#   PIPELINE & DISTANCE HELPERS
# ==============================
def predict_sentiment_pipeline(text):
    """
    Menjalankan normalisasi, koreksi ejaan, aturan sentimen, dan klasifikasi SVC.
    Mengembalikan (final_text, sentiment, model_score).
    """
    normalized_text = text_normalization(text)
    final_text = context_aware_spelling(
        normalized_text,
        bigram_lm,
        VOCAB,
        INDONESIAN_DICTIONARY
    )
    tokens = final_text.split()

    # 1. Rule-based umpatan
    if detect_umpatan(tokens):
        return final_text, "negatif", 1.0

    # 2. Rule-based konteks hewan literal
    if "anjing" in tokens and is_literal_animal_context(tokens):
        return final_text, "positif", 1.0

    # 3. Rule-based formal sentence
    formal_res = rule_based_sentiment(final_text)
    if formal_res is not None:
        return final_text, formal_res, 1.0

    # 4. SVM Classifier
    X = vectorizer.transform([final_text])
    if X.nnz == 0:
        return final_text, "netral", 0.3333

    pred_idx = model.predict(X)[0]
    sentiment = label_encoder.inverse_transform([pred_idx])[0]

    # Model score (uncalibrated probability from SVC)
    try:
        proba = model.predict_proba(X)[0]
        model_score = round(float(proba[pred_idx]), 4)
    except Exception:
        model_score = 1.0

    return final_text, sentiment, model_score

def compute_word_edit_distance(text_a, text_b):
    """
    Menghitung Word Edit Distance (Levenshtein jarak tingkat kata) antara dua string.
    Mengembalikan (distance_integer, ratio_float).
    """
    words_a = text_a.strip().split()
    words_b = text_b.strip().split()
    len_a, len_b = len(words_a), len(words_b)

    if len_a == 0 and len_b == 0:
        return 0, 0.0
    if len_a == 0:
        return len_b, 1.0
    if len_b == 0:
        return len_a, 1.0

    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(len_a + 1):
        dp[i][0] = i
    for j in range(len_b + 1):
        dp[0][j] = j

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if words_a[i - 1].lower() == words_b[j - 1].lower() else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    dist = dp[len_a][len_b]
    max_len = max(len_a, len_b)
    ratio = dist / max_len if max_len > 0 else 0.0
    return dist, ratio

# ==============================
#        ROUTES
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main')
def main_page():
    return render_template('main.html')

@app.route('/tts_opening')
def tts_opening():
    opening_text = "Halo, saya adalah sistem voice recognition untuk analisis sentimen!"
    filename = text_to_speech(opening_text, prefix="opening")
    return jsonify({"filename": filename})

@app.route('/tts_silakan')
def tts_silakan():
    filename = text_to_speech("Silakan berbicara", prefix="prompt")
    return jsonify({"filename": filename})

# ==============================
#   ANALYZE (STT UTAMA)
# ==============================
@app.route('/analyze', methods=['POST'])
def analyze():
    cleanup_list = []
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        temp_wav_path, cleanup_list, err_msg = process_uploaded_audio_to_wav(audio_file)
        if err_msg:
            return jsonify({"error": err_msg}), 400 if "Format audio" in err_msg else 500

        # Calculate audio duration
        audio_duration = 0.0
        try:
            with wave.open(temp_wav_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    audio_duration = round(frames / float(rate), 2)
        except Exception as e:
            logging.warning(f"Error calculating audio duration: {e}")

        # Google Speech Recognition
        r = sr.Recognizer()
        start_time = time.perf_counter()
        with sr.AudioFile(temp_wav_path) as source:
            audio_data = r.record(source)
            raw_text = r.recognize_google(audio_data, language="id-ID")
        end_time = time.perf_counter()
        execution_time = round(end_time - start_time, 4)

        logging.info(f"[STT-Google] Duration: {audio_duration}s | Exec Time: {execution_time}s")
        logging.info(f"RAW TEXT: {raw_text}")

        # Sentiment Analysis Pipeline
        final_text, sentiment, model_score = predict_sentiment_pipeline(raw_text)
        logging.info(f"FINAL TEXT: {final_text} | SENTIMENT: {sentiment} | SCORE: {model_score}")

        # TTS Feedback
        output_filename = text_to_speech(f"Hasil analisis sentimen Anda adalah {sentiment}")
        question_filename = text_to_speech("Apakah ada lagi yang ingin Anda bicarakan?", prefix="question")

        # Simpan riwayat speech ke Neon DB
        history_id = None
        try:
            saved_record = db.insert_speech_record(
                text_raw=raw_text,
                text_normalized=final_text,
                sentiment=sentiment,
                model_score=model_score,
                audio_duration=audio_duration,
                execution_time=execution_time
            )
            if saved_record:
                history_id = saved_record["id"]
        except Exception as db_err:
            logging.warning(f"Error saving speech to Neon DB: {db_err}")

        return jsonify({
            "history_id": history_id,
            "text_raw": raw_text,
            "text_normalized": final_text,
            "sentiment": sentiment,
            "model_score": model_score,
            "confidence_label": "Skor Model (tidak terkalibrasi)",
            "audio_duration": audio_duration,
            "execution_time": execution_time,
            "tts_audio": output_filename,
            "question_audio": question_filename
        })

    except sr.UnknownValueError:
        return jsonify({"error": "Ucapan tidak dapat dikenali oleh sistem STT. Silakan ulangi dengan lebih jelas."}), 422
    except sr.RequestError as e:
        return jsonify({"error": f"Layanan STT tidak dapat dihubungi: {e}"}), 503
    except Exception as e:
        logging.error(f"Error in /analyze: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        for p in cleanup_list:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

# ==============================
#   REANALYZE TEXT (KOREKSI PENGGUNA)
# ==============================
@app.route('/reanalyze_text', methods=['POST'])
def reanalyze_text():
    """
    Menganalisis ulang teks yang telah dikoreksi pengguna dan menghitung jarak edit kata.
    """
    try:
        data = request.get_json(force=True)
        raw_text = data.get("raw_text", "")
        corrected_text = data.get("corrected_text", "")
        history_id = data.get("history_id")

        if not corrected_text or not corrected_text.strip():
            return jsonify({"error": "Teks koreksi tidak boleh kosong."}), 400

        # Sentiment inference pada teks koreksi
        final_text, sentiment, model_score = predict_sentiment_pipeline(corrected_text)

        # Hitung jarak edit kata dari teks STT asli ke teks koreksi pengguna
        edit_dist, edit_ratio = compute_word_edit_distance(raw_text, corrected_text)

        # Perbarui record di Neon DB jika history_id tersedia
        if history_id:
            try:
                db.update_speech_correction(
                    record_id=history_id,
                    user_corrected_text=corrected_text,
                    corrected_sentiment=sentiment,
                    word_edit_distance=edit_dist,
                    word_edit_ratio=edit_ratio
                )
            except Exception as db_err:
                logging.warning(f"Error updating correction in Neon DB: {db_err}")

        return jsonify({
            "history_id": history_id,
            "text_raw": raw_text,
            "text_corrected": corrected_text,
            "text_normalized": final_text,
            "sentiment": sentiment,
            "model_score": model_score,
            "confidence_label": "Skor Model (tidak terkalibrasi)",
            "word_edit_distance": edit_dist,
            "word_edit_ratio": round(edit_ratio, 4),
            "edit_distance_label": "Jarak Edit Kata dari Koreksi Pengguna"
        })

    except Exception as e:
        logging.error(f"Error in /reanalyze_text: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ==============================
#        ANALYZE ANSWER
# ==============================
@app.route('/analyze_answer', methods=['POST'])
def analyze_answer():
    cleanup_list = []
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        temp_wav_path, cleanup_list, err_msg = process_uploaded_audio_to_wav(audio_file)
        if err_msg:
            return jsonify({"error": err_msg}), 500

        r = sr.Recognizer()
        with sr.AudioFile(temp_wav_path) as source:
            audio_data = r.record(source)
            answer_text = r.recognize_google(audio_data, language="id-ID").lower()

        restart_words = ["ya", "iya", "yup", "oke", "lanjut", "bicara", "silahkan", "silakan", "mau", "lagi", "ya dong"]
        exit_words = ["tidak", "enggak", "ga", "no", "stop", "berhenti", "tidak deh", "ga deh", "ga dong", "tidak dong"]

        for word in restart_words:
            if fuzz.ratio(answer_text, word) > 80 or word in answer_text:
                return jsonify({"action": "restart"})

        for word in exit_words:
            if fuzz.ratio(answer_text, word) > 80 or word in answer_text:
                closing_filename = text_to_speech("Terima kasih, sampai jumpa!", prefix="closing")
                return jsonify({"action": "exit", "tts_audio": closing_filename})

        return jsonify({"action": "unknown", "text": answer_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        for p in cleanup_list:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

# ==============================
#        WAKE WORD
# ==============================
@app.route('/wake_word', methods=['POST'])
def wake_word():
    cleanup_list = []
    try:
        if 'audio' not in request.files:
            return jsonify({"wake": False})

        audio_file = request.files['audio']
        temp_wav_path, cleanup_list, err_msg = process_uploaded_audio_to_wav(audio_file)
        if err_msg:
            return jsonify({"wake": False})

        r = sr.Recognizer()
        with sr.AudioFile(temp_wav_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data, language="id-ID").lower()

        logging.info(f"WAKE WORD TEXT: {text}")
        wake_words = ["halo neo", "halo neu", "halo meo"]
        for word in wake_words:
            if fuzz.ratio(text, word) > 80:
                return jsonify({"wake": True})

        return jsonify({"wake": False})

    except Exception as e:
        logging.info(f"Wake word recognition: {e}")
        return jsonify({"wake": False})
    finally:
        for p in cleanup_list:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

# ==============================
#        TTS SERVE
# ==============================
@app.route('/tts/<path:filename>')
@app.route('/static/tts/<path:filename>')
def tts_audio(filename):
    return send_from_directory(TTS_FOLDER, filename)

# ==============================
#   NEON DB HISTORY & EXPORT
# ==============================
@app.route('/api/history', methods=['GET'])
def get_history():
    """Mengambil daftar riwayat speech dari Neon DB."""
    try:
        limit = request.args.get('limit', default=100, type=int)
        sentiment = request.args.get('sentiment', default=None, type=str)
        keyword = request.args.get('q', default=None, type=str)

        records = db.get_speech_history(limit=limit, sentiment_filter=sentiment, keyword=keyword)
        return jsonify({
            "status": "success",
            "data": records,
            "count": len(records)
        })
    except Exception as e:
        logging.error(f"Error fetching history: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/api/history/<int:record_id>', methods=['DELETE'])
def delete_history_item(record_id):
    """Menghapus satu record riwayat dari Neon DB."""
    try:
        success = db.delete_speech_record(record_id)
        if success:
            return jsonify({"status": "success", "message": f"Data #{record_id} berhasil dihapus."})
        return jsonify({"status": "error", "message": "Data tidak ditemukan."}), 404
    except Exception as e:
        logging.error(f"Error deleting record: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/api/history/clear', methods=['POST', 'DELETE'])
def clear_history():
    """Membersihkan seluruh riwayat di Neon DB."""
    try:
        success = db.clear_all_history()
        if success:
            return jsonify({"status": "success", "message": "Seluruh riwayat berhasil dibersihkan."})
        return jsonify({"status": "error", "message": "Gagal membersihkan riwayat."}), 500
    except Exception as e:
        logging.error(f"Error clearing history: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/export/excel', methods=['GET'])
def export_excel():
    """Ekspor seluruh atau filtered riwayat analisis ke format Excel (.xlsx)."""
    try:
        sentiment = request.args.get('sentiment', default=None, type=str)
        keyword = request.args.get('q', default=None, type=str)
        records = db.get_speech_history(limit=1000, sentiment_filter=sentiment, keyword=keyword)

        output = exporter.generate_excel(records)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"riwayat_sentimen_neon_{timestamp}.xlsx"
        return send_file(
            output,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logging.error(f"Error exporting excel: {e}", exc_info=True)
        return jsonify({"error": f"Gagal mengekspor file Excel: {e}"}), 500

@app.route('/export/pdf', methods=['GET'])
def export_pdf():
    """Ekspor seluruh atau filtered riwayat analisis ke format PDF."""
    try:
        sentiment = request.args.get('sentiment', default=None, type=str)
        keyword = request.args.get('q', default=None, type=str)
        records = db.get_speech_history(limit=1000, sentiment_filter=sentiment, keyword=keyword)

        output = exporter.generate_pdf(records)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"riwayat_sentimen_neon_{timestamp}.pdf"
        return send_file(
            output,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logging.error(f"Error exporting pdf: {e}", exc_info=True)
        return jsonify({"error": f"Gagal mengekspor file PDF: {e}"}), 500

# ==============================
#        RUN APP
# ==============================
if __name__ == '__main__':
    app.run(debug=False)
