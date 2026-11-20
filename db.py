import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

# Default Neon DB Connection URL
DEFAULT_DB_URL = "postgresql://neondb_owner:npg_An3OriHysUC0@ep-snowy-haze-at8e3hc2-pooler.c-9.us-east-1.aws.neon.tech/sentimen-analisis?sslmode=require&channel_binding=require"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DB_URL)

logger = logging.getLogger(__name__)

def get_connection():
    """Membuka koneksi ke Neon PostgreSQL database."""
    return psycopg2.connect(DATABASE_URL)

def init_db():
    """Inisialisasi tabel riwayat speech jika belum ada."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS speech_history (
        id SERIAL PRIMARY KEY,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        text_raw TEXT NOT NULL,
        text_normalized TEXT,
        sentiment VARCHAR(50) NOT NULL,
        model_score DOUBLE PRECISION,
        audio_duration DOUBLE PRECISION,
        execution_time DOUBLE PRECISION,
        user_corrected_text TEXT,
        corrected_sentiment VARCHAR(50),
        word_edit_distance INTEGER,
        word_edit_ratio DOUBLE PRECISION
    );
    CREATE INDEX IF NOT EXISTS idx_speech_history_created_at ON speech_history (created_at DESC);
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
            conn.commit()
        logger.info("Tabel speech_history di Neon DB berhasil diverifikasi / diinisialisasi.")
        return True
    except Exception as e:
        logger.error(f"Gagal inisialisasi database Neon: {e}", exc_info=True)
        return False

def insert_speech_record(text_raw, text_normalized, sentiment, model_score, audio_duration, execution_time):
    """Menyimpan hasil analisis ucapan baru ke tabel speech_history."""
    insert_sql = """
    INSERT INTO speech_history (
        text_raw, text_normalized, sentiment, model_score, audio_duration, execution_time
    ) VALUES (%s, %s, %s, %s, %s, %s)
    RETURNING id, created_at;
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(insert_sql, (
                    text_raw,
                    text_normalized,
                    sentiment,
                    model_score,
                    audio_duration,
                    execution_time
                ))
                row = cur.fetchone()
                conn.commit()
                return {"id": row[0], "created_at": row[1].isoformat() if row[1] else None}
    except Exception as e:
        logger.error(f"Gagal menyimpan riwayat speech ke Neon DB: {e}", exc_info=True)
        return None

def update_speech_correction(record_id, user_corrected_text, corrected_sentiment, word_edit_distance, word_edit_ratio):
    """Memperbarui record riwayat ucapan dengan data hasil koreksi pengguna."""
    update_sql = """
    UPDATE speech_history
    SET user_corrected_text = %s,
        corrected_sentiment = %s,
        word_edit_distance = %s,
        word_edit_ratio = %s
    WHERE id = %s
    RETURNING id;
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(update_sql, (
                    user_corrected_text,
                    corrected_sentiment,
                    word_edit_distance,
                    word_edit_ratio,
                    record_id
                ))
                row = cur.fetchone()
                conn.commit()
                return row is not None
    except Exception as e:
        logger.error(f"Gagal memperbarui riwayat koreksi ke Neon DB: {e}", exc_info=True)
        return False

def get_speech_history(limit=100, sentiment_filter=None, keyword=None):
    """Mengambil riwayat analisis dari Neon DB dengan filter opsional."""
    query = """
    SELECT 
        id,
        created_at,
        text_raw,
        text_normalized,
        sentiment,
        model_score,
        audio_duration,
        execution_time,
        user_corrected_text,
        corrected_sentiment,
        word_edit_distance,
        word_edit_ratio
    FROM speech_history
    WHERE 1=1
    """
    params = []

    if sentiment_filter and sentiment_filter.lower() in ["positif", "negatif", "netral"]:
        query += " AND LOWER(sentiment) = %s"
        params.append(sentiment_filter.lower())

    if keyword and keyword.strip():
        query += " AND (text_raw ILIKE %s OR text_normalized ILIKE %s OR user_corrected_text ILIKE %s)"
        kw = f"%{keyword.strip()}%"
        params.extend([kw, kw, kw])

    query += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)

    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
                # Format timestamps to ISO strings for JSON serialization
                for row in rows:
                    if row.get("created_at"):
                        row["created_at_formatted"] = row["created_at"].strftime("%d/%m/%Y %H:%M:%S")
                        row["created_at"] = row["created_at"].isoformat()
                    else:
                        row["created_at_formatted"] = "-"
                return rows
    except Exception as e:
        logger.error(f"Gagal mengambil riwayat speech dari Neon DB: {e}", exc_info=True)
        return []

def delete_speech_record(record_id):
    """Menghapus satu record riwayat berdasarkan ID."""
    query = "DELETE FROM speech_history WHERE id = %s RETURNING id;"
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (record_id,))
                row = cur.fetchone()
                conn.commit()
                return row is not None
    except Exception as e:
        logger.error(f"Gagal menghapus record {record_id} dari Neon DB: {e}", exc_info=True)
        return False

def clear_all_history():
    """Menghapus seluruh data riwayat speech."""
    query = "TRUNCATE TABLE speech_history RESTART IDENTITY;"
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Gagal membersihkan seluruh riwayat di Neon DB: {e}", exc_info=True)
        return False
