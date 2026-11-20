@echo off
title FLASK2 - Voice-to-Voice Sentiment Analysis
cd /d "%~dp0"

echo ===================================================
echo     FLASK2 - VOICE-TO-VOICE SENTIMENT ANALYSIS
echo ===================================================
echo.

REM Tambahkan FFmpeg lokal ke PATH jika ada
if exist "%~dp0ffmpeg\bin" (
    set "PATH=%~dp0ffmpeg\bin;%PATH%"
    echo [*] FFmpeg lokal ditemukan dan ditambahkan ke PATH.
)

REM Cek keberadaan Virtual Environment [venv]
if exist "%~dp0venv\Scripts\activate.bat" (
    echo [*] Mengaktifkan virtual environment [venv]...
    call "%~dp0venv\Scripts\activate.bat"
) else (
    echo [!] Virtual environment [venv] tidak ditemukan!
    echo [*] Mencoba menjalankan dengan Python sistem...
)

echo.
echo [*] Membuka browser di http://127.0.0.1:5000 dalam 3 detik...
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:5000"

echo [*] Menjalankan server Flask [app.py]...
echo [i] Tekan CTRL + C untuk menghentikan server.
echo.

python app.py
if %errorlevel% neq 0 (
    echo.
    echo [!] Terjadi kesalahan saat menjalankan aplikasi [Error code: %errorlevel%].
    echo [!] Pastikan library terinstall: pip install -r requirements.txt
)

echo.
echo Tekan tombol apa saja untuk keluar...
pause >nul

