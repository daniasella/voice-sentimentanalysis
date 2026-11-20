@echo off
setlocal enabledelayedexpansion
title Push Repository - SENTIMEN-ANALISIS-TEXT-TO-SPEECH
cd /d "%~dp0.."

echo ===================================================================
echo     PUSH REPO KE GITHUB
echo     Repo: Amirul-Hasan99/SENTIMEN-ANALISIS-TEXT-TO-SPEECH
echo ===================================================================
echo.

REM 1. Periksa ketersediaan Git
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Git tidak terdeteksi di sistem PATH!
    echo Pastikan Git sudah terinstall di komputer Anda.
    echo Download: https://git-scm.com/
    pause
    exit /b 1
)

REM 2. Inisialisasi Git jika belum diinisialisasi
if not exist ".git" (
    echo [*] Menginisialisasi repositori Git lokal...
    git init
    git branch -M main
) else (
    echo [*] Repositori Git lokal terdeteksi.
)

REM 3. Konfigurasi Remote Origin
set "REPO_URL=https://github.com/Amirul-Hasan99/SENTIMEN-ANALISIS-TEXT-TO-SPEECH.git"
git remote get-url origin >nul 2>nul
if %errorlevel% neq 0 (
    echo [*] Menambahkan remote origin: %REPO_URL%
    git remote add origin %REPO_URL%
) else (
    echo [*] Memperbarui URL remote origin ke: %REPO_URL%
    git remote set-url origin %REPO_URL%
)

REM 4. Konfigurasi User Git jika belum ada
git config user.name >nul 2>nul
if %errorlevel% neq 0 (
    git config user.name "Amirul-Hasan99"
    git config user.email "amirulhasan@users.noreply.github.com"
    echo [*] Mengatur identitas commit default.
)

echo.
echo ===================================================================
echo                     STATUS PERUBAHAN FILE
echo ===================================================================
git status -s
echo ===================================================================
echo.

REM 5. Input Pesan Commit
set "DEFAULT_MSG=Integrasi Neon DB, Riwayat Analisis Speech, dan Fitur Ekspor Excel/PDF"
set "COMMIT_MSG="
set /p COMMIT_MSG="Masukkan pesan commit (tekan ENTER untuk default): "
if "!COMMIT_MSG!"=="" (
    set "COMMIT_MSG=%DEFAULT_MSG%"
)

echo.
echo [*] Melakukan git add semua file yang relevan...
git add .

echo [*] Melakukan git commit dengan pesan: "!COMMIT_MSG!"...
git commit -m "!COMMIT_MSG!"
if %errorlevel% neq 0 (
    echo [i] Tidak ada perubahan baru yang perlu di-commit, atau commit telah selesai.
)

echo.
echo [*] Mengirim (push) ke GitHub origin main...
echo [i] Jika diminta kredensial GitHub, silakan login melalui popup browser / Personal Access Token.
echo.

git push -u origin main
if %errorlevel% neq 0 (
    echo.
    echo [!] Push ke branch main gagal atau butuh branch master. Mencoba push ke master...
    git push -u origin master
)

if %errorlevel% equ 0 (
    echo.
    echo ===================================================================
    echo [SUCCESS] Berhasil melakukan push ke:
    echo           https://github.com/Amirul-Hasan99/SENTIMEN-ANALISIS-TEXT-TO-SPEECH
    echo ===================================================================
) else (
    echo.
    echo ===================================================================
    echo [CATATAN] Jika push gagal karena autentikasi, pastikan akun GitHub
    echo           Anda memiliki izin push ke repositori tersebut.
    echo ===================================================================
)

echo.
pause
