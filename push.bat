@echo off
setlocal enabledelayedexpansion
title Push Repository - SENTIMEN-ANALISIS-TEXT-TO-SPEECH
cd /d "%~dp0"

echo ===================================================================
echo             AUTO-PUSH GITHUB & DEPLOY VERCEL
echo      Repo: Amirul-Hasan99/SENTIMEN-ANALISIS-TEXT-TO-SPEECH
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
set "DEFAULT_MSG=Update aplikasi dan perbaikan deployment Vercel"
set "COMMIT_MSG="
set /p COMMIT_MSG="Masukkan pesan commit (tekan ENTER untuk default: '!DEFAULT_MSG!'): "
if "!COMMIT_MSG!"=="" (
    set "COMMIT_MSG=%DEFAULT_MSG%"
)

echo.
echo [*] Menambahkan semua perubahan ke staging (git add .)...
git add .

echo [*] Melakukan commit: "!COMMIT_MSG!"...
git commit -m "!COMMIT_MSG!"
if %errorlevel% neq 0 (
    echo [i] Tidak ada perubahan baru yang perlu di-commit, atau commit sudah tersimpan.
)

echo.
echo [*] Mengirim (git push) ke GitHub...
git push -u origin main
if %errorlevel% neq 0 (
    echo [!] Mencoba branch master...
    git push -u origin master
)

REM 6. Deploy langsung ke Vercel Production
where vercel >nul 2>nul
if %errorlevel% equ 0 (
    echo.
    echo [*] Menjalankan update deployment ke Vercel Production (vercel --prod)...
    vercel --prod --yes
)

if %errorlevel% equ 0 (
    echo.
    echo ===================================================================
    echo [SUKSES] Perubahan berhasil di-push ke GitHub dan Vercel Production!
    echo          Website Anda sudah terupdate dengan versi terbaru.
    echo          URL Repo: https://github.com/Amirul-Hasan99/SENTIMEN-ANALISIS-TEXT-TO-SPEECH
    echo ===================================================================
) else (
    echo.
    echo ===================================================================
    echo [SELESAI] Proses telah selesai dijalankan.
    echo ===================================================================
)

echo.
pause
