import os
import sys

# Tambahkan path direktori root proyek ke sys.path agar seluruh modul dapat diimpor
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import app

# Vercel WSGI entry point
# Vercel Serverless Function akan otomatis mengenali objek `app` ini.
