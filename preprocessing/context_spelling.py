# preprocessing/context_spelling.py

import math
import sys
import os
sys.path.insert(0, os.path.abspath("."))
from preprocessing.spelling_candidate import generate_candidates

COMMON_WORDS = {
    "saya", "aku", "dia", "kami", "kita", "anda",
    "tidak", "bukan", "dan", "atau", "yang",
    "ini", "itu", "pada", "karena"
}

def context_aware_spelling(text, lm, vocabulary, indo_dict, max_edit=1):
    tokens = text.split()
    corrected = []

    for i, token in enumerate(tokens):

        # 🔒 jangan sentuh kata aman (ada di vocab model, kata umum, kamus, atau terlalu pendek)
        if token in vocabulary or token in COMMON_WORDS:
            corrected.append(token)
            continue
        if token in indo_dict:
            corrected.append(token)
            continue
        if len(token) < 4:  # kata pendek diabaikan
            corrected.append(token)
            continue

        prev_word = corrected[i - 1] if i > 0 else "<s>"

        candidates = generate_candidates(token, vocabulary, max_edit)

        if not candidates:
            corrected.append(token)
            continue

        # Laplace smoothed probability P(word | prev_word) in (0, 1]
        # log(P) <= 0. Nilai yang lebih tinggi (lebih dekat ke 0) berarti probabilitas lebih besar.
        best_word = token
        best_log_prob = math.log(lm.get_prob(prev_word, token))

        # Hanya ganti jika kandidat memiliki probabilitas transisi yang secara signifikan lebih tinggi
        for cand in candidates:
            cand_log_prob = math.log(lm.get_prob(prev_word, cand))
            if cand_log_prob > best_log_prob + 2.0:
                best_log_prob = cand_log_prob
                best_word = cand

        corrected.append(best_word)

    return " ".join(corrected)

