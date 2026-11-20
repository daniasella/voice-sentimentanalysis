/* =========================================================
   RECORD.JS - Voice Sentiment Analysis & Correction Evaluation
   ========================================================= */

/* =========================================================
   WAV RECORDING HELPERS (16kHz Mono PCM)
   Eliminates FFmpeg requirement on the server (100% Vercel compatible)
   ========================================================= */
function writeWavString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function downsampleBuffer(buffer, inputSampleRate, outputSampleRate = 16000) {
  if (outputSampleRate >= inputSampleRate) {
    return buffer;
  }
  const sampleRateRatio = inputSampleRate / outputSampleRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0, count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function encodeWAV(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeWavString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeWavString(view, 8, 'WAVE');
  writeWavString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // Mono (1 channel)
  view.setUint32(24, sampleRate, true); // 16000 Hz
  view.setUint32(28, sampleRate * 2, true); // Byte rate (16000 * 1 * 2)
  view.setUint16(32, 2, true); // Block align (1 * 2)
  view.setUint16(34, 16, true); // Bits per sample (16 bit)
  writeWavString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([view], { type: 'audio/wav' });
}

// Rekam audio selama durasi tertentu (ms) dan hasilkan WAV Blob 16kHz
async function recordWavForDuration(durationMs) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioContextClass();
  if (ctx.state === 'suspended') await ctx.resume();

  const source = ctx.createMediaStreamSource(stream);
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];

  processor.onaudioprocess = (e) => {
    chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };

  source.connect(processor);
  processor.connect(ctx.destination);

  return new Promise((resolve) => {
    setTimeout(async () => {
      processor.disconnect();
      source.disconnect();
      const inSampleRate = ctx.sampleRate;
      if (ctx.state !== 'closed') await ctx.close();
      stream.getTracks().forEach(t => t.stop());

      const totalLen = chunks.reduce((acc, c) => acc + c.length, 0);
      if (totalLen === 0) {
        resolve(null);
        return;
      }
      const full = new Float32Array(totalLen);
      let off = 0;
      for (const c of chunks) {
        full.set(c, off);
        off += c.length;
      }
      const down = downsampleBuffer(full, inSampleRate, 16000);
      resolve(encodeWAV(down, 16000));
    }, durationMs);
  });
}

// State variables
let isMainRecording = false;
let mainAudioStream = null;
let mainAudioContext = null;
let mainProcessor = null;
let mainPcmChunks = [];
let mainAnalyser = null;
let isSystemActive = true;
let standbyInterval = null;
let wakeLock = false;
let silenceTimer = null;
let mainRetryCount = 0;
let answerRetryCount = 0;
let currentHistoryId = null;

const MAX_MAIN_RETRY = 2;
const MAX_ANSWER_RETRY = 2;
const ANSWER_DURATION = 4000;
const SILENCE_LIMIT = 2500;
const SILENCE_THRESHOLD = 0.02;

// DOM Elements
const systemStatus = document.getElementById("systemStatus");
const recordBtn = document.getElementById("recordBtn");
const recordBtnText = document.getElementById("recordBtnText");
const stopBtn = document.getElementById("stopBtn");
const recordingFeedback = document.getElementById("recordingFeedback");
const wave = document.getElementById("wave");
const ttsPlayer = document.getElementById("ttsPlayer");

// STT Elements
const sttRawText = document.getElementById("sttRawText");
const sttNormalizedText = document.getElementById("sttNormalizedText");
const sttSentiment = document.getElementById("sttSentiment");
const sttModelScore = document.getElementById("sttModelScore");
const sttDuration = document.getElementById("sttDuration");
const sttExecTime = document.getElementById("sttExecTime");

// User Correction Elements
const userCorrectionInput = document.getElementById("userCorrectionInput");
const reanalyzeBtn = document.getElementById("reanalyzeBtn");
const reanalysisResult = document.getElementById("reanalysisResult");
const corrSentiment = document.getElementById("corrSentiment");
const corrModelScore = document.getElementById("corrModelScore");
const corrEditDistance = document.getElementById("corrEditDistance");
const sentimentShiftStatus = document.getElementById("sentimentShiftStatus");

// State UI helper
function setStatus(text, badgeClass) {
  if (!systemStatus) return;
  systemStatus.textContent = text;
  systemStatus.className = `status-badge ${badgeClass}`;
}

function updateSentimentBadge(element, sentiment) {
  if (!element) return;
  element.textContent = sentiment ? sentiment.toUpperCase() : "-";
  element.className = "badge-sentiment";
  const s = (sentiment || "").toLowerCase();
  if (s === "positif") {
    element.classList.add("badge-positif");
  } else if (s === "negatif") {
    element.classList.add("badge-negatif");
  } else if (s === "netral") {
    element.classList.add("badge-netral");
  } else {
    element.classList.add("badge-empty");
  }
}

/* =========================================================
   RECORDING CONTROLS
   ========================================================= */
recordBtn.onclick = () => {
  if (isMainRecording) return;
  isSystemActive = true;
  startMainRecording();
};

stopBtn.onclick = () => {
  if (isMainRecording) {
    stopMainRecording();
  }
};

async function startMainRecording() {
  try {
    setStatus("Merekam...", "status-recording");
    recordingFeedback.textContent = "Mendengarkan ucapan Anda... Bicaralah sekarang.";
    recordBtn.disabled = true;
    stopBtn.style.display = "inline-flex";
    wave.classList.add("recording");

    mainAudioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    mainAudioContext = new AudioContextClass();
    if (mainAudioContext.state === 'suspended') {
      await mainAudioContext.resume();
    }

    const source = mainAudioContext.createMediaStreamSource(mainAudioStream);
    mainAnalyser = mainAudioContext.createAnalyser();
    mainAnalyser.fftSize = 2048;
    source.connect(mainAnalyser);

    mainPcmChunks = [];
    mainProcessor = mainAudioContext.createScriptProcessor(4096, 1, 1);
    mainProcessor.onaudioprocess = (e) => {
      if (!isMainRecording) return;
      mainPcmChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };

    source.connect(mainProcessor);
    mainProcessor.connect(mainAudioContext.destination);

    isMainRecording = true;
    recordBtnText.textContent = "Merekam...";
    detectSilence(mainAnalyser);

  } catch (err) {
    console.error("Microphone access error:", err);
    setStatus("Gagal Akses Mic", "status-error");
    recordingFeedback.innerHTML = `<span style="color:#b91c1c;">Gagal mengakses mikrofon. Pastikan mikrofon terpasang dan izin browser diaktifkan.</span>`;
    recordBtn.disabled = false;
    stopBtn.style.display = "none";
    wave.classList.remove("recording");
  }
}

async function stopMainRecording() {
  if (!isMainRecording) return;
  isMainRecording = false;

  wave.classList.remove("recording");
  stopBtn.style.display = "none";
  recordBtn.disabled = false;
  recordBtnText.textContent = "Mulai Bicara";

  if (mainProcessor) {
    mainProcessor.disconnect();
    mainProcessor = null;
  }

  const sampleRate = mainAudioContext ? mainAudioContext.sampleRate : 44100;
  if (mainAudioContext && mainAudioContext.state !== "closed") {
    await mainAudioContext.close();
    mainAudioContext = null;
  }

  if (mainAudioStream) {
    mainAudioStream.getTracks().forEach(t => t.stop());
    mainAudioStream = null;
  }

  const totalLength = mainPcmChunks.reduce((acc, c) => acc + c.length, 0);
  if (totalLength === 0) {
    setStatus("Audio Kosong", "status-error");
    recordingFeedback.textContent = "Tidak ada audio yang tertangkap. Silakan coba lagi.";
    return;
  }

  const fullBuffer = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of mainPcmChunks) {
    fullBuffer.set(chunk, offset);
    offset += chunk.length;
  }
  mainPcmChunks = [];

  const downsampled = downsampleBuffer(fullBuffer, sampleRate, 16000);
  const wavBlob = encodeWAV(downsampled, 16000);

  await processMainAudio(wavBlob);
}

/* =========================================================
   SILENCE DETECTION
   ========================================================= */
function detectSilence(analyser) {
  if (!isMainRecording) return;
  const data = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(data);

  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    sum += Math.abs(data[i] - 128);
  }
  const volume = sum / data.length / 128;

  if (volume > SILENCE_THRESHOLD) {
    if (silenceTimer) {
      clearTimeout(silenceTimer);
      silenceTimer = null;
    }
  } else {
    if (!silenceTimer) {
      silenceTimer = setTimeout(() => {
        if (isMainRecording) {
          console.log("Auto-stopping recording due to silence detection");
          stopMainRecording();
        }
      }, SILENCE_LIMIT);
    }
  }

  if (isMainRecording) {
    requestAnimationFrame(() => detectSilence(analyser));
  }
}

/* =========================================================
   PROCESS AUDIO & INFERENCE
   ========================================================= */
async function processMainAudio(wavBlob) {
  if (!wavBlob) {
    setStatus("Audio Kosong", "status-error");
    recordingFeedback.textContent = "Tidak ada audio yang tertangkap. Silakan coba lagi.";
    return;
  }

  setStatus("Memproses...", "status-processing");
  recordingFeedback.textContent = "Mengonversi audio & menjalankan model analisis sentimen...";

  const formData = new FormData();
  formData.append("audio", wavBlob, "record.wav");

  try {
    const res = await fetch("/analyze", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      console.warn("Server analyze error:", data.error);
      setStatus("Gagal Proses", "status-error");
      recordingFeedback.innerHTML = `<span style="color:#b91c1c;">${data.error}</span>`;

      if (mainRetryCount < MAX_MAIN_RETRY) {
        mainRetryCount++;
        recordingFeedback.innerHTML += ` (Percobaan ulang otomatis ${mainRetryCount}/${MAX_MAIN_RETRY}...)`;
        setTimeout(() => startMainRecording(), 1800);
      } else {
        mainRetryCount = 0;
        recordingFeedback.innerHTML += `<br>Silakan periksa mikrofon dan tekan tombol "Mulai Bicara" lagi.`;
        startWakeWordStandby();
      }
      return;
    }

    mainRetryCount = 0;
    currentHistoryId = data.history_id || null;

    // Display STT Results
    sttRawText.textContent = data.text_raw || "(Tidak ada ucapan terdeteksi)";
    sttRawText.classList.remove("text-box-muted");
    sttNormalizedText.textContent = data.text_normalized || "-";
    updateSentimentBadge(sttSentiment, data.sentiment);
    sttModelScore.textContent = typeof data.model_score === "number" ? data.model_score.toFixed(4) : "-";
    sttDuration.textContent = data.audio_duration ? `${data.audio_duration} s` : "-";
    sttExecTime.textContent = data.execution_time ? `${data.execution_time} s` : "-";

    // Setup User Correction Area
    userCorrectionInput.value = data.text_raw || "";
    reanalyzeBtn.disabled = false;
    reanalysisResult.style.display = "none";

    setStatus("Selesai", "status-ready");
    recordingFeedback.textContent = "Analisis selesai. Anda dapat mengoreksi teks di panel sebelah kanan untuk menguji sensitivitas.";

    // Refresh history table
    loadSpeechHistory();

    // Play TTS feedback
    if (data.tts_audio) {
      ttsPlayer.src = "/static/" + data.tts_audio.split("static/").pop().replace(/^\/+/, '');
      await ttsPlayer.play().catch(e => console.log("TTS play interrupted"));

      ttsPlayer.onended = async () => {
        if (data.question_audio) {
          ttsPlayer.src = "/static/" + data.question_audio.split("static/").pop().replace(/^\/+/, '');
          await ttsPlayer.play().catch(e => console.log("Question TTS play interrupted"));
          ttsPlayer.onended = () => {
            answerRetryCount = 0;
            startAnswerRecording();
          };
        }
      };
    }

  } catch (err) {
    console.error("Network / server error:", err);
    setStatus("Error Jaringan", "status-error");
    recordingFeedback.textContent = "Gagal menghubungi server. Pastikan server Flask berjalan.";
  }
}

/* =========================================================
   USER CORRECTION & SENSITIVITY REANALYSIS
   ========================================================= */
reanalyzeBtn.onclick = async () => {
  const raw = sttRawText.textContent.trim();
  const corrected = userCorrectionInput.value.trim();

  if (!corrected) {
    alert("Teks koreksi tidak boleh kosong.");
    return;
  }

  setStatus("Memproses Koreksi...", "status-processing");
  reanalyzeBtn.disabled = true;

  try {
    const res = await fetch("/reanalyze_text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        raw_text: raw,
        corrected_text: corrected,
        history_id: currentHistoryId
      })
    });

    const data = await res.json();

    if (data.error) {
      alert("Error analisis koreksi: " + data.error);
      setStatus("Selesai", "status-ready");
      reanalyzeBtn.disabled = false;
      return;
    }

    // Display Reanalysis Result
    reanalysisResult.style.display = "flex";
    updateSentimentBadge(corrSentiment, data.sentiment);
    corrModelScore.textContent = typeof data.model_score === "number" ? data.model_score.toFixed(4) : "-";

    const editPercent = (data.word_edit_ratio * 100).toFixed(1);
    corrEditDistance.textContent = `${data.word_edit_distance} kata (${editPercent}%)`;

    // Compare with original STT sentiment
    const originalSent = sttSentiment.textContent.toLowerCase();
    const newSent = (data.sentiment || "").toLowerCase();

    if (originalSent === newSent) {
      sentimentShiftStatus.innerHTML = `<span style="color:#059669; font-weight:600;">✓ Sentimen Tetap (${newSent})</span>`;
    } else {
      sentimentShiftStatus.innerHTML = `<span style="color:#dc2626; font-weight:600;">⚡ Sentimen Berubah: <em>${originalSent}</em> &rarr; <em>${newSent}</em></span>`;
    }

    setStatus("Evaluasi Koreksi Selesai", "status-ready");
    reanalyzeBtn.disabled = false;
    loadSpeechHistory();

  } catch (err) {
    console.error("Error during reanalysis:", err);
    alert("Terjadi kesalahan saat menghubungi server.");
    setStatus("Error", "status-error");
    reanalyzeBtn.disabled = false;
  }
};

/* =========================================================
   ANSWER RECORDING (YA / TIDAK) WITH BOUNDED RETRIES
   ========================================================= */
async function startAnswerRecording() {
  if (!isSystemActive) return;

  recordingFeedback.textContent = "Menunggu jawaban Anda (katakan 'Ya' untuk lanjut atau 'Tidak' untuk selesai)...";

  try {
    const wavBlob = await recordWavForDuration(ANSWER_DURATION);
    if (!wavBlob) {
      startWakeWordStandby();
      return;
    }

    const fd = new FormData();
    fd.append("audio", wavBlob, "answer.wav");

    try {
      const res = await fetch("/analyze_answer", { method: "POST", body: fd });
      const data = await res.json();

      if (data.action === "restart") {
        const promptRes = await fetch("/tts_silakan");
        const prompt = await promptRes.json();
        ttsPlayer.src = "/static/" + prompt.filename.split("static/").pop().replace(/^\/+/, '');
        await ttsPlayer.play().catch(e => console.log("TTS play"));
        ttsPlayer.onended = () => {
          recordBtn.click();
        };
        return;
      }

      if (data.action === "exit") {
        isSystemActive = false;
        if (data.tts_audio) {
          ttsPlayer.src = "/static/" + data.tts_audio.split("static/").pop().replace(/^\/+/, '');
          await ttsPlayer.play().catch(e => console.log("Closing TTS"));
        }
        recordingFeedback.innerHTML = `Terima kasih! Sesi selesai. Ucapkan <strong>'Halo Neo'</strong> untuk memulai lagi.`;
        startWakeWordStandby();
        return;
      }

      // Unrecognized answer -> bounded retry
      answerRetryCount++;
      if (answerRetryCount <= MAX_ANSWER_RETRY) {
        recordingFeedback.textContent = `Jawaban tidak jelas (${answerRetryCount}/${MAX_ANSWER_RETRY}). Silakan katakan 'Ya' atau 'Tidak'.`;
        setTimeout(() => startAnswerRecording(), 800);
      } else {
        recordingFeedback.textContent = "Sesi selesai. Tekan 'Mulai Bicara' atau ucapkan 'Halo Neo' untuk sesi baru.";
        startWakeWordStandby();
      }

    } catch (err) {
      console.error("Error analyzing answer:", err);
      startWakeWordStandby();
    }
  } catch (err) {
    console.error("Microphone access in answer recording failed:", err);
    startWakeWordStandby();
  }
}

/* =========================================================
   WAKE WORD STANDBY
   ========================================================= */
function startWakeWordStandby() {
  if (standbyInterval) return;

  setStatus("Standby (Wake Word)", "status-standby");
  console.log("Standby mode: Listening for wake word 'Halo Neo'...");

  standbyInterval = setInterval(async () => {
    if (isSystemActive || wakeLock || isMainRecording) return;

    try {
      const wavBlob = await recordWavForDuration(3000);
      if (!wavBlob || isSystemActive || wakeLock || isMainRecording) return;

      const fd = new FormData();
      fd.append("audio", wavBlob, "wake.wav");

      try {
        const res = await fetch("/wake_word", { method: "POST", body: fd });
        const data = await res.json();

        if (data.wake === true && !wakeLock) {
          wakeLock = true;
          clearInterval(standbyInterval);
          standbyInterval = null;

          isSystemActive = true;
          setStatus("Wake Word Terdeteksi", "status-ready");
          recordingFeedback.textContent = "Wake word terdeteksi! Mempersiapkan perekaman...";

          const resTTS = await fetch("/tts_silakan");
          const tts = await resTTS.json();
          ttsPlayer.src = "/static/" + tts.filename.split("static/").pop().replace(/^\/+/, '');
          await ttsPlayer.play().catch(e => console.log("Wake prompt TTS"));

          ttsPlayer.onended = () => {
            wakeLock = false;
            startMainRecording();
          };
        }
      } catch (e) {
        console.log("Wake word check failed:", e);
      }
    } catch (err) {
      console.warn("Mic access in standby failed:", err);
      clearInterval(standbyInterval);
      standbyInterval = null;
    }
  }, 3500);
}

/* =========================================================
   NEON DB SPEECH HISTORY & EXPORT DASHBOARD
   ========================================================= */
const historyTableBody = document.getElementById("historyTableBody");
const historySearchInput = document.getElementById("historySearchInput");
const sentimentFilter = document.getElementById("sentimentFilter");
const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");
const exportExcelBtn = document.getElementById("exportExcelBtn");
const exportPdfBtn = document.getElementById("exportPdfBtn");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");

const hstatTotal = document.getElementById("hstatTotal");
const hstatPos = document.getElementById("hstatPos");
const hstatNeg = document.getElementById("hstatNeg");
const hstatNeu = document.getElementById("hstatNeu");

let searchDebounceTimeout = null;

async function loadSpeechHistory() {
  if (!historyTableBody) return;

  const sentiment = sentimentFilter ? sentimentFilter.value : "";
  const query = historySearchInput ? historySearchInput.value.trim() : "";

  const params = new URLSearchParams();
  if (sentiment) params.append("sentiment", sentiment);
  if (query) params.append("q", query);

  try {
    const res = await fetch(`/api/history?${params.toString()}`);
    const data = await res.json();

    if (data.status === "success") {
      renderHistoryTable(data.data || []);
      updateHistoryStats(data.data || []);
    } else {
      historyTableBody.innerHTML = `<tr><td colspan="9" class="table-empty" style="color:#b91c1c;">Gagal memuat riwayat: ${data.error || "Unknown error"}</td></tr>`;
    }
  } catch (err) {
    console.error("Error loading speech history:", err);
    historyTableBody.innerHTML = `<tr><td colspan="9" class="table-empty" style="color:#b91c1c;">Gagal menghubungi database Neon. Pastikan koneksi internet aktif.</td></tr>`;
  }
}

function updateHistoryStats(records) {
  if (!hstatTotal) return;
  const total = records.length;
  let pos = 0, neg = 0, neu = 0;
  records.forEach(r => {
    const s = (r.sentiment || "").toLowerCase();
    if (s === "positif") pos++;
    else if (s === "negatif") neg++;
    else if (s === "netral") neu++;
  });

  hstatTotal.textContent = total;
  if (hstatPos) hstatPos.textContent = pos;
  if (hstatNeg) hstatNeg.textContent = neg;
  if (hstatNeu) hstatNeu.textContent = neu;
}

function escapeHtml(text) {
  if (!text) return "";
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function renderHistoryTable(records) {
  if (!historyTableBody) return;

  if (records.length === 0) {
    historyTableBody.innerHTML = `<tr><td colspan="9" class="table-empty">Belum ada rekaman ucapan tersimpan di database Neon.</td></tr>`;
    return;
  }

  historyTableBody.innerHTML = records.map((r, idx) => {
    const time = r.created_at_formatted || "-";
    const raw = escapeHtml(r.text_raw);
    const norm = escapeHtml(r.text_normalized);
    const corr = r.user_corrected_text ? escapeHtml(r.user_corrected_text) : "";
    const sent = (r.sentiment || "-").toLowerCase();
    const corrSent = (r.corrected_sentiment || "").toLowerCase();
    const score = typeof r.model_score === "number" ? r.model_score.toFixed(4) : "-";
    const dur = typeof r.audio_duration === "number" ? `${r.audio_duration.toFixed(2)}s` : "-";
    const lat = typeof r.execution_time === "number" ? `${r.execution_time.toFixed(3)}s` : "-";

    let badgeClass = "badge-table-netral";
    if (sent === "positif") badgeClass = "badge-table-positif";
    else if (sent === "negatif") badgeClass = "badge-table-negatif";

    let normDisplay = norm || "-";
    if (corr) {
      let corrBadge = "";
      if (corrSent) {
        let cbClass = corrSent === "positif" ? "badge-table-positif" : (corrSent === "negatif" ? "badge-table-negatif" : "badge-table-netral");
        corrBadge = ` <span class="badge-table ${cbClass}">${corrSent.toUpperCase()}</span>`;
      }
      normDisplay += `<div class="text-meta" style="color:var(--accent); margin-top:4px;"><strong>Koreksi:</strong> "${corr}"${corrBadge}</div>`;
    }

    return `
      <tr id="history-row-${r.id}">
        <td><strong>${idx + 1}</strong></td>
        <td style="white-space:nowrap; font-size:12px; color:var(--text-secondary);">${time}</td>
        <td>${raw}</td>
        <td>${normDisplay}</td>
        <td><span class="badge-table ${badgeClass}">${sent.toUpperCase()}</span></td>
        <td style="font-family:monospace; font-size:12px;">${score}</td>
        <td style="font-size:12px;">${dur}</td>
        <td style="font-size:12px;">${lat}</td>
        <td style="text-align:center;">
          <button class="btn-table-del" onclick="deleteHistoryRecord(${r.id})" title="Hapus record ini">🗑️</button>
        </td>
      </tr>
    `;
  }).join("");
}

async function deleteHistoryRecord(id) {
  if (!confirm(`Hapus rekaman #${id} dari database Neon?`)) return;

  try {
    const res = await fetch(`/api/history/${id}`, { method: "DELETE" });
    const data = await res.json();
    if (data.status === "success") {
      loadSpeechHistory();
    } else {
      alert("Gagal menghapus data: " + (data.error || data.message));
    }
  } catch (err) {
    console.error("Error deleting history record:", err);
    alert("Terjadi kesalahan saat menghapus data.");
  }
}

// Global scope access for delete button onclick
window.deleteHistoryRecord = deleteHistoryRecord;

if (clearHistoryBtn) {
  clearHistoryBtn.onclick = async () => {
    if (!confirm("Apakah Anda yakin ingin MENGHAPUS SEMUA riwayat analisis dari database Neon? Tindakan ini tidak dapat dibatalkan.")) return;

    try {
      const res = await fetch("/api/history/clear", { method: "POST" });
      const data = await res.json();
      if (data.status === "success") {
        loadSpeechHistory();
      } else {
        alert("Gagal membersihkan riwayat: " + (data.error || data.message));
      }
    } catch (err) {
      console.error("Error clearing history:", err);
      alert("Gagal menghubungi server untuk menghapus riwayat.");
    }
  };
}

if (refreshHistoryBtn) {
  refreshHistoryBtn.onclick = () => {
    loadSpeechHistory();
  };
}

if (exportExcelBtn) {
  exportExcelBtn.onclick = () => {
    const sentiment = sentimentFilter ? sentimentFilter.value : "";
    const query = historySearchInput ? historySearchInput.value.trim() : "";
    const params = new URLSearchParams();
    if (sentiment) params.append("sentiment", sentiment);
    if (query) params.append("q", query);
    window.location.href = `/export/excel?${params.toString()}`;
  };
}

if (exportPdfBtn) {
  exportPdfBtn.onclick = () => {
    const sentiment = sentimentFilter ? sentimentFilter.value : "";
    const query = historySearchInput ? historySearchInput.value.trim() : "";
    const params = new URLSearchParams();
    if (sentiment) params.append("sentiment", sentiment);
    if (query) params.append("q", query);
    window.location.href = `/export/pdf?${params.toString()}`;
  };
}

if (sentimentFilter) {
  sentimentFilter.onchange = () => {
    loadSpeechHistory();
  };
}

if (historySearchInput) {
  historySearchInput.oninput = () => {
    clearTimeout(searchDebounceTimeout);
    searchDebounceTimeout = setTimeout(() => {
      loadSpeechHistory();
    }, 300);
  };
}

// Initial load on page ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", loadSpeechHistory);
} else {
  loadSpeechHistory();
}

