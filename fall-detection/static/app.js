(() => {
  const WAVE_BARS = 48;
  const POLL_INTERVAL_MS = 1000;

  const dropzone = document.getElementById("dropzone");
  const dropzoneTitle = document.getElementById("dropzoneTitle");
  const fileInput = document.getElementById("fileInput");
  const errorMsg = document.getElementById("errorMsg");

  const readoutPanel = document.getElementById("readoutPanel");
  const readoutCount = document.getElementById("readoutCount");
  const readoutStatus = document.getElementById("readoutStatus");
  const waveform = document.getElementById("waveform");

  const resultPanel = document.getElementById("resultPanel");
  const resultBadgeV1 = document.getElementById("resultBadgeV1");
  const resultVideoV1 = document.getElementById("resultVideoV1");
  const eventsBoxV1 = document.getElementById("eventsV1");

  const resultBadgeV2 = document.getElementById("resultBadgeV2");
  const resultVideoV2 = document.getElementById("resultVideoV2");
  const eventsBoxV2 = document.getElementById("eventsV2");

  const resetBtn = document.getElementById("resetBtn");

  const statusDot = document.getElementById("statusDot");
  const stateLabel = document.getElementById("stateLabel");

  let pollTimer = null;

  // Build the waveform bars once.
  const bars = [];
  for (let i = 0; i < WAVE_BARS; i++) {
    const bar = document.createElement("div");
    bar.className = "wave-bar";
    waveform.appendChild(bar);
    bars.push(bar);
  }

  function setDotState(state) {
    statusDot.classList.remove("processing", "alert");
    if (state === "processing") statusDot.classList.add("processing");
    if (state === "alert") statusDot.classList.add("alert");
    stateLabel.textContent = state;
  }

  function paintWaveform(progressPct) {
    const activeCount = Math.round((progressPct / 100) * WAVE_BARS);
    bars.forEach((bar, i) => {
      const isActive = i < activeCount;
      bar.classList.toggle("active", isActive);
      const base = isActive ? 35 + Math.random() * 55 : 6;
      bar.style.height = `${base}%`;
    });
  }

  function resetWaveform() {
    bars.forEach((bar) => {
      bar.classList.remove("active");
      bar.style.height = "6%";
    });
  }

  function showError(message) {
    errorMsg.textContent = message;
    errorMsg.hidden = false;
    setDotState("idle");
  }

  function clearError() {
    errorMsg.hidden = true;
    errorMsg.textContent = "";
  }

  function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(1);
    return `${m}:${s.padStart(4, "0")}`;
  }

  async function uploadFile(file) {
    clearError();
    dropzoneTitle.textContent = file.name;

    const formData = new FormData();
    formData.append("file", file);

    readoutPanel.hidden = false;
    resultPanel.hidden = true;
    readoutStatus.textContent = "yuklanmoqda\u2026";
    setDotState("processing");
    resetWaveform();

    let response;
    try {
      response = await fetch("/analyze", { method: "POST", body: formData });
    } catch (err) {
      showError("Serverga ulanib bo'lmadi. Sahifani yangilab qayta urinib ko'ring.");
      readoutPanel.hidden = true;
      return;
    }

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      showError(detail.detail || `Xato: ${response.status}`);
      readoutPanel.hidden = true;
      return;
    }

    const { job_id: jobId } = await response.json();
    pollStatus(jobId);
  }

  function pollStatus(jobId) {
    if (pollTimer) clearInterval(pollTimer);

    pollTimer = setInterval(async () => {
      let job;
      try {
        const res = await fetch(`/status/${jobId}`);
        job = await res.json();
      } catch {
        return;
      }

      if (job.status === "queued" || job.status === "processing") {
        const pct = job.progress || 0;
        readoutCount.textContent = `${pct}%`;
        readoutStatus.textContent =
          pct < 50
            ? "v1 kadrlar tahlil qilinmoqda\u2026"
            : "v2 kadrlar tahlil qilinmoqda\u2026";
        paintWaveform(pct);
        return;
      }

      clearInterval(pollTimer);

      if (job.status === "error") {
        showError(job.error || "Qayta ishlashda xato yuz berdi.");
        readoutPanel.hidden = true;
        return;
      }

      if (job.status === "done") {
        readoutPanel.hidden = true;
        renderResult(job);
      }
    }, POLL_INTERVAL_MS);
  }

  function renderResult(job) {
    resultPanel.hidden = false;

    // --- Detector V1 ---
    const v1 = job.v1 || job;
    if (v1 && v1.video_url) {
      resultVideoV1.src = v1.video_url;
      if (v1.fall_detected) {
        resultBadgeV1.textContent = "FALL DETECTED";
        resultBadgeV1.classList.add("fall");
      } else {
        resultBadgeV1.textContent = "STANDING";
        resultBadgeV1.classList.remove("fall");
      }

      eventsBoxV1.innerHTML = "";
      (v1.fall_events || []).forEach((ev) => {
        const span = document.createElement("span");
        span.textContent = formatTime(ev.time_sec);
        span.style.cursor = "pointer";
        span.title = "Videoni shu vaqtga o'tkazish";
        span.addEventListener("click", () => {
          resultVideoV1.currentTime = ev.time_sec;
          resultVideoV1.play();
        });
        eventsBoxV1.appendChild(span);
      });
    }

    // --- Detector V2 ---
    const v2 = job.v2 || job;
    if (v2 && v2.video_url) {
      resultVideoV2.src = v2.video_url;
      if (v2.fall_detected) {
        resultBadgeV2.textContent = "FALL DETECTED";
        resultBadgeV2.classList.add("fall");
      } else {
        resultBadgeV2.textContent = "STANDING";
        resultBadgeV2.classList.remove("fall");
      }

      eventsBoxV2.innerHTML = "";
      (v2.fall_events || []).forEach((ev) => {
        const span = document.createElement("span");
        const trackStr = (ev.track_id !== undefined && ev.track_id !== null) ? `ID${ev.track_id}: ` : "";
        span.textContent = `${trackStr}${formatTime(ev.time_sec)}`;
        span.style.cursor = "pointer";
        span.title = "Videoni shu vaqtga o'tkazish";
        span.addEventListener("click", () => {
          resultVideoV2.currentTime = ev.time_sec;
          resultVideoV2.play();
        });
        eventsBoxV2.appendChild(span);
      });
    }

    // Global alert dot
    const anyFall = (v1 && v1.fall_detected) || (v2 && v2.fall_detected);
    if (anyFall) {
      setDotState("alert");
    } else {
      setDotState("idle");
    }
  }

  // ---- Dropzone interactions ------------------------------------------
  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );

  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) uploadFile(file);
  });

  resetBtn.addEventListener("click", () => {
    resultPanel.hidden = true;
    readoutPanel.hidden = true;
    resultVideoV1.src = "";
    resultVideoV2.src = "";
    dropzoneTitle.textContent = "Videoni shu yerga tashlang";
    fileInput.value = "";
    setDotState("idle");
    clearError();
  });
})();
