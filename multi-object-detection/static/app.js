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
  const resultBadge = document.getElementById("resultBadge");
  const resultVideo = document.getElementById("resultVideo");
  const eventsBox = document.getElementById("eventsBox");

  const resetBtn = document.getElementById("resetBtn");

  const statusDot = document.getElementById("statusDot");
  const stateLabel = document.getElementById("stateLabel");

  let pollTimer = null;

  // 48 ta waveform ustunlarini yaratamiz
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
        readoutStatus.textContent = "ONNX modellari videoni tahlil qilmoqda\u2026";
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

    if (job && job.video_url) {
      resultVideo.src = job.video_url;

      const hasFire = job.fire_detected;
      const hasSmoke = job.smoke_detected;
      const hasFall = job.fall_detected;

      const alerts = [];
      if (hasFire) alerts.push("FIRE");
      if (hasSmoke) alerts.push("SMOKE");
      if (hasFall) alerts.push("FALL");

      if (alerts.length > 0) {
        resultBadge.textContent = `${alerts.join(" & ")} DETECTED`;
        resultBadge.classList.add("alert");
        setDotState("alert");
      } else {
        resultBadge.textContent = "CLEAR";
        resultBadge.classList.remove("alert");
        setDotState("idle");
      }

      eventsBox.innerHTML = "";
      (job.events || []).forEach((ev) => {
        const span = document.createElement("span");
        if (ev.type === "FALL DETECTED") {
          span.classList.add("fall-event");
        }
        const confStr = (ev.confidence && ev.type !== "FALL DETECTED") ? ` (${Math.round(ev.confidence * 100)}%)` : "";
        span.textContent = `[${ev.type}] ${formatTime(ev.time_sec)}${confStr}`;
        span.style.cursor = "pointer";
        span.title = "Videoni shu vaqtga o'tkazish";
        span.addEventListener("click", () => {
          resultVideo.currentTime = ev.time_sec;
          resultVideo.play();
        });
        eventsBox.appendChild(span);
      });
    }
  }

  // ---- Dropzone hodisalari ------------------------------------------
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
    resultVideo.src = "";
    dropzoneTitle.textContent = "Videoni shu yerga tashlang";
    fileInput.value = "";
    setDotState("idle");
    clearError();
  });
})();
