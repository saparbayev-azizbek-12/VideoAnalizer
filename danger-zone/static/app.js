// app.js - Danger Zone Detection Web Client
let serverUrl = "http://localhost:8000";
let currentVideoId = null;
let videoMeta = null;
let currentFrameIdx = 0;
let points = []; // [[x, y], ...]
let canvasImage = new Image();
let pollingInterval = null;

// DOM Elements
const serverUrlInput = document.getElementById("serverUrl");
const checkConnBtn = document.getElementById("checkConnBtn");
const gpuStatus = document.getElementById("gpuStatus");

const uploadBox = document.getElementById("uploadBox");
const videoFileInput = document.getElementById("videoFileInput");
const videoMetaBox = document.getElementById("videoMeta");

const frameSeekerSection = document.getElementById("frameSeekerSection");
const frameSlider = document.getElementById("frameSlider");
const currentFrameLabel = document.getElementById("currentFrameLabel");
const maxFrameLabel = document.getElementById("maxFrameLabel");
const btnPrevFrame = document.getElementById("btnPrevFrame");
const btnNextFrame = document.getElementById("btnNextFrame");

const modelSelect = document.getElementById("modelSelect");
const confRange = document.getElementById("confRange");
const confVal = document.getElementById("confVal");
const startProcessBtn = document.getElementById("startProcessBtn");

const zoneCanvas = document.getElementById("zoneCanvas");
const ctx = zoneCanvas.getContext("2d");
const canvasPlaceholder = document.getElementById("canvasPlaceholder");
const btnUndoPoint = document.getElementById("btnUndoPoint");
const btnClearPoints = document.getElementById("btnClearPoints");
const pointCount = document.getElementById("pointCount");
const pointsList = document.getElementById("pointsList");

const monitoringSection = document.getElementById("monitoringSection");
const alertBanner = document.getElementById("alertBanner");
const progressBarFill = document.getElementById("progressBarFill");
const progressText = document.getElementById("progressText");
const statStatus = document.getElementById("statStatus");
const statFrames = document.getElementById("statFrames");
const statBreaches = document.getElementById("statBreaches");
const statPeopleInZone = document.getElementById("statPeopleInZone");

const resultActions = document.getElementById("resultActions");
const downloadVideoBtn = document.getElementById("downloadVideoBtn");
const downloadCsvBtn = document.getElementById("downloadCsvBtn");

// Initialize
document.addEventListener("DOMContentLoaded", () => {
    // URL auto set
    if (window.location.origin.startsWith("http")) {
        serverUrl = window.location.origin;
        serverUrlInput.value = serverUrl;
    }
    checkConnection();
    setupEventListeners();
});

function setupEventListeners() {
    checkConnBtn.addEventListener("click", checkConnection);

    // Upload box click
    uploadBox.addEventListener("click", () => videoFileInput.click());
    videoFileInput.addEventListener("change", handleFileSelect);

    // Drag and drop
    uploadBox.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = "#3b82f6";
    });
    uploadBox.addEventListener("dragleave", () => {
        uploadBox.style.borderColor = "rgba(59, 130, 246, 0.4)";
    });
    uploadBox.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = "rgba(59, 130, 246, 0.4)";
        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });

    // Seeker slider & buttons
    frameSlider.addEventListener("input", (e) => {
        currentFrameIdx = parseInt(e.target.value);
        loadFrame(currentFrameIdx);
    });

    btnPrevFrame.addEventListener("click", () => {
        if (!videoMeta) return;
        const step = Math.round((videoMeta.fps || 25) * 5);
        currentFrameIdx = Math.max(0, currentFrameIdx - step);
        frameSlider.value = currentFrameIdx;
        loadFrame(currentFrameIdx);
    });

    btnNextFrame.addEventListener("click", () => {
        if (!videoMeta) return;
        const step = Math.round((videoMeta.fps || 25) * 5);
        currentFrameIdx = Math.min(videoMeta.total_frames - 1, currentFrameIdx + step);
        frameSlider.value = currentFrameIdx;
        loadFrame(currentFrameIdx);
    });

    // Confidence threshold slider
    confRange.addEventListener("input", (e) => {
        confVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Canvas click event for points
    zoneCanvas.addEventListener("click", (e) => {
        if (!videoMeta) return;
        const rect = zoneCanvas.getBoundingClientRect();
        const scaleX = zoneCanvas.width / rect.width;
        const scaleY = zoneCanvas.height / rect.height;

        const x = Math.round((e.clientX - rect.left) * scaleX);
        const y = Math.round((e.clientY - rect.top) * scaleY);

        points.push([x, y]);
        updateCanvas();
        updatePointsInfo();
    });

    btnUndoPoint.addEventListener("click", () => {
        points.pop();
        updateCanvas();
        updatePointsInfo();
    });

    btnClearPoints.addEventListener("click", () => {
        points = [];
        updateCanvas();
        updatePointsInfo();
    });

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
        if (e.key === "z" || e.key === "Z") {
            points.pop();
            updateCanvas();
            updatePointsInfo();
        } else if (e.key === "r" || e.key === "R") {
            points = [];
            updateCanvas();
            updatePointsInfo();
        }
    });

    // Process Start
    startProcessBtn.addEventListener("click", startProcessing);
}

// 1. Check Server Connection & GPU Status
async function checkConnection() {
    serverUrl = serverUrlInput.value.trim().replace(/\/$/, "");
    const statusText = gpuStatus.querySelector(".status-text");

    try {
        const resp = await fetch(`${serverUrl}/api/health`);
        if (!resp.ok) throw new Error("API Javob bermadi");
        const data = await resp.json();

        gpuStatus.className = "status-badge online";
        statusText.textContent = `Server Onlayn (${data.gpu_name})`;
        return true;
    } catch (err) {
        gpuStatus.className = "status-badge offline";
        statusText.textContent = "Serverga ulanib bo'lmadi";
        return false;
    }
}

// 2. Upload Video
function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        uploadFile(e.target.files[0]);
    }
}

async function uploadFile(file) {
    if (!await checkConnection()) {
        alert("Avval superkompyuter serveriga ulaning!");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    uploadBox.innerHTML = `<div class="upload-icon">⏳</div><p>Video serverga yuklanmoqda...</p>`;

    try {
        const resp = await fetch(`${serverUrl}/api/upload`, {
            method: "POST",
            body: formData,
        });

        if (!resp.ok) {
            const errData = await resp.json();
            throw new Error(errData.detail || "Video yuklashda xatolik");
        }

        videoMeta = await resp.json();
        currentVideoId = videoMeta.video_id;

        // Render meta info
        document.getElementById("metaFilename").textContent = videoMeta.filename;
        document.getElementById("metaResolution").textContent = `${videoMeta.width}x${videoMeta.height}`;
        document.getElementById("metaDuration").textContent = `${videoMeta.duration_seconds}s`;
        document.getElementById("metaFrames").textContent = videoMeta.total_frames;
        videoMetaBox.classList.remove("hidden");

        // Set slider range
        frameSlider.min = 0;
        frameSlider.max = videoMeta.total_frames - 1;
        frameSlider.value = 0;
        maxFrameLabel.textContent = videoMeta.total_frames - 1;
        frameSeekerSection.classList.remove("hidden");

        uploadBox.innerHTML = `
            <div class="upload-icon">✅</div>
            <p><strong>${videoMeta.filename}</strong> yuklandi</p>
            <span class="upload-hint">Boshqa fayl tanlash uchun bosing</span>
        `;

        // Canvas dimensions setup
        zoneCanvas.width = videoMeta.width;
        zoneCanvas.height = videoMeta.height;
        canvasPlaceholder.classList.add("hidden");

        // Load 0-frame
        currentFrameIdx = 0;
        loadFrame(0);

    } catch (err) {
        alert(`Xatolik: ${err.message}`);
        uploadBox.innerHTML = `
            <div class="upload-icon">📁</div>
            <p>Videoni shu yerga tashlang yoki <strong>fayl tanlang</strong></p>
            <span class="upload-hint">MP4, AVI, MOV, MKV formatlar</span>
        `;
    }
}

// 3. Load Video Frame
function loadFrame(frameIdx) {
    if (!currentVideoId) return;

    currentFrameLabel.textContent = frameIdx;
    canvasImage.src = `${serverUrl}/api/video/${currentVideoId}/frame/${frameIdx}`;
    canvasImage.onload = () => {
        updateCanvas();
    };
}

// 4. Update Canvas with polygon overlay
function updateCanvas() {
    if (!canvasImage.src) return;

    ctx.clearRect(0, 0, zoneCanvas.width, zoneCanvas.height);
    ctx.drawImage(canvasImage, 0, 0, zoneCanvas.width, zoneCanvas.height);

    if (points.length === 0) return;

    // Draw lines & vertices
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = Math.max(2, Math.round(zoneCanvas.width / 400));
    ctx.fillStyle = "#ef4444";

    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i][0], points[i][1]);
    }

    if (points.length >= 3) {
        ctx.closePath();
        ctx.fillStyle = "rgba(239, 68, 68, 0.25)";
        ctx.fill();
    }

    ctx.stroke();

    // Draw vertex dots and numbers
    points.forEach((pt, idx) => {
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], Math.max(5, Math.round(zoneCanvas.width / 200)), 0, 2 * Math.PI);
        ctx.fillStyle = "#ef4444";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = "#ffffff";
        ctx.font = `bold ${Math.max(14, Math.round(zoneCanvas.width / 90))}px Inter`;
        ctx.fillText(`${idx + 1}`, pt[0] + 10, pt[1] - 10);
    });
}

function updatePointsInfo() {
    pointCount.textContent = points.length;
    startProcessBtn.disabled = points.length < 3 || !currentVideoId;

    if (points.length === 0) {
        pointsList.innerHTML = "Hech qanday nuqta belgilanmadi";
    } else {
        pointsList.innerHTML = points
            .map((p, i) => `<span class="chip">#${i + 1}: [${p[0]}, ${p[1]}]</span>`)
            .join("");
    }
}

// 5. Start Processing on Supercomputer
async function startProcessing() {
    if (!currentVideoId || points.length < 3) {
        alert("Iltimos, video yuklang va kamida 3 nuqtali xavfli hudud belgilang!");
        return;
    }

    startProcessBtn.disabled = true;
    startProcessBtn.innerHTML = `⏳ Superkompyuterga topshiriq yuborilmoqda...`;

    const payload = {
        video_id: currentVideoId,
        polygon: points,
        model_name: modelSelect.value,
        conf_threshold: parseFloat(confRange.value),
    };

    try {
        const resp = await fetch(`${serverUrl}/api/process`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "Topshiriq yuborishda xatolik");
        }

        const data = await resp.json();
        const taskId = data.task_id;

        monitoringSection.classList.remove("hidden");
        resultActions.classList.add("hidden");
        alertBanner.classList.add("hidden");

        // Start Polling
        if (pollingInterval) clearInterval(pollingInterval);
        pollingInterval = setInterval(() => pollTaskStatus(taskId), 1000);

    } catch (err) {
        alert(`Xatolik: ${err.message}`);
        startProcessBtn.disabled = false;
        startProcessBtn.innerHTML = `⚡ Superkompyuterda Qayta Ishlashni Boshlash`;
    }
}

// 6. Poll Task Status
async function pollTaskStatus(taskId) {
    try {
        const resp = await fetch(`${serverUrl}/api/tasks/${taskId}`);
        if (!resp.ok) return;
        const task = await resp.json();

        // Update Progress UI
        const progress = task.progress || 0;
        progressBarFill.style.width = `${progress}%`;
        progressText.textContent = `${progress}%`;

        statStatus.textContent = task.status;
        statFrames.textContent = `${task.current_frame} / ${task.total_frames}`;
        statBreaches.textContent = task.breach_frames || 0;
        statPeopleInZone.textContent = task.people_in_zone || 0;

        if (task.recent_breach) {
            alertBanner.classList.remove("hidden");
        } else {
            alertBanner.classList.add("hidden");
        }

        if (task.status === "completed") {
            clearInterval(pollingInterval);
            startProcessBtn.disabled = false;
            startProcessBtn.innerHTML = `⚡ Superkompyuterda Qayta Ishlashni Boshlash`;

            downloadVideoBtn.href = `${serverUrl}${task.output_video}`;
            downloadCsvBtn.href = `${serverUrl}${task.output_log}`;
            resultActions.classList.remove("hidden");
            alertBanner.classList.add("hidden");
            statStatus.textContent = "Tayyor!";
        } else if (task.status === "failed") {
            clearInterval(pollingInterval);
            startProcessBtn.disabled = false;
            startProcessBtn.innerHTML = `⚡ Superkompyuterda Qayta Ishlashni Boshlash`;
            alert(`Xatolik yuz berdi: ${task.error}`);
        }
    } catch (err) {
        console.error("Polling error:", err);
    }
}
