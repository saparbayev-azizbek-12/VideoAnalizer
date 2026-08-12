from __future__ import annotations
import csv
import json
import os
import queue
import sys
import threading
import time
import uuid
import urllib.request
import urllib.parse
import urllib.error
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional
import cv2
import numpy as np
from PIL import Image, ImageTk

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, os.path.dirname(_PKG_DIR))

from multi_object_detector import config
from multi_object_detector.analysis.frame_filter import StreamCorruptionFilter

BG = "#1a1a2e"
BG2 = "#16213e"
CARD = "#0f3460"
ACCENT = "#e94560"
ACCENT2 = "#533483"
FG = "#eaeaea"
FG2 = "#a0a0c0"
GREEN = "#00d26a"
YELLOW = "#f5c518"
RED = "#ff4757"

FONT_TITLE = ("Segoe UI", 14, "bold")
FONT_LABEL = ("Segoe UI", 9)
FONT_SMALL = ("Segoe UI", 8)
FONT_MONO = ("Consolas", 9)

PREVIEW_W = 640
PREVIEW_H = 380
DEFAULT_SERVER_URL = "http://localhost:8001"

MODEL_INFO = {
    "fire": {"label": "🔥 Yong'in / Tutun", "slow": False},
    "fall": {"label": "🚨 Yiqilib tushish", "slow": True},
    "ppe":  {"label": "🦺 PPE / Xavfsizlik", "slow": False},
}


def _sec_to_hms(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _analyze_frame_multi(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    models_enabled: dict[str, bool],
    _fire_model=None,
    _ppe_model=None,
    _fall_model=None,
    _fall_pose=None,
    _fall_track_states=None,
) -> tuple[np.ndarray, list[dict]]:
    from multi_object_detector.analysis.detectors import fire_detector, ppe_detector, fall_detector
    annotated = frame.copy()
    events = []
    ts = frame_idx / max(fps, 1.0)
    h_f, w_f = annotated.shape[:2]

    # 1. Fire / Smoke
    if models_enabled.get("fire") and _fire_model is not None:
        try:
            ann, det_events, has_fire, has_smoke = fire_detector.detect_fire_frame(
                annotated, model=_fire_model,
                conf_threshold=config.VIT_FIRE_CONF_THRESHOLD,
                fire_conf_threshold=config.VIT_FIRE_CONF_THRESHOLD,
                min_color_ratio=fire_detector._MIN_FIRE_PIXEL_RATIO,
            )
            annotated = ann
            if has_fire:
                events.append({"frame": frame_idx, "ts": ts, "model": "fire", "label": "FIRE", "conf": None})
            if has_smoke:
                events.append({"frame": frame_idx, "ts": ts, "model": "fire", "label": "SMOKE", "conf": None})
        except Exception:
            pass

    # 2. PPE Detection
    if models_enabled.get("ppe") and _ppe_model is not None:
        try:
            ann, violations, has_v = ppe_detector.analyze_ppe_frame(
                annotated, model=_ppe_model, conf=config.PPE_CONF_THRESHOLD
            )
            annotated = ann
            for v in violations:
                events.append({
                    "frame": frame_idx, "ts": ts, "model": "ppe",
                    "label": "PPE_VIOLATION",
                    "conf": round(v.get("confidence", 0), 3),
                })
        except Exception:
            pass

    # 3. Fall Detection
    if models_enabled.get("fall") and _fall_model is not None and _fall_pose is not None:
        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        any_fall_detected = False
        try:
            results_det = _fall_model.track(
                annotated, persist=True, classes=[0],
                tracker="bytetrack.yaml", verbose=False
            )
            for result in results_det:
                boxes = result.boxes
                if boxes.id is None:
                    continue
                for bbox, track_id_t in zip(boxes.xyxy, boxes.id):
                    track_id = int(track_id_t)
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, w_f), min(y2, h_f)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    bw, bh = x2 - x1, y2 - y1
                    if max(bw, bh) < 30 or (bw * bh) < 500:
                        continue

                    state = _fall_track_states.setdefault(track_id, fall_detector.TrackState())
                    state.last_seen_frame = frame_idx
                    ar = bh / max(bw, 1)

                    # Padded crop for MediaPipe context
                    pad_x = int(bw * 0.20)
                    pad_y = int(bh * 0.20)
                    cx1 = max(0, x1 - pad_x)
                    cy1 = max(0, y1 - pad_y)
                    cx2 = min(w_f, x2 + pad_x)
                    cy2 = min(h_f, y2 + pad_y)
                    crop = annotated[cy1:cy2, cx1:cx2].copy()
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pr = _fall_pose.process(crop_rgb)

                    posture = "Unknown"
                    smoothed_angle = 0.0

                    if pr.pose_landmarks:
                        lms = pr.pose_landmarks.landmark
                        cw = cx2 - cx1
                        ch = cy2 - cy1
                        l_sh = lms[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                        r_sh = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                        l_hip = lms[mp_pose.PoseLandmark.LEFT_HIP.value]
                        r_hip = lms[mp_pose.PoseLandmark.RIGHT_HIP.value]

                        sh_vis = max(l_sh.visibility, r_sh.visibility)
                        hip_vis = max(l_hip.visibility, r_hip.visibility)

                        if sh_vis >= 0.20 and hip_vis >= 0.20:
                            sh_x = (l_sh.x if l_sh.visibility >= r_sh.visibility else r_sh.x) * cw
                            sh_y = (l_sh.y if l_sh.visibility >= r_sh.visibility else r_sh.y) * ch
                            if l_sh.visibility >= 0.20 and r_sh.visibility >= 0.20:
                                sh_x = (l_sh.x + r_sh.x) * 0.5 * cw
                                sh_y = (l_sh.y + r_sh.y) * 0.5 * ch

                            hip_x = (l_hip.x if l_hip.visibility >= r_hip.visibility else r_hip.x) * cw
                            hip_y = (l_hip.y if l_hip.visibility >= r_hip.visibility else r_hip.y) * ch
                            if l_hip.visibility >= 0.20 and r_hip.visibility >= 0.20:
                                hip_x = (l_hip.x + r_hip.x) * 0.5 * cw
                                hip_y = (l_hip.y + r_hip.y) * 0.5 * ch

                            dy = hip_y - sh_y
                            dx = hip_x - sh_x
                            angle = abs(90.0 - np.degrees(math.atan2(dy, dx)))
                            state.angle_history.append(angle)
                            smoothed_angle = float(np.median(state.angle_history))
                            posture = fall_detector.classify_posture(smoothed_angle)

                            mp_drawing.draw_landmarks(
                                crop,
                                pr.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2, circle_radius=2),
                            )
                            annotated[cy1:cy2, cx1:cx2] = crop

                    # Geometric Aspect-Ratio Fallback
                    if posture == "Unknown":
                        if ar <= 0.85:
                            posture = "Lying Down"
                            smoothed_angle = max(65.0, 90.0 - (ar * 45.0))
                        elif ar <= 1.15:
                            posture = "Falling"
                            smoothed_angle = 45.0
                        else:
                            posture = "Standing"
                            smoothed_angle = 15.0

                    state.posture_history.append(posture)

                    # 5-frame window transition check (e.g., 2 standing + 3 falling, 1 falling + 4 lying, etc.)
                    is_transition_5f = fall_detector.check_5frame_transition(state.posture_history)

                    if posture in ("Falling", "Lying Down") or ar < 0.90:
                        state.falled_count += 1
                        state.standing_frames = 0
                    else:
                        state.falled_count = max(0, state.falled_count - 1)
                        state.standing_frames += 1

                    # Trigger "FALLED" on 5-frame transition or sustained fallen posture
                    if is_transition_5f or (state.is_falled and (posture in ("Falling", "Lying Down") or ar < 1.0)):
                        state.is_falled = True
                        if not state.fall_detected:
                            events.append({
                                "frame": frame_idx, "ts": round(ts, 3), "model": "fall",
                                "label": "FALLED", "conf": None
                            })
                            state.fall_detected = True

                    # Recovery reset when standing back up
                    if posture == "Standing" and state.standing_frames >= 8 and ar > 1.25:
                        state.is_falled = False
                        state.fall_detected = False
                        state.falled_count = 0

                    if state.is_falled or state.fall_detected:
                        any_fall_detected = True
                        box_color = (0, 0, 255)
                        lbl_text = f"ID{track_id}: FALLED ({smoothed_angle:.0f}deg)"
                    elif posture == "Falling":
                        box_color = (0, 140, 255)
                        lbl_text = f"ID{track_id}: Falling ({smoothed_angle:.0f}deg)"
                    elif posture == "Lying Down":
                        box_color = (0, 140, 255)
                        lbl_text = f"ID{track_id}: Lying ({smoothed_angle:.0f}deg)"
                    else:
                        box_color = (0, 200, 60)
                        lbl_text = f"ID{track_id}: Standing ({smoothed_angle:.0f}deg)"

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    (tw, th), bl = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    ty = max(th + 4, y1)
                    cv2.rectangle(annotated, (x1, ty - th - 6), (x1 + tw + 6, ty + bl + 2), box_color, -1)
                    cv2.putText(annotated, lbl_text, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            if any_fall_detected:
                cv2.rectangle(annotated, (0, 0), (w_f, 42), (0, 0, 220), -1)
                cv2.putText(annotated, "ALARM: FALL DETECTED (YIQILISH)", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        except Exception:
            pass

    return annotated, events


class VideoAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎬 Video Tahlil Tizimi")
        self.configure(bg=BG)
        self.geometry("1200x700")
        self.minsize(920, 580)
        self.resizable(True, True)

        self._video_path: Optional[str] = None
        self._video_info: dict = {}
        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._fire_model = None
        self._ppe_model = None
        self._fall_model = None

        self._events: list[dict] = []
        self._output_video_path: Optional[str] = None
        self._active_job_id: Optional[str] = None

        self._ui_queue: queue.Queue = queue.Queue()

        self._mode_var = tk.StringVar(value="server")
        self._server_url_var = tk.StringVar(value=DEFAULT_SERVER_URL)
        self._model_vars: dict[str, tk.BooleanVar] = {
            "fire": tk.BooleanVar(value=True),
            "fall": tk.BooleanVar(value=False),
            "ppe": tk.BooleanVar(value=True),
        }
        self._every_n_var = tk.IntVar(value=3)
        self._conf_var = tk.DoubleVar(value=0.50)

        self._build_ui()
        self._poll_ui_queue()
        self._check_server_status_async()

    def _styled_frame(self, parent, bg=None, **kwargs):
        return tk.Frame(parent, bg=bg or BG2, **kwargs)

    def _label(self, parent, text, font=None, fg=None, bg=None, **kw):
        return tk.Label(parent, text=text, font=font or FONT_LABEL,
                        fg=fg or FG, bg=bg or BG2, **kw)

    def _build_ui(self):
        # Top Bar
        top_bar = tk.Frame(self, bg=BG, pady=6)
        top_bar.pack(fill="x", padx=10)
        tk.Label(top_bar, text="🎬  Video Tahlil Tizimi", font=("Segoe UI", 15, "bold"),
                 fg=ACCENT, bg=BG).pack(side="left")

        # Top Bar Download Button (Always 100% visible on top!)
        self._top_dl_btn = tk.Button(
            top_bar, text="💾  Tahlil qilingan videoni yuklab olish (MP4)",
            command=self._download_video,
            bg=GREEN, fg="#ffffff", font=("Segoe UI", 10, "bold"),
            bd=0, relief="flat", cursor="hand2", padx=14, pady=4,
            activebackground=GREEN, activeforeground="#ffffff"
        )
        self._top_dl_btn.pack(side="right", padx=10)
        self._top_dl_btn.config(state="disabled")

        self._server_status_lbl = tk.Label(
            top_bar, text="Tekshirilmoqda...", font=FONT_SMALL,
            fg=YELLOW, bg=CARD, padx=8, pady=3
        )
        self._server_status_lbl.pack(side="right", padx=4)

        tk.Frame(self, bg=CARD, height=1).pack(fill="x")

        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True, padx=0, pady=0)

        left = self._build_left_panel(content)
        left.pack(side="left", fill="y", padx=(8, 4), pady=6)

        right = self._build_right_panel(content)
        right.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=6)

    def _build_left_panel(self, parent) -> tk.Frame:
        panel = self._styled_frame(parent, bg=BG2, width=310)
        panel.pack_propagate(False)

        # 1. Execution Mode
        self._section(panel, "🌐  Tahlil Rejimi")
        mode_f = self._styled_frame(panel)
        mode_f.pack(fill="x", padx=8, pady=(0, 2))

        m_row = tk.Frame(mode_f, bg=BG2)
        m_row.pack(fill="x")
        tk.Radiobutton(
            m_row, text="🌐 Server", variable=self._mode_var,
            value="server", font=FONT_LABEL, fg=FG, bg=BG2,
            selectcolor=CARD, activebackground=BG2, activeforeground=FG
        ).pack(side="left", padx=(0, 10))
        tk.Radiobutton(
            m_row, text="💻 Lokal", variable=self._mode_var,
            value="local", font=FONT_LABEL, fg=FG, bg=BG2,
            selectcolor=CARD, activebackground=BG2, activeforeground=FG
        ).pack(side="left")

        url_row = tk.Frame(mode_f, bg=BG2)
        url_row.pack(fill="x", pady=(2, 2))
        tk.Label(url_row, text="IP:", font=FONT_SMALL, fg=FG2, bg=BG2).pack(side="left")
        tk.Entry(
            url_row, textvariable=self._server_url_var, font=FONT_SMALL,
            bg=CARD, fg=FG, insertbackground=FG, bd=0
        ).pack(side="left", fill="x", expand=True, padx=(3, 3))
        tk.Button(
            url_row, text="🔌 Tekshir", command=self._test_server_connection,
            bg=CARD, fg=FG, font=FONT_SMALL, bd=0, padx=4, pady=1, cursor="hand2"
        ).pack(side="right")

        tk.Frame(panel, bg=CARD, height=1).pack(fill="x", padx=8, pady=3)

        # 2. File Picker
        self._section(panel, "📁  Video Yuklash")
        self._btn(panel, "📂  Video Tanlash", self._pick_video, color=ACCENT).pack(
            fill="x", padx=8, pady=(0, 2))

        info_f = self._styled_frame(panel)
        info_f.pack(fill="x", padx=8, pady=(0, 3))
        self._file_name_lbl = self._label(info_f, "Fayl tanlanmagan", fg=FG2, font=FONT_SMALL)
        self._file_name_lbl.pack(anchor="w")
        self._file_info_lbl = self._label(info_f, "", fg=GREEN, font=FONT_SMALL)
        self._file_info_lbl.pack(anchor="w")

        tk.Frame(panel, bg=CARD, height=1).pack(fill="x", padx=8, pady=3)

        # 3. Models
        self._section(panel, "🤖  Modellar")
        for model_id, info in MODEL_INFO.items():
            row = tk.Frame(panel, bg=BG2)
            row.pack(fill="x", padx=8, pady=1)
            tk.Checkbutton(
                row, text=info["label"],
                variable=self._model_vars[model_id],
                font=FONT_LABEL, fg=FG, bg=BG2,
                selectcolor=CARD, activebackground=BG2,
                activeforeground=FG, bd=0, highlightthickness=0
            ).pack(side="left")
            if info["slow"]:
                tk.Label(row, text="(sekin)", font=FONT_SMALL, fg=YELLOW, bg=BG2).pack(side="left", padx=2)

        tk.Frame(panel, bg=CARD, height=1).pack(fill="x", padx=8, pady=3)

        # 4. Settings
        self._section(panel, "⚙️  Sozlamalar")
        cfg_f = self._styled_frame(panel)
        cfg_f.pack(fill="x", padx=8, pady=(0, 3))

        for label, var, frm, to, inc, fmt in [
            ("Har N-kadr:", self._every_n_var, 1, 30, 1, None),
            ("Min ishonch:", self._conf_var, 0.1, 1.0, 0.05, "%.2f"),
        ]:
            row = tk.Frame(cfg_f, bg=BG2)
            row.pack(fill="x", pady=1)
            self._label(row, label, bg=BG2, font=FONT_SMALL).pack(side="left")
            kw = dict(from_=frm, to=to, increment=inc, textvariable=var,
                      width=5, bg=CARD, fg=FG, insertbackground=FG,
                      buttonbackground=CARD, font=FONT_SMALL, bd=0)
            if fmt:
                kw["format"] = fmt
            tk.Spinbox(row, **kw).pack(side="right")

        tk.Frame(panel, bg=CARD, height=1).pack(fill="x", padx=8, pady=3)

        # 5. Controls & Main Download
        self._section(panel, "▶  Boshqaruv & Yuklab Olish")
        ctrl_f = self._styled_frame(panel)
        ctrl_f.pack(fill="x", padx=8, pady=(0, 2))

        self._start_btn = self._btn(ctrl_f, "▶  Tahlilni Boshlash", self._start_analysis, color=GREEN)
        self._start_btn.pack(fill="x", pady=2)

        self._stop_btn = self._btn(ctrl_f, "⏹  To'xtatish", self._stop_analysis, color=RED)
        self._stop_btn.pack(fill="x", pady=1)
        self._stop_btn.config(state="disabled")

        # Progress bar
        self._progress_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("P.Horizontal.TProgressbar",
                        troughcolor=CARD, background=ACCENT,
                        darkcolor=ACCENT, lightcolor=ACCENT, bordercolor=BG2)
        ttk.Progressbar(panel, variable=self._progress_var, maximum=100,
                        style="P.Horizontal.TProgressbar").pack(fill="x", padx=8, pady=(3, 1))
        self._progress_lbl = self._label(panel, "Tayyor", fg=FG2, font=FONT_SMALL)
        self._progress_lbl.pack(padx=8, anchor="w")

        # Main Download Button inside Left Panel
        self._dl_video_btn = tk.Button(
            panel, text="💾  Videoni Yuklab Olish (MP4)",
            command=self._download_video,
            bg=GREEN, fg="#ffffff", font=("Segoe UI", 10, "bold"),
            bd=0, relief="flat", cursor="hand2", pady=8, padx=6,
            activebackground=GREEN, activeforeground="#ffffff"
        )
        self._dl_video_btn.pack(fill="x", padx=8, pady=(6, 2))
        self._dl_video_btn.config(state="disabled")

        sub_dl_row = tk.Frame(panel, bg=BG2)
        sub_dl_row.pack(fill="x", padx=8, pady=(0, 4))
        self._dl_csv_btn = tk.Button(
            sub_dl_row, text="📋 CSV", command=self._download_csv,
            bg=CARD, fg=FG, font=FONT_SMALL, bd=0, padx=4, pady=2, cursor="hand2"
        )
        self._dl_csv_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))
        self._dl_csv_btn.config(state="disabled")

        self._dl_json_btn = tk.Button(
            sub_dl_row, text="📄 JSON", command=self._download_json,
            bg=CARD, fg=FG, font=FONT_SMALL, bd=0, padx=4, pady=2, cursor="hand2"
        )
        self._dl_json_btn.pack(side="right", fill="x", expand=True, padx=(2, 0))
        self._dl_json_btn.config(state="disabled")

        return panel

    def _build_right_panel(self, parent) -> tk.Frame:
        panel = self._styled_frame(parent, bg=BG)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=3)
        panel.rowconfigure(1, weight=2)

        # Preview Card
        preview_card = tk.Frame(panel, bg=CARD, bd=0)
        preview_card.grid(row=0, column=0, sticky="nsew", pady=(0, 4))

        hdr = tk.Frame(preview_card, bg=CARD)
        hdr.pack(fill="x", padx=8, pady=(4, 2))
        tk.Label(hdr, text="🎞  Joriy Kadr Preview", font=("Segoe UI", 11, "bold"),
                 fg=FG, bg=CARD).pack(side="left")
        self._frame_lbl = tk.Label(hdr, text="—", font=FONT_SMALL, fg=FG2, bg=CARD)
        self._frame_lbl.pack(side="right")

        self._canvas = tk.Canvas(preview_card, bg="#0a0a1a",
                                 width=PREVIEW_W, height=PREVIEW_H, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self._photo: Optional[ImageTk.PhotoImage] = None

        # Events Card
        events_card = tk.Frame(panel, bg=CARD, bd=0)
        events_card.grid(row=1, column=0, sticky="nsew")

        ehdr = tk.Frame(events_card, bg=CARD)
        ehdr.pack(fill="x", padx=8, pady=(4, 2))
        tk.Label(ehdr, text="📋  Aniqlangan Hodisalar", font=("Segoe UI", 10, "bold"),
                 fg=FG, bg=CARD).pack(side="left")

        # Download button right inside table header too!
        self._table_dl_btn = tk.Button(
            ehdr, text="💾 Videoni yuklab olish", command=self._download_video,
            bg=GREEN, fg="#ffffff", font=("Segoe UI", 8, "bold"),
            bd=0, padx=8, pady=1, cursor="hand2"
        )
        self._table_dl_btn.pack(side="right", padx=4)
        self._table_dl_btn.config(state="disabled")

        self._event_count_lbl = tk.Label(ehdr, text="0 ta hodisa", font=FONT_SMALL, fg=FG2, bg=CARD)
        self._event_count_lbl.pack(side="right", padx=6)

        tree_frame = tk.Frame(events_card, bg=CARD)
        tree_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        style = ttk.Style()
        style.configure("Dark.Treeview",
                        background=BG2, foreground=FG,
                        fieldbackground=BG2, rowheight=20,
                        bordercolor=CARD, borderwidth=0, font=FONT_MONO)
        style.configure("Dark.Treeview.Heading",
                        background=CARD, foreground=FG,
                        relief="flat", font=("Segoe UI", 9, "bold"))
        style.map("Dark.Treeview", background=[("selected", ACCENT2)])

        cols = ("Frame", "Vaqt", "Model", "Natija", "Ishonch")
        self._tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                  style="Dark.Treeview", height=6)
        for col, w in zip(cols, [65, 75, 110, 170, 75]):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, minwidth=35)

        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._tree.pack(side="left", fill="both", expand=True)

        return panel

    def _section(self, parent, text):
        tk.Label(parent, text=text, font=("Segoe UI", 9, "bold"),
                 fg=ACCENT, bg=BG2).pack(anchor="w", padx=8, pady=(4, 1))

    def _btn(self, parent, text, cmd, color=ACCENT):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=color, fg="#ffffff", font=("Segoe UI", 9, "bold"),
                      bd=0, relief="flat", cursor="hand2",
                      activebackground=color, activeforeground="#ffffff",
                      pady=5, padx=4)
        b.bind("<Enter>", lambda e: b.config(bg=self._lighten(color)))
        b.bind("<Leave>", lambda e: b.config(bg=color))
        return b

    def _lighten(self, hex_color: str) -> str:
        h = hex_color.lstrip("#")
        r, g, bl = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"#{min(255,r+30):02x}{min(255,g+30):02x}{min(255,bl+30):02x}"

    def _check_server_status_async(self):
        def _check():
            url = self._server_url_var.get().strip().rstrip("/")
            try:
                req = urllib.request.Request(f"{url}/", headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    if resp.status == 200:
                        self._ui_queue.put(("server_status", "🟢 Server Online", GREEN))
                        return
            except Exception:
                pass
            self._ui_queue.put(("server_status", "🔴 Server Offline (Lokal)", YELLOW))

        threading.Thread(target=_check, daemon=True).start()

    def _test_server_connection(self):
        url = self._server_url_var.get().strip().rstrip("/")
        self._server_status_lbl.config(text="Tekshirilmoqda...", fg=YELLOW)

        def _check():
            try:
                req = urllib.request.Request(f"{url}/", headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=2.5) as resp:
                    if resp.status == 200:
                        self._ui_queue.put(("server_status", "🟢 Server Online", GREEN))
                        self._ui_queue.put(("toast", f"Muvaffaqiyatli: Server online va tayyor!\n\nManzil: {url}"))
                        return
            except Exception as e:
                self._ui_queue.put(("server_status", "🔴 Server Offline", RED))
                self._ui_queue.put(("error_popup", f"Serverga ulanib bo'lmadi:\n{url}\n\nSabab: {e}\n\nServerni ishga tushirish: uv run python video_server.py"))

        threading.Thread(target=_check, daemon=True).start()

    def _pick_video(self):
        path = filedialog.askopenfilename(
            title="Video fayl tanlash",
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.ts"), ("Barchasi", "*.*")]
        )
        if not path:
            return
        self._video_path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Xatolik", "Video faylni ochib bo'lmadi!")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = n / fps if fps > 0 else 0
        cap.release()
        self._video_info = {"fps": fps, "w": w, "h": h, "n": n, "dur": dur}
        bn = os.path.basename(path)
        self._file_name_lbl.config(text=bn[:36] + "…" if len(bn) > 36 else bn, fg=FG)
        self._file_info_lbl.config(
            text=f"{w}x{h} | {fps:.1f} fps | {n} kadr | {_sec_to_hms(dur)}")
        self._reset_results()

    def _reset_results(self):
        self._events = []
        self._output_video_path = None
        self._active_job_id = None
        self._progress_var.set(0)
        self._progress_lbl.config(text="Tayyor")
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._event_count_lbl.config(text="0 ta hodisa")
        self._set_download_state("disabled")
        self._canvas.delete("all")
        self._frame_lbl.config(text="—")

    def _set_download_state(self, state: str):
        for b in (self._top_dl_btn, self._dl_video_btn, self._table_dl_btn, self._dl_csv_btn, self._dl_json_btn):
            b.config(state=state)

    def _start_analysis(self):
        if not self._video_path:
            messagebox.showwarning("Ogohlantirish", "Avval video fayl tanlang!")
            return
        if not any(v.get() for v in self._model_vars.values()):
            messagebox.showwarning("Ogohlantirish", "Kamida bitta modelni tanlang!")
            return

        self._reset_results()
        self._stop_event.clear()
        self._running = True
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")

        mode = self._mode_var.get()
        if mode == "server":
            self._thread = threading.Thread(target=self._run_server_analysis, daemon=True)
        else:
            self._thread = threading.Thread(target=self._run_local_analysis, daemon=True)
        self._thread.start()

    def _stop_analysis(self):
        self._stop_event.set()
        self._running = False
        if self._active_job_id:
            server_url = self._server_url_var.get().strip().rstrip("/")
            try:
                req = urllib.request.Request(f"{server_url}/api/video/cancel/{self._active_job_id}", method="POST")
                urllib.request.urlopen(req, timeout=2.0)
            except Exception:
                pass

        self._progress_lbl.config(text="To'xtatildi")
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")

    # --- Server Mode ---
    def _run_server_analysis(self):
        server_url = self._server_url_var.get().strip().rstrip("/")
        self._ui_queue.put(("status", "🌐 Serverga video yuklanmoqda…"))

        models_list = [k for k, v in self._model_vars.items() if v.get()]
        models_str = ",".join(models_list)
        every_n = self._every_n_var.get()
        conf = self._conf_var.get()

        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        filename = os.path.basename(self._video_path)

        body_parts = []
        for field_name, val in [("models", models_str), ("every_n", str(every_n)), ("conf", f"{conf:.2f}")]:
            body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field_name}\"\r\n\r\n{val}\r\n".encode("utf-8"))

        file_header = f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: video/mp4\r\n\r\n".encode("utf-8")
        file_footer = f"\r\n--{boundary}--\r\n".encode("utf-8")

        try:
            with open(self._video_path, "rb") as f:
                file_bytes = f.read()

            full_body = b"".join(body_parts) + file_header + file_bytes + file_footer

            upload_req = urllib.request.Request(
                f"{server_url}/api/video/upload",
                data=full_body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST"
            )
            with urllib.request.urlopen(upload_req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                job_id = data.get("job_id")
                self._active_job_id = job_id

        except Exception as e:
            self._ui_queue.put(("error", f"Serverga ulanish xatosi:\n{e}\n\nServer ishlayaptimi (uv run python video_server.py)?"))
            return

        self._ui_queue.put(("status", f"🌐 Serverda tahlil ketmoqda (Job: {job_id})…"))

        last_event_count = 0
        while not self._stop_event.is_set():
            time.sleep(0.25)
            try:
                status_req = urllib.request.Request(f"{server_url}/api/video/status/{job_id}")
                with urllib.request.urlopen(status_req, timeout=3.0) as resp:
                    s_data = json.loads(resp.read().decode("utf-8"))

                st = s_data.get("status")
                pct = s_data.get("progress_pct", 0.0)
                cf = s_data.get("current_frame", 0)
                tf = s_data.get("total_frames", 1)
                ev_cnt = s_data.get("events_count", 0)

                # Fetch preview frame
                try:
                    prev_req = urllib.request.Request(f"{server_url}/api/video/preview/{job_id}")
                    with urllib.request.urlopen(prev_req, timeout=1.5) as p_resp:
                        if p_resp.status == 200:
                            jpeg_bytes = p_resp.read()
                            nparr = np.frombuffer(jpeg_bytes, np.uint8)
                            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            self._ui_queue.put(("progress", pct, cf, tf, frame_bgr))
                        else:
                            self._ui_queue.put(("progress", pct, cf, tf, None))
                except Exception:
                    self._ui_queue.put(("progress", pct, cf, tf, None))

                # Fetch new events
                if ev_cnt > last_event_count:
                    try:
                        ev_req = urllib.request.Request(f"{server_url}/api/video/events/{job_id}")
                        with urllib.request.urlopen(ev_req, timeout=2.0) as ev_resp:
                            ev_data = json.loads(ev_resp.read().decode("utf-8"))
                            events = ev_data.get("events", [])
                            self._events = events
                            self._ui_queue.put(("events_update", events))
                            last_event_count = len(events)
                    except Exception:
                        pass

                if st == "completed":
                    try:
                        ev_req = urllib.request.Request(f"{server_url}/api/video/events/{job_id}")
                        with urllib.request.urlopen(ev_req, timeout=2.0) as ev_resp:
                            ev_data = json.loads(ev_resp.read().decode("utf-8"))
                            self._events = ev_data.get("events", [])
                    except Exception:
                        pass
                    self._ui_queue.put(("done", self._events, f"server://{job_id}"))
                    break
                elif st == "failed":
                    err_msg = s_data.get("error_message", "Serverda xatolik yuz berdi")
                    self._ui_queue.put(("error", f"Server xatoligi: {err_msg}"))
                    break
                elif st == "stopped":
                    self._ui_queue.put(("stopped", self._events))
                    break

            except Exception:
                time.sleep(0.5)

    # --- Local Mode ---
    def _run_local_analysis(self):
        from multi_object_detector.analysis.detectors import fire_detector, ppe_detector, fall_detector

        self._ui_queue.put(("status", "💻 Modellar yuklanmoqda (Lokal)…"))

        fire_m = None
        if self._model_vars["fire"].get():
            if self._fire_model is None:
                self._fire_model = fire_detector.get_model()
            fire_m = self._fire_model

        ppe_m = None
        if self._model_vars["ppe"].get():
            if self._ppe_model is None:
                self._ppe_model = ppe_detector.get_model()
            ppe_m = self._ppe_model

        fall_m = None
        fall_pose = None
        fall_states: dict = {}
        if self._model_vars["fall"].get():
            if self._fall_model is None:
                self._fall_model = fall_detector.get_model()
            fall_m = self._fall_model
            import mediapipe as mp
            fall_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                min_detection_confidence=0.75,
                min_tracking_confidence=0.75,
            )

        models_enabled = {k: v.get() for k, v in self._model_vars.items()}
        every_n = max(1, self._every_n_var.get())

        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            self._ui_queue.put(("error", "Video ochilmadi"))
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        out_dir = os.path.dirname(self._video_path)
        base = os.path.splitext(os.path.basename(self._video_path))[0]
        out_path = os.path.join(out_dir, f"{base}_analyzed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        filt = StreamCorruptionFilter(camera_id="video_ui")
        events: list[dict] = []
        frame_idx = 0

        self._ui_queue.put(("status", "💻 Tahlil boshlandi (Lokal)…"))

        try:
            while cap.isOpened():
                if self._stop_event.is_set():
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                pct = (frame_idx / total) * 100.0

                valid, _ = filt.check_frame(frame)
                if not valid:
                    writer.write(frame)
                    self._ui_queue.put(("progress", pct, frame_idx, total, None))
                    continue

                if frame_idx % every_n == 0:
                    annotated, new_events = _analyze_frame_multi(
                        frame, frame_idx, fps, models_enabled,
                        _fire_model=fire_m, _ppe_model=ppe_m,
                        _fall_model=fall_m, _fall_pose=fall_pose,
                        _fall_track_states=fall_states,
                    )
                    events.extend(new_events)
                    writer.write(annotated)
                    self._ui_queue.put(("progress", pct, frame_idx, total, annotated))
                else:
                    writer.write(frame)
                    self._ui_queue.put(("progress", pct, frame_idx, total, None))
        finally:
            cap.release()
            writer.release()
            if fall_pose is not None:
                try:
                    fall_pose.close()
                except Exception:
                    pass

        if not self._stop_event.is_set():
            self._output_video_path = out_path
            self._ui_queue.put(("done", events, out_path))
        else:
            self._ui_queue.put(("stopped", events))

    def _poll_ui_queue(self):
        try:
            while True:
                msg = self._ui_queue.get_nowait()
                self._handle_ui_msg(msg)
        except queue.Empty:
            pass
        self.after(50, self._poll_ui_queue)

    def _handle_ui_msg(self, msg):
        kind = msg[0]
        if kind == "server_status":
            _, text, color = msg
            self._server_status_lbl.config(text=text, fg=color)

        elif kind == "status":
            self._progress_lbl.config(text=msg[1])

        elif kind == "progress":
            _, pct, fidx, total, frame = msg
            self._progress_var.set(pct)
            self._progress_lbl.config(text=f"Kadr: {fidx}/{total}  ({pct:.1f}%)")
            self._frame_lbl.config(text=f"Kadr: {fidx} / {total}")
            if frame is not None:
                self._show_frame(frame)

        elif kind == "events_update":
            _, events = msg
            self._populate_tree(events)

        elif kind == "done":
            _, events, out_path = msg
            self._events = events
            self._output_video_path = out_path if not out_path.startswith("server://") else None
            self._progress_var.set(100)
            self._progress_lbl.config(text=f"✅ Tayyor! {len(events)} ta hodisa aniqlandi.")
            self._running = False
            self._start_btn.config(state="normal")
            self._stop_btn.config(state="disabled")
            self._populate_tree(events)
            self._set_download_state("normal")

        elif kind == "stopped":
            _, events = msg
            self._events = events
            self._running = False
            self._start_btn.config(state="normal")
            self._stop_btn.config(state="disabled")
            self._populate_tree(events)
            if events:
                self._dl_csv_btn.config(state="normal")
                self._dl_json_btn.config(state="normal")

        elif kind == "toast":
            messagebox.showinfo("Server holati", msg[1])

        elif kind == "error_popup":
            messagebox.showerror("Server holati", msg[1])

        elif kind == "error":
            self._running = False
            self._start_btn.config(state="normal")
            self._stop_btn.config(state="disabled")
            messagebox.showerror("Xatolik", msg[1])

    def _show_frame(self, frame_bgr: np.ndarray):
        cw = self._canvas.winfo_width() or PREVIEW_W
        ch = self._canvas.winfo_height() or PREVIEW_H
        if cw < 10 or ch < 10:
            return
        fh, fw = frame_bgr.shape[:2]
        scale = min(cw / fw, ch / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        self._canvas.create_image((cw - nw) // 2, (ch - nh) // 2, anchor="nw", image=self._photo)

    def _populate_tree(self, events: list[dict]):
        for item in self._tree.get_children():
            self._tree.delete(item)
        for ev in events:
            self._tree.insert("", "end", values=(
                ev.get("frame", ""),
                _sec_to_hms(ev.get("ts", 0)),
                ev.get("model", "").upper(),
                ev.get("label", ""),
                f"{ev['conf']:.2f}" if ev.get("conf") is not None else "—",
            ))
        self._event_count_lbl.config(text=f"{len(events)} ta hodisa")

    def _download_video(self):
        dest = filedialog.asksaveasfilename(
            title="Tahlil qilingan videoni saqlash",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")],
            initialfile="analyzed_video.mp4",
        )
        if not dest:
            return

        if self._active_job_id and self._mode_var.get() == "server":
            server_url = self._server_url_var.get().strip().rstrip("/")
            try:
                dl_url = f"{server_url}/api/video/download/{self._active_job_id}"
                urllib.request.urlretrieve(dl_url, dest)
                self._output_video_path = dest
                messagebox.showinfo("Muvaffaqiyat", f"Tahlil qilingan video saqlandi:\n{dest}")
                return
            except Exception as e:
                messagebox.showerror("Xatolik", f"Serverdan yuklab olishda xatolik: {e}")
                return

        if self._output_video_path and os.path.exists(self._output_video_path):
            import shutil
            shutil.copy2(self._output_video_path, dest)
            messagebox.showinfo("Muvaffaqiyat", f"Tahlil qilingan video saqlandi:\n{dest}")

    def _download_csv(self):
        if not self._events:
            messagebox.showwarning("Ogohlantirish", "Hali hech qanday hodisa aniqlanmadi.")
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV fayl", "*.csv")],
            initialfile="tahlil_natija.csv",
        )
        if not dest:
            return

        if self._active_job_id and self._mode_var.get() == "server":
            server_url = self._server_url_var.get().strip().rstrip("/")
            try:
                dl_url = f"{server_url}/api/video/download_csv/{self._active_job_id}"
                urllib.request.urlretrieve(dl_url, dest)
                messagebox.showinfo("Muvaffaqiyat", f"CSV hisobot saqlandi:\n{dest}")
                return
            except Exception:
                pass

        with open(dest, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "ts", "model", "label", "conf"])
            writer.writeheader()
            for ev in self._events:
                writer.writerow({
                    "frame": ev.get("frame"),
                    "ts": round(ev.get("ts", 0), 3),
                    "model": ev.get("model"),
                    "label": ev.get("label"),
                    "conf": ev.get("conf"),
                })
        messagebox.showinfo("Muvaffaqiyat", f"CSV saqlandi:\n{dest}")

    def _download_json(self):
        if not self._events:
            messagebox.showwarning("Ogohlantirish", "Hali hech qanday hodisa aniqlanmadi.")
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON fayl", "*.json")],
            initialfile="tahlil_natija.json",
        )
        if not dest:
            return

        if self._active_job_id and self._mode_var.get() == "server":
            server_url = self._server_url_var.get().strip().rstrip("/")
            try:
                dl_url = f"{server_url}/api/video/download_json/{self._active_job_id}"
                urllib.request.urlretrieve(dl_url, dest)
                messagebox.showinfo("Muvaffaqiyat", f"JSON hisobot saqlandi:\n{dest}")
                return
            except Exception:
                pass

        payload = {
            "video": self._video_path,
            "video_info": self._video_info,
            "models_used": [k for k, v in self._model_vars.items() if v.get()],
            "total_events": len(self._events),
            "events": self._events,
        }
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("Muvaffaqiyat", f"JSON saqlandi:\n{dest}")


if __name__ == "__main__":
    app = VideoAnalyzerApp()
    app.mainloop()
