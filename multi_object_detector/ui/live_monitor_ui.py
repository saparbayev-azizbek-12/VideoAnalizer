from __future__ import annotations
import cv2
import time
import requests
import threading
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, filedialog

from multi_object_detector.core import config
from multi_object_detector.streaming import camera_manager
from multi_object_detector.detectors import manager as models_manager

MODEL_ICONS = {"fire": "🔥", "fall": "🚨", "danger_zone": "⛔", "ppe": "🦺"}


class CameraPopout(tk.Toplevel):
    def __init__(self, app: "MonitoringApp", cam_id: str, cam_name: str):
        super().__init__(app.root)
        self.app = app
        self.cam_id = cam_id
        self.title(f"📷 {cam_name}")
        self.geometry("880x560")
        self.configure(bg="#000000")
        self.minsize(400, 300)
        self._running = True
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._photo_ref = None
        self._session = requests.Session()
        self.label = tk.Label(self, bg="#000000")
        self.label.pack(fill=tk.BOTH, expand=True)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        threading.Thread(target=self._fetch_loop, daemon=True).start()
        self.after(config.POPOUT_UPDATE_MS, self._refresh)

    def _fetch_loop(self) -> None:
        while self._running:
            t0 = time.time()
            frame = None
            if self.app._is_local_mode():
                cam = camera_manager.get_camera(self.cam_id)
                if cam is not None:
                    frame = cam.get_annotated_frame()
            else:
                try:
                    url = self.app.server_url.get().rstrip("/")
                    res = self._session.get(f"{url}/api/cameras/{self.cam_id}/preview?max_w=960&max_h=540", timeout=config.API_TIMEOUT)
                    if res.status_code == 200 and res.content:
                        frame = cv2.imdecode(np.frombuffer(res.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                except Exception:
                    pass

            if frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame

            elapsed = time.time() - t0
            target = config.POPOUT_UPDATE_MS / 1000
            time.sleep(max(0.005, target - elapsed))

    def _refresh(self) -> None:
        if not self._running:
            return
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is not None:
            h, w = frame.shape[:2]
            max_w = max(200, self.label.winfo_width() or 860)
            max_h = max(200, self.label.winfo_height() or 540)
            scale = min(max_w / w, max_h / h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(frame, (new_w, new_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self._photo_ref = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.label.config(image=self._photo_ref)
        self.after(config.POPOUT_UPDATE_MS, self._refresh)

    def on_close(self) -> None:
        self._running = False
        self.app.popouts.pop(self.cam_id, None)
        self.destroy()


class ZoneEditorWindow(tk.Toplevel):
    MAX_W, MAX_H = 900, 600

    def __init__(self, app: "MonitoringApp", cam_id: str, cam_name: str):
        super().__init__(app.root)
        self.app = app
        self.cam_id = cam_id
        self.title(f"⛶ Xavfli hudud belgilash - {cam_name}")
        self.configure(bg="#1e1e1e")
        self.resizable(False, False)
        self.points: list[list[float]] = []
        self.scale = 1.0
        self.canvas: tk.Canvas | None = None
        self.photo_ref = None

        tk.Label(
            self,
            text=("Rasmda ketma-ket bosib poligon nuqtalarini belgilang (kamida 3 ta). "
                  "Tugagach 'Saqlash' tugmasini bosing."),
            bg="#1e1e1e", fg="#cccccc", font=("Segoe UI", 9),
            wraplength=860, justify=tk.LEFT,
        ).pack(padx=10, pady=(10, 5), anchor=tk.W)

        self.canvas_frame = tk.Frame(self, bg="#000000")
        self.canvas_frame.pack(padx=10, pady=5)

        btn_frame = tk.Frame(self, bg="#1e1e1e")
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        tk.Button(btn_frame, text="↩ Oxirgi nuqtani o'chirish", command=self.undo_point,
                  bg="#3a3d41", fg="#fff", relief=tk.FLAT, padx=10, pady=5, cursor="hand2").pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="🗑 Tozalash", command=self.clear_zone,
                  bg="#d13438", fg="#fff", relief=tk.FLAT, padx=10, pady=5, cursor="hand2").pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="✖ Yopish", command=self.destroy,
                  bg="#3a3d41", fg="#fff", relief=tk.FLAT, padx=10, pady=5, cursor="hand2").pack(side=tk.RIGHT, padx=4)
        tk.Button(btn_frame, text="💾 Saqlash", command=self.save_zone,
                  bg="#0e8a16", fg="#fff", relief=tk.FLAT, padx=10, pady=5, cursor="hand2").pack(side=tk.RIGHT, padx=4)

        self.status_label = tk.Label(self, text="Kadr yuklanmoqda...", bg="#1e1e1e", fg="#ffcc00", font=("Segoe UI", 9))
        self.status_label.pack(padx=10, pady=(0, 10), anchor=tk.W)
        threading.Thread(target=self._load_frame, daemon=True).start()

    def _load_frame(self) -> None:
        if self.app._is_local_mode():
            cam = camera_manager.get_camera(self.cam_id)
            if cam is None:
                self.after(0, lambda: self.status_label.config(text="Kamera topilmadi", fg="#ff5555"))
                return
            frame = cam.get_raw_frame()
            existing = cam.get_zone()
            if frame is None:
                self.after(0, lambda: self.status_label.config(
                    text="Kadr olinmadi (kamera hali ulanmagan bo'lishi mumkin)", fg="#ff5555"))
                return
            self.after(0, lambda: self._display_frame(frame, existing))
            return

        url = self.app.server_url.get().rstrip("/")
        try:
            res = requests.get(f"{url}/api/cameras/{self.cam_id}/preview", params={"raw": 1, "max_w": 0, "max_h": 0}, timeout=config.API_TIMEOUT)
            frame = cv2.imdecode(np.frombuffer(res.content, dtype=np.uint8), cv2.IMREAD_COLOR)
            zone_res = requests.get(f"{url}/api/cameras/{self.cam_id}/zone", timeout=config.API_TIMEOUT)
            existing = zone_res.json().get("polygon") if zone_res.status_code == 200 else None
        except Exception as e:
            self.after(0, lambda: self.status_label.config(text=f"Xatolik: {e}", fg="#ff5555"))
            return
        if frame is None:
            self.after(0, lambda: self.status_label.config(
                text="Kadr olinmadi (kamera hali ulanmagan bo'lishi mumkin)", fg="#ff5555"))
            return
        self.after(0, lambda: self._display_frame(frame, existing))

    def _display_frame(self, frame: np.ndarray, existing_polygon) -> None:
        self.frame_h, self.frame_w = frame.shape[:2]
        self.scale = min(self.MAX_W / self.frame_w, self.MAX_H / self.frame_h, 1.0)
        self.disp_w = int(self.frame_w * self.scale)
        self.disp_h = int(self.frame_h * self.scale)
        resized = cv2.resize(frame, (self.disp_w, self.disp_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.photo_ref = ImageTk.PhotoImage(image=Image.fromarray(rgb))

        self.canvas = tk.Canvas(self.canvas_frame, width=self.disp_w, height=self.disp_h,
                                 bg="#000000", highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_ref)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        if existing_polygon:
            is_norm = all(0.0 <= float(p[0]) <= 1.0 and 0.0 <= float(p[1]) <= 1.0 for p in existing_polygon)
            if is_norm:
                self.points = [[float(px) * self.disp_w, float(py) * self.disp_h] for px, py in existing_polygon]
            else:
                self.points = [[float(px) * self.scale, float(py) * self.scale] for px, py in existing_polygon]
            self._redraw_polygon()

        self.status_label.config(text=f"Nuqtalar: {len(self.points)}", fg="#00e676")

    def on_canvas_click(self, event) -> None:
        self.points.append([event.x, event.y])
        self._redraw_polygon()
        self.status_label.config(text=f"Nuqtalar: {len(self.points)}", fg="#00e676")

    def undo_point(self) -> None:
        if self.points:
            self.points.pop()
            self._redraw_polygon()
            self.status_label.config(text=f"Nuqtalar: {len(self.points)}", fg="#00e676")

    def _redraw_polygon(self) -> None:
        if self.canvas is None:
            return
        self.canvas.delete("zone")
        r = 4
        for i, (x, y) in enumerate(self.points):
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="#ff5555", outline="", tags="zone")
            if i > 0:
                px, py = self.points[i - 1]
                self.canvas.create_line(px, py, x, y, fill="#ff5555", width=2, tags="zone")
        if len(self.points) >= 3:
            fx, fy = self.points[0]
            lx, ly = self.points[-1]
            self.canvas.create_line(lx, ly, fx, fy, fill="#ff5555", width=2, dash=(4, 2), tags="zone")

    def clear_zone(self) -> None:
        self.points = []
        self._redraw_polygon()
        self.status_label.config(text="Nuqtalar: 0", fg="#00e676")
        threading.Thread(target=self._clear_zone_worker, daemon=True).start()

    def _clear_zone_worker(self) -> None:
        if self.app._is_local_mode():
            cam = camera_manager.get_camera(self.cam_id)
            if cam is not None:
                cam.clear_zone()
            return
        url = self.app.server_url.get().rstrip("/")
        try:
            requests.delete(f"{url}/api/cameras/{self.cam_id}/zone", timeout=config.API_TIMEOUT)
        except Exception:
            pass

    def save_zone(self) -> None:
        if len(self.points) < 3:
            messagebox.showwarning("Ogohlantirish", "Kamida 3 ta nuqta belgilang!", parent=self)
            return
        disp_w = max(1, getattr(self, "disp_w", int(self.MAX_W * self.scale)))
        disp_h = max(1, getattr(self, "disp_h", int(self.MAX_H * self.scale)))
        polygon_norm = [[round(x / disp_w, 6), round(y / disp_h, 6)] for x, y in self.points]
        threading.Thread(target=self._save_zone_worker, args=(polygon_norm,), daemon=True).start()

    def _save_zone_worker(self, polygon_norm: list[list[float]]) -> None:
        if self.app._is_local_mode():
            cam = camera_manager.get_camera(self.cam_id)
            if cam is not None:
                cam.set_zone(polygon_norm)
                self.after(0, self._finish_saved)
            return

        url = self.app.server_url.get().rstrip("/")
        try:
            res = requests.post(f"{url}/api/cameras/{self.cam_id}/zone", json={"polygon": polygon_norm}, timeout=config.API_TIMEOUT)
            if res.status_code == 200:
                self.after(0, self._finish_saved)
            else:
                err = res.json().get("detail", "Noma'lum xatolik")
                self.after(0, lambda: messagebox.showerror("Xatolik", err, parent=self))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Xatolik", str(e), parent=self))

    def _finish_saved(self) -> None:
        messagebox.showinfo("Saqlandi", "Xavfli hudud saqlandi.", parent=self)
        self.destroy()


class MonitoringApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(config.APP_TITLE)
        self.geometry = "1620x960"
        self.root.geometry(self.geometry)
        self.root.configure(bg="#1e1e1e")

        self.server_url = tk.StringVar(value=config.DEFAULT_SERVER_URL)
        self.is_connected = False
        self.view_mode = "grid"
        self.selected_camera_id: str | None = None

        self.camera_cache: dict[str, dict] = {}
        self.camera_order: list[str] = []
        self.popouts: dict[str, CameraPopout] = {}
        self._session = requests.Session()

        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._photo_ref = None
        self._last_shown_w = 0
        self._last_shown_h = 0

        self.model_vars: dict[str, tk.BooleanVar] = {}

        self._setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        threading.Thread(target=self._preview_fetch_loop, daemon=True).start()
        self.root.after(400, self.poll_status_loop)
        self.root.after(config.GUI_UPDATE_MS, self.update_display)

    def _setup_ui(self) -> None:
        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        conn_frame = tk.Frame(main_frame, bg="#252526", bd=1, relief=tk.SOLID)
        conn_frame.pack(fill=tk.X, pady=(0, 8), ipady=6, ipadx=10)

        tk.Label(conn_frame, text="🖥 Server URL (GPU server):", font=("Segoe UI", 11, "bold"),
                 fg="#ffffff", bg="#252526").pack(side=tk.LEFT, padx=(10, 5))

        tk.Entry(conn_frame, textvariable=self.server_url, font=("Segoe UI", 11), width=30,
                 bg="#3c3c3c", fg="#ffffff", insertbackground="#ffffff").pack(side=tk.LEFT, padx=5)

        tk.Button(conn_frame, text="🔌 Ulanishni Tekshirish", command=self.check_server_connection,
                   font=("Segoe UI", 10, "bold"), bg="#007acc", fg="#ffffff",
                   activebackground="#005999", activeforeground="#ffffff",
                   relief=tk.FLAT, padx=12, pady=3, cursor="hand2").pack(side=tk.LEFT, padx=5)

        tk.Button(conn_frame, text="📦 Datasetni yuklab olish", command=self.download_dataset,
                   font=("Segoe UI", 10, "bold"), bg="#6f42c1", fg="#ffffff",
                   activebackground="#5a32a3", activeforeground="#ffffff",
                   relief=tk.FLAT, padx=12, pady=3, cursor="hand2").pack(side=tk.LEFT, padx=5)

        self.conn_status_label = tk.Label(conn_frame, text="● Tekshirilmadi", font=("Segoe UI", 11, "bold"),
                                           fg="#ffcc00", bg="#252526")
        self.conn_status_label.pack(side=tk.RIGHT, padx=15)

        models_frame = tk.Frame(main_frame, bg="#252526", bd=1, relief=tk.SOLID)
        models_frame.pack(fill=tk.X, pady=(0, 8), ipady=6, ipadx=10)

        tk.Label(models_frame, text="🧠 Faol modellar:", font=("Segoe UI", 10, "bold"),
                 fg="#ffffff", bg="#252526").pack(side=tk.LEFT, padx=(10, 10))

        for mid in config.MODEL_IDS:
            var = tk.BooleanVar(value=config.MODEL_DEFAULT_ENABLED.get(mid, True))
            self.model_vars[mid] = var
            tk.Checkbutton(
                models_frame, text=config.MODEL_LABELS[mid], variable=var,
                command=lambda m=mid: self.on_model_toggle(m),
                font=("Segoe UI", 10), fg="#ffffff", bg="#252526", selectcolor="#3c3c3c",
                activebackground="#252526", activeforeground="#ffffff", cursor="hand2",
            ).pack(side=tk.LEFT, padx=8)

        tk.Label(models_frame, text="(o'chirilgan model uchun tahlil/ramka ko'rsatilmaydi)",
                 font=("Segoe UI", 8, "italic"), fg="#888888", bg="#252526").pack(side=tk.LEFT, padx=10)

        body_frame = tk.Frame(main_frame, bg="#1e1e1e")
        body_frame.pack(fill=tk.BOTH, expand=True)

        self._setup_sidebar(body_frame)
        self._setup_right_panel(body_frame)

    def _setup_sidebar(self, parent: tk.Frame) -> None:
        sidebar = tk.Frame(parent, bg="#2d2d2d", width=300)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="📷 KAMERALAR", font=("Segoe UI", 11, "bold"),
                 fg="#00e676", bg="#2d2d2d").pack(anchor=tk.W, padx=10, pady=(10, 4))

        tk.Button(sidebar, text="➕ Kamera qo'shish", command=self.open_add_camera_dialog,
                  font=("Segoe UI", 10, "bold"), bg="#0e8a16", fg="#ffffff",
                  relief=tk.FLAT, padx=8, pady=6, cursor="hand2").pack(fill=tk.X, padx=10, pady=(0, 8))

        list_container = tk.Frame(sidebar, bg="#2d2d2d")
        list_container.pack(fill=tk.BOTH, expand=True, padx=6)

        canvas = tk.Canvas(list_container, bg="#2d2d2d", highlightthickness=0)
        scrollbar = tk.Scrollbar(list_container, orient="vertical", command=canvas.yview)
        self.camera_list_frame = tk.Frame(canvas, bg="#2d2d2d")

        self.camera_list_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.camera_list_frame, anchor="nw", width=270)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._rebuild_camera_list([])

    def _setup_right_panel(self, parent: tk.Frame) -> None:
        right_frame = tk.Frame(parent, bg="#1e1e1e")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        view_bar = tk.Frame(right_frame, bg="#252526")
        view_bar.pack(fill=tk.X, pady=(0, 6))

        self.view_status_label = tk.Label(view_bar, text="🔲 Barcha kameralar (bosib kattalashtiring)",
                                           font=("Segoe UI", 10, "bold"), fg="#ffffff", bg="#252526")
        self.view_status_label.pack(side=tk.LEFT, padx=10, pady=6)

        self.back_btn = tk.Button(view_bar, text="⬅ Orqaga (barchasi)", command=self.back_to_grid,
                                   state=tk.DISABLED, font=("Segoe UI", 10, "bold"),
                                   bg="#3a3d41", fg="#ffffff", relief=tk.FLAT, padx=10, pady=4, cursor="hand2")
        self.back_btn.pack(side=tk.RIGHT, padx=10, pady=4)

        display_frame = tk.Frame(right_frame, bg="#000000", bd=2, relief=tk.SUNKEN)
        display_frame.pack(fill=tk.BOTH, expand=True)

        self.display_label = tk.Label(display_frame, bg="#000000", cursor="hand2")
        self.display_label.pack(fill=tk.BOTH, expand=True)
        self.display_label.bind("<Button-1>", self.on_display_click)

        alerts_frame = tk.Frame(right_frame, bg="#252526", height=150)
        alerts_frame.pack(fill=tk.X, pady=(6, 0))
        alerts_frame.pack_propagate(False)

        tk.Label(alerts_frame, text="🔔 So'nggi hodisalar", font=("Segoe UI", 10, "bold"),
                 fg="#ffcc00", bg="#252526").pack(anchor=tk.W, padx=10, pady=(6, 2))

        text_container = tk.Frame(alerts_frame, bg="#252526")
        text_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        alerts_scroll = tk.Scrollbar(text_container)
        alerts_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.alerts_text = tk.Text(text_container, height=6, bg="#1a1a1a", fg="#dddddd",
                                    font=("Consolas", 9), wrap=tk.WORD, bd=0,
                                    yscrollcommand=alerts_scroll.set, state=tk.DISABLED)
        self.alerts_text.pack(fill=tk.BOTH, expand=True)
        alerts_scroll.config(command=self.alerts_text.yview)

    def download_dataset(self) -> None:
        if not self.is_connected:
            messagebox.showwarning("Ogohlantirish", "Avval serverga ulaning!", parent=self.root)
            return

        save_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Datasetni saqlash",
            defaultextension=".zip",
            filetypes=[("Zip fayllar", "*.zip"), ("Barcha fayllar", "*.*")],
            initialfile=f"dataset_{int(time.time())}.zip"
        )
        if not save_path:
            return

        threading.Thread(target=self._download_dataset_worker, args=(save_path,), daemon=True).start()

    def _download_dataset_worker(self, save_path: str) -> None:
        url = self.server_url.get().rstrip("/")
        try:
            res = requests.get(f"{url}/api/dataset/download", stream=True, timeout=60)
            if res.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in res.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                self.root.after(0, lambda: messagebox.showinfo("Muvaffaqiyatli", f"Dataset muvaffaqiyatli yuklab olindi!\nManzil: {save_path}", parent=self.root))
            else:
                try:
                    err = res.json().get("detail", "Server xatoligi")
                except Exception:
                    err = f"HTTP {res.status_code}"
                self.root.after(0, lambda: messagebox.showwarning("Ogohlantirish", f"Dataset yuklab olinmadi: {err}", parent=self.root))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Xatolik", f"Yuklab olishda xatolik yuz berdi: {e}", parent=self.root))

    def _is_local_mode(self) -> bool:
        url = self.server_url.get().strip().lower()
        return any(url.startswith(p) for p in ("http://localhost", "http://127.0.0.1", "http://0.0.0.0", "local"))

    def check_server_connection(self) -> None:
        self.conn_status_label.config(text="⏳ Tekshirilmoqda...", fg="#ffcc00")
        threading.Thread(target=self._check_connection_worker, daemon=True).start()

    def _check_connection_worker(self) -> None:
        if self._is_local_mode():
            import torch
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Direct In-Memory (CPU)"
            cam_count = len(camera_manager.list_cameras())
            self.root.after(0, lambda: self._on_connected(f"Direct In-Memory ({gpu_name})", cam_count))
            return

        url = self.server_url.get().rstrip("/")
        try:
            res = requests.get(f"{url}/api/health", timeout=config.API_TIMEOUT)
            if res.status_code == 200:
                data = res.json()
                gpu_name = data.get("gpu_name", "Noma'lum")
                cam_count = data.get("camera_count", 0)
                self.root.after(0, lambda: self._on_connected(gpu_name, cam_count))
            else:
                self.root.after(0, lambda: self._on_connect_failed("Server xatoligi"))
        except Exception:
            self.root.after(0, lambda: self._on_connect_failed("Ulanib bo'lmadi"))

    def _on_connected(self, gpu_name: str, cam_count: int) -> None:
        self.is_connected = True
        self.conn_status_label.config(text=f"🟢 Ulangan: {gpu_name} | {cam_count} kamera", fg="#00e676")

    def _on_connect_failed(self, msg: str) -> None:
        self.is_connected = False
        self.conn_status_label.config(text=f"🔴 {msg}", fg="#ff5555")

    def poll_status_loop(self) -> None:
        threading.Thread(target=self._poll_status_worker, daemon=True).start()
        self.root.after(config.STATUS_POLL_MS, self.poll_status_loop)

    def _poll_status_worker(self) -> None:
        if self._is_local_mode():
            cams = [
                {
                    "id": c.id,
                    "name": c.name,
                    "source": c.source,
                    "connected": c.connected,
                    "running": c.running,
                    "has_zone": c.get_zone() is not None,
                    "last_error": c.last_error,
                }
                for c in camera_manager.list_cameras()
            ]
            models_list = [
                {"id": mid, "label": config.MODEL_LABELS[mid], "enabled": models_manager.is_enabled(mid)}
                for mid in config.MODEL_IDS
            ]
            events = [
                {"camera_id": c.id, "camera_name": c.name, **e}
                for c in camera_manager.list_cameras()
                for e in c.get_recent_events(40)
            ]
            events.sort(key=lambda e: e.get("ts", 0), reverse=True)
            self.root.after(0, lambda: self._apply_status(cams, models_list, events))
            return

        if not self.is_connected:
            return

        url = self.server_url.get().rstrip("/")
        try:
            cams = requests.get(f"{url}/api/cameras", timeout=config.API_TIMEOUT).json().get("cameras", [])
            models_list = requests.get(f"{url}/api/models", timeout=config.API_TIMEOUT).json().get("models", [])
            events = requests.get(f"{url}/api/events", params={"limit": 40}, timeout=config.API_TIMEOUT).json().get("events", [])
            self.root.after(0, lambda: self._apply_status(cams, models_list, events))
        except Exception:
            self.root.after(0, lambda: self._on_connect_failed("Ulanish uzildi"))

    def _apply_status(self, cams: list[dict], models_list: list[dict], events: list[dict]) -> None:
        self._rebuild_camera_list(cams)
        self._sync_model_checkboxes(models_list)
        self._update_alerts(events)

    def on_model_toggle(self, model_id: str) -> None:
        enabled = self.model_vars[model_id].get()
        threading.Thread(target=self._toggle_model_worker, args=(model_id, enabled), daemon=True).start()

    def _toggle_model_worker(self, model_id: str, enabled: bool) -> None:
        if self._is_local_mode():
            models_manager.set_enabled(model_id, enabled)
            return

        url = self.server_url.get().rstrip("/")
        try:
            requests.post(f"{url}/api/models/toggle", json={"model_id": model_id, "enabled": enabled},
                          timeout=config.API_TIMEOUT)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Xatolik", f"Modelni o'zgartirishda xatolik: {e}"))

    def _sync_model_checkboxes(self, models_list: list[dict]) -> None:
        for m in models_list:
            var = self.model_vars.get(m["id"])
            if var is not None and var.get() != m["enabled"]:
                var.set(m["enabled"])

    def _rebuild_camera_list(self, cams: list[dict]) -> None:
        self.camera_cache = {c["id"]: c for c in cams}
        self.camera_order = [c["id"] for c in cams]

        for w in self.camera_list_frame.winfo_children():
            w.destroy()

        if not cams:
            tk.Label(self.camera_list_frame, text="Hali kamera qo'shilmagan.\n'➕ Kamera qo'shish' tugmasini bosing.",
                     fg="#888888", bg="#2d2d2d", font=("Segoe UI", 9, "italic"),
                     justify=tk.LEFT, wraplength=250).pack(anchor=tk.W, pady=10, padx=5)
            return

        for c in cams:
            row = tk.Frame(self.camera_list_frame, bg="#3a3a3a" if c["id"] == self.selected_camera_id else "#2d2d2d")
            row.pack(fill=tk.X, pady=2, padx=2)

            dot_color = "#00e676" if c["connected"] else "#ff5555"
            tk.Label(row, text="●", fg=dot_color, bg=row["bg"], font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=(4, 4))

            name_text = c["name"] + (" ⛶" if c.get("has_zone") else "")
            name_lbl = tk.Label(row, text=name_text, fg="#ffffff", bg=row["bg"],
                                 font=("Segoe UI", 9, "bold"), anchor=tk.W, cursor="hand2")
            name_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            name_lbl.bind("<Button-1>", lambda e, cid=c["id"]: self._switch_to_single(cid))

            tk.Button(row, text="🔍", width=2, command=lambda cid=c["id"], cn=c["name"]: self.open_popout(cid, cn),
                      bg="#3a3d41", fg="#fff", relief=tk.FLAT, cursor="hand2").pack(side=tk.LEFT, padx=1)
            tk.Button(row, text="⛶", width=2, command=lambda cid=c["id"], cn=c["name"]: self.open_zone_editor(cid, cn),
                      bg="#3a3d41", fg="#fff", relief=tk.FLAT, cursor="hand2").pack(side=tk.LEFT, padx=1)
            tk.Button(row, text="🗑", width=2, command=lambda cid=c["id"], cn=c["name"]: self.remove_camera(cid, cn),
                      bg="#d13438", fg="#fff", relief=tk.FLAT, cursor="hand2").pack(side=tk.LEFT, padx=(1, 4))

            if c.get("last_error") and not c.get("connected"):
                tk.Label(self.camera_list_frame, text=f"   ⚠ {c['last_error']}", fg="#ffcc00",
                         bg="#2d2d2d", font=("Segoe UI", 8), wraplength=260, justify=tk.LEFT).pack(anchor=tk.W, padx=6)

    def open_add_camera_dialog(self) -> None:
        dlg = tk.Toplevel(self.root)
        dlg.title("Kamera qo'shish")
        dlg.configure(bg="#2d2d2d")
        dlg.geometry("440x230")
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(dlg, text="Kamera nomi:", bg="#2d2d2d", fg="#fff", font=("Segoe UI", 10)).pack(
            anchor=tk.W, padx=14, pady=(14, 2))
        name_var = tk.StringVar(value=f"Kamera{len(self.camera_order) + 1}")
        tk.Entry(dlg, textvariable=name_var, font=("Segoe UI", 10), bg="#3c3c3c", fg="#fff",
                  insertbackground="#fff").pack(fill=tk.X, padx=14)

        tk.Label(dlg, text="Manba (RTSP URL / video fayl yo'li / vebkamera raqami):",
                 bg="#2d2d2d", fg="#fff", font=("Segoe UI", 10)).pack(anchor=tk.W, padx=14, pady=(10, 2))
        source_var = tk.StringVar(value=config.DEFAULT_CAMERA_SOURCE)

        tk.Entry(dlg, textvariable=source_var, font=("Segoe UI", 10), bg="#3c3c3c", fg="#fff",
                  insertbackground="#fff").pack(fill=tk.X, padx=14)

        tk.Label(dlg, text="Masalan: rtsp://user:parol@192.168.1.10:554/Streaming/Channels/102",
                 bg="#2d2d2d", fg="#888888", font=("Segoe UI", 8, "italic")).pack(anchor=tk.W, padx=14, pady=(2, 10))

        btn_frame = tk.Frame(dlg, bg="#2d2d2d")
        btn_frame.pack(fill=tk.X, padx=14, pady=6)

        def submit() -> None:
            name = name_var.get().strip()
            source = source_var.get().strip()
            if not name or not source:
                messagebox.showwarning("Ogohlantirish", "Nomi va manbasini kiriting!", parent=dlg)
                return
            dlg.destroy()
            threading.Thread(target=self._add_camera_worker, args=(name, source), daemon=True).start()

        tk.Button(btn_frame, text="✔ Qo'shish", command=submit, bg="#0e8a16", fg="#fff",
                  relief=tk.FLAT, padx=12, pady=5, cursor="hand2").pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="✖ Bekor qilish", command=dlg.destroy, bg="#3a3d41", fg="#fff",
                  relief=tk.FLAT, padx=12, pady=5, cursor="hand2").pack(side=tk.LEFT, padx=4)

    def _add_camera_worker(self, name: str, source: str) -> None:
        if self._is_local_mode():
            camera_manager.add_camera(name, source)
            self.root.after(0, self._poll_status_worker)
            return

        url = self.server_url.get().rstrip("/")
        try:
            res = requests.post(f"{url}/api/cameras", json={"name": name, "source": source}, timeout=config.API_TIMEOUT)
            if res.status_code != 200:
                err = res.json().get("detail", "Noma'lum xatolik")
                self.root.after(0, lambda: messagebox.showerror("Xatolik", f"Kamera qo'shilmadi: {err}"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Xatolik", f"Serverga ulanishda xatolik: {e}"))

    def remove_camera(self, cam_id: str, name: str) -> None:
        if not messagebox.askyesno("Tasdiqlash", f"'{name}' kamerasini o'chirmoqchimisiz?"):
            return
        if self.selected_camera_id == cam_id:
            self.back_to_grid()
        popout = self.popouts.pop(cam_id, None)
        if popout is not None:
            popout.on_close()
        threading.Thread(target=self._remove_camera_worker, args=(cam_id,), daemon=True).start()

    def _remove_camera_worker(self, cam_id: str) -> None:
        if self._is_local_mode():
            camera_manager.remove_camera(cam_id)
            self.root.after(0, self._poll_status_worker)
            return

        url = self.server_url.get().rstrip("/")
        try:
            requests.delete(f"{url}/api/cameras/{cam_id}", timeout=config.API_TIMEOUT)
        except Exception:
            pass

    def open_popout(self, cam_id: str, cam_name: str) -> None:
        existing = self.popouts.get(cam_id)
        if existing is not None:
            existing.lift()
            existing.focus_force()
            return
        self.popouts[cam_id] = CameraPopout(self, cam_id, cam_name)

    def open_zone_editor(self, cam_id: str, cam_name: str) -> None:
        ZoneEditorWindow(self, cam_id, cam_name)

    def _switch_to_single(self, cam_id: str) -> None:
        self.view_mode = "single"
        self.selected_camera_id = cam_id
        name = self.camera_cache.get(cam_id, {}).get("name", cam_id)
        self.view_status_label.config(text=f"📷 {name} (rasmga bosing -> orqaga qaytish)")
        self.back_btn.config(state=tk.NORMAL)
        self._rebuild_camera_list(list(self.camera_cache.values()))

    def back_to_grid(self) -> None:
        self.view_mode = "grid"
        self.selected_camera_id = None
        self.view_status_label.config(text="🔲 Barcha kameralar (bosib kattalashtiring)")
        self.back_btn.config(state=tk.DISABLED)
        self._rebuild_camera_list(list(self.camera_cache.values()))

    def on_display_click(self, event) -> None:
        if self.view_mode == "single":
            self.back_to_grid()
            return

        n = len(self.camera_order)
        if n == 0 or not self._last_shown_w or not self._last_shown_h:
            return

        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        col = min(cols - 1, int(event.x / (self._last_shown_w / cols)))
        row = min(rows - 1, int(event.y / (self._last_shown_h / rows)))
        idx = row * cols + col
        if idx < n:
            self._switch_to_single(self.camera_order[idx])

    def _update_alerts(self, events: list[dict]) -> None:
        self.alerts_text.config(state=tk.NORMAL)
        self.alerts_text.delete("1.0", tk.END)
        if not events:
            self.alerts_text.insert(tk.END, "Hozircha hodisalar yo'q.\n")
        else:
            for ev in events[:40]:
                t = time.strftime("%H:%M:%S", time.localtime(ev.get("ts", time.time())))
                cam_name = ev.get("camera_name", "?")
                icon = MODEL_ICONS.get(ev.get("model", ""), "•")
                self.alerts_text.insert(tk.END, f"[{t}] {icon} {cam_name}: {ev.get('label', '')}\n")
        self.alerts_text.config(state=tk.DISABLED)

    def _preview_fetch_loop(self) -> None:
        while True:
            t0 = time.time()
            frame = None

            if self._is_local_mode():
                if self.view_mode == "single" and self.selected_camera_id:
                    cam = camera_manager.get_camera(self.selected_camera_id)
                    if cam is not None:
                        frame = cam.get_annotated_frame()
                else:
                    cams = camera_manager.list_cameras()
                    if not cams:
                        frame = None
                    else:
                        tw, th = config.GRID_TILE_W, config.GRID_TILE_H
                        cols = int(np.ceil(np.sqrt(len(cams))))
                        rows = int(np.ceil(len(cams) / cols))
                        tiles = []
                        for cam in cams:
                            f = cam.get_annotated_frame()
                            if f is None:
                                tile = np.zeros((th, tw, 3), dtype=np.uint8)
                                status = cam.last_error or "ulanmoqda..."
                                cv2.putText(tile, f"{cam.name}: {status}", (10, th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1, cv2.LINE_AA)
                            else:
                                tile = cv2.resize(f, (tw, th))
                            cv2.rectangle(tile, (0, 0), (tw, 26), (30, 30, 30), -1)
                            dot_color = (0, 230, 0) if cam.connected else (0, 0, 255)
                            cv2.circle(tile, (14, 13), 5, dot_color, -1)
                            cv2.putText(tile, cam.name, (26, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                            tiles.append(tile)
                        while len(tiles) < rows * cols:
                            tiles.append(np.zeros((th, tw, 3), dtype=np.uint8))
                        row_imgs = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
                        frame = np.vstack(row_imgs)

                if frame is not None:
                    with self._frame_lock:
                        self._latest_frame = frame

                elapsed = time.time() - t0
                target = (config.GUI_UPDATE_MS if self.view_mode == "single" else config.GRID_UPDATE_MS) / 1000
                time.sleep(max(0.005, target - elapsed))
                continue

            if not self.is_connected:
                time.sleep(0.3)
                continue

            interval = config.GRID_UPDATE_MS / 1000
            try:
                url = self.server_url.get().rstrip("/")
                if self.view_mode == "single" and self.selected_camera_id:
                    endpoint = f"{url}/api/cameras/{self.selected_camera_id}/preview?max_w=960&max_h=540"
                    interval = config.GUI_UPDATE_MS / 1000
                else:
                    endpoint = f"{url}/api/cameras_grid_preview"
                    interval = config.GRID_UPDATE_MS / 1000

                res = self._session.get(endpoint, timeout=config.API_TIMEOUT)
                if res.status_code == 200 and res.content:
                    frame = cv2.imdecode(np.frombuffer(res.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self._frame_lock:
                            self._latest_frame = frame
            except Exception:
                pass
            elapsed = time.time() - t0
            time.sleep(max(0.002, interval - elapsed))

    def update_display(self) -> None:
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()

        if frame is not None:
            h, w = frame.shape[:2]
            max_w = max(200, self.display_label.winfo_width() or 1200)
            max_h = max(200, self.display_label.winfo_height() or 600)
            scale = min(max_w / w, max_h / h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            if (new_w, new_h) != (w, h):
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                resized = frame
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self._photo_ref = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.display_label.config(image=self._photo_ref)
            self._last_shown_w, self._last_shown_h = new_w, new_h

        self.root.after(config.GUI_UPDATE_MS, self.update_display)

    def on_closing(self) -> None:
        for win in list(self.popouts.values()):
            try:
                win.on_close()
            except Exception:
                pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    MonitoringApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
