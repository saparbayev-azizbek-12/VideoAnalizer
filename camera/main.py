import os
import cv2
import time
import requests
import threading
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk, filedialog, messagebox

import config


class NVRClientApp:
    def __init__(self, root):
        self.root = root
        self.root.title(config.APP_TITLE)
        self.root.geometry("1500x920")
        self.root.configure(bg="#1e1e1e")

        self.server_url = tk.StringVar(value=config.DEFAULT_SERVER_URL)
        self.source_type = tk.StringVar(value="file")
        self.selected_file_path = tk.StringVar(value="")
        self.conf_threshold = tk.DoubleVar(value=config.CONF_THRESHOLD)
        self.min_area_ratio = tk.DoubleVar(value=config.MIN_BOX_AREA_RATIO)

        self.selected_channel = None
        self.is_connected = False
        self.is_running = False
        self.photo_ref = None
        self.lock = threading.Lock()
        self.latest_frame = None

        self._setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Permanent background thread for preview image fetching
        threading.Thread(target=self.fetch_preview_loop, daemon=True).start()

        # Periodic status check and preview refresh
        self.root.after(500, self.poll_server_status)
        self.root.after(config.GUI_UPDATE_MS, self.update_preview_display)

    def _setup_ui(self):
        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # ---------------------------------------------------------
        # TOP SERVER CONNECTION BAR
        # ---------------------------------------------------------
        conn_frame = tk.Frame(main_frame, bg="#252526", bd=1, relief=tk.SOLID)
        conn_frame.pack(fill=tk.X, pady=(0, 10), ipady=6, ipadx=10)

        tk.Label(
            conn_frame,
            text="🖥 Server URL (Ubuntu GPU):",
            font=("Segoe UI", 11, "bold"),
            fg="#ffffff",
            bg="#252526"
        ).pack(side=tk.LEFT, padx=(10, 5))

        self.url_entry = tk.Entry(
            conn_frame,
            textvariable=self.server_url,
            font=("Segoe UI", 11),
            width=30,
            bg="#3c3c3c",
            fg="#ffffff",
            insertbackground="#ffffff"
        )
        self.url_entry.pack(side=tk.LEFT, padx=5)

        self.check_conn_btn = tk.Button(
            conn_frame,
            text="🔌 Ulanishni Tekshirish",
            command=self.check_server_connection,
            font=("Segoe UI", 10, "bold"),
            bg="#007acc",
            fg="#ffffff",
            activebackground="#005999",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=3,
            cursor="hand2"
        )
        self.check_conn_btn.pack(side=tk.LEFT, padx=10)

        self.conn_status_label = tk.Label(
            conn_frame,
            text="● Tekshirilmadi",
            font=("Segoe UI", 11, "bold"),
            fg="#ffcc00",
            bg="#252526"
        )
        self.conn_status_label.pack(side=tk.RIGHT, padx=15)

        # ---------------------------------------------------------
        # SOURCE SELECTION & CONFIG PANEL
        # ---------------------------------------------------------
        config_frame = tk.Frame(main_frame, bg="#2d2d2d", bd=1, relief=tk.FLAT)
        config_frame.pack(fill=tk.X, pady=(0, 10), ipady=8, ipadx=10)

        # Left: Source Radio buttons
        source_box = tk.LabelFrame(
            config_frame,
            text=" Manba Tanlash ",
            font=("Segoe UI", 10, "bold"),
            fg="#00e676",
            bg="#2d2d2d",
            bd=1
        )
        source_box.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)

        tk.Radiobutton(
            source_box,
            text="Local Video Fayl (.mp4, .avi, .mkv)",
            variable=self.source_type,
            value="file",
            font=("Segoe UI", 10),
            fg="#ffffff",
            bg="#2d2d2d",
            selectcolor="#3c3c3c",
            activebackground="#2d2d2d",
            activeforeground="#ffffff"
        ).pack(anchor=tk.W, padx=10, pady=2)

        tk.Radiobutton(
            source_box,
            text="NVR RTSP Kanallari (1-8 kanallar)",
            variable=self.source_type,
            value="nvr",
            font=("Segoe UI", 10),
            fg="#ffffff",
            bg="#2d2d2d",
            selectcolor="#3c3c3c",
            activebackground="#2d2d2d",
            activeforeground="#ffffff"
        ).pack(anchor=tk.W, padx=10, pady=2)

        # Middle: File Picker & Controls
        file_box = tk.Frame(config_frame, bg="#2d2d2d")
        file_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        btn_select_file = tk.Button(
            file_box,
            text="📁 Video Fayl Tanlash...",
            command=self.select_video_file,
            font=("Segoe UI", 10, "bold"),
            bg="#3a3d41",
            fg="#ffffff",
            activebackground="#4e5155",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2"
        )
        btn_select_file.pack(anchor=tk.W, pady=(5, 2))

        self.file_path_label = tk.Label(
            file_box,
            textvariable=self.selected_file_path,
            font=("Segoe UI", 9, "italic"),
            fg="#aaaaaa",
            bg="#2d2d2d",
            anchor=tk.W,
            wraplength=450
        )
        self.file_path_label.pack(anchor=tk.W)

        # Right: Action Buttons
        actions_box = tk.Frame(config_frame, bg="#2d2d2d")
        actions_box.pack(side=tk.RIGHT, padx=10)

        self.start_btn = tk.Button(
            actions_box,
            text="▶ Serverda Qayta Ishlashni Boshlash",
            command=self.start_processing,
            font=("Segoe UI", 11, "bold"),
            bg="#0e8a16",
            fg="#ffffff",
            activebackground="#107c41",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=8,
            cursor="hand2"
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(
            actions_box,
            text="⏹ To'xtatish",
            command=self.stop_processing,
            font=("Segoe UI", 11, "bold"),
            bg="#d13438",
            fg="#ffffff",
            activebackground="#a80000",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=8,
            state=tk.DISABLED,
            cursor="hand2"
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.grid_btn = tk.Button(
            actions_box,
            text="📱 Barcha Kanallar (8 ta)",
            command=self.reset_to_grid_view,
            font=("Segoe UI", 11, "bold"),
            bg="#007acc",
            fg="#ffffff",
            activebackground="#005999",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=14,
            pady=8,
            cursor="hand2"
        )
        self.grid_btn.pack(side=tk.LEFT, padx=5)

        # ---------------------------------------------------------
        # PROGRESS BAR & STATS BAR
        # ---------------------------------------------------------
        stats_frame = tk.Frame(main_frame, bg="#252526", bd=1, relief=tk.FLAT)
        stats_frame.pack(fill=tk.X, pady=(0, 10), ipady=6, ipadx=10)

        self.progress_bar = ttk.Progressbar(stats_frame, mode="determinate")
        self.progress_bar.pack(fill=tk.X, padx=10, pady=(5, 5))

        self.stats_info_label = tk.Label(
            stats_frame,
            text="Holat: Tayyor | Saqlangan crops: 0 | Freymlar: 0 / 0",
            font=("Segoe UI", 11, "bold"),
            fg="#00e676",
            bg="#252526"
        )
        self.stats_info_label.pack(side=tk.LEFT, padx=10)

        # ---------------------------------------------------------
        # MAIN PREVIEW VIDEO DISPLAY
        # ---------------------------------------------------------
        display_frame = tk.Frame(main_frame, bg="#000000", bd=2, relief=tk.SUNKEN)
        display_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(display_frame, bg="#000000", cursor="hand2")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.video_label.bind("<Button-1>", self.on_video_click)

    def reset_to_grid_view(self):
        self.selected_channel = None

    def on_video_click(self, event):
        if self.source_type.get() != "nvr":
            return

        if self.selected_channel is not None:
            # Single camera view active -> clicking anywhere returns to 8-camera grid
            self.selected_channel = None
            return

        # 8-camera grid mode active -> calculate clicked channel tile
        if self.photo_ref is not None:
            w = self.photo_ref.width()
            h = self.photo_ref.height()
            if w > 0 and h > 0:
                col = min(3, int(event.x / (w / 4)))
                row = min(1, int(event.y / (h / 2)))
                ch_idx = row * 4 + col
                if ch_idx < len(config.CHANNELS):
                    self.selected_channel = config.CHANNELS[ch_idx]


    def check_server_connection(self):
        url = self.server_url.get().rstrip("/")
        try:
            res = requests.get(f"{url}/api/health", timeout=4)
            if res.status_code == 200:
                data = res.json()
                gpu_name = data.get("gpu_name", "Noma'lum")
                self.conn_status_label.config(
                    text=f"🟢 Ulangan: {gpu_name}",
                    fg="#00e676"
                )
                self.is_connected = True
            else:
                self.conn_status_label.config(text="🔴 Server Xatoligi", fg="#ff5555")
                self.is_connected = False
        except Exception as e:
            self.conn_status_label.config(text="🔴 Ulanib bo'lmadi", fg="#ff5555")
            self.is_connected = False

    def select_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Video Fayl Tanlang",
            filetypes=[("Video Fayllar", "*.mp4 *.avi *.mkv *.mov *.flv"), ("Barchasi", "*.*")]
        )
        if file_path:
            self.selected_file_path.set(file_path)
            self.source_type.set("file")

    def start_processing(self):
        url = self.server_url.get().rstrip("/")
        stype = self.source_type.get()

        if not self.is_connected:
            self.check_server_connection()
            if not self.is_connected:
                messagebox.showerror("Xatolik", "Serverga ulanib bo'lmadi. Server URL ni tekshiring!")
                return

        uploaded_filename = None

        if stype == "file":
            local_path = self.selected_file_path.get()
            if not local_path or not os.path.exists(local_path):
                messagebox.showwarning("Ogohlantirish", "Iltimos, avval video fayl tanlang!")
                return

            filename = os.path.basename(local_path)
            self.stats_info_label.config(text=f"Fayl serverga yuklanmoqda: {filename} ...", fg="#ffcc00")
            self.root.update_idletasks()

            try:
                with open(local_path, "rb") as f:
                    files = {"file": (filename, f, "video/mp4")}
                    res = requests.post(f"{url}/api/upload_video", files=files, timeout=600)
                if res.status_code == 200:
                    uploaded_filename = filename
                else:
                    messagebox.showerror("Xatolik", f"Faylni yuklashda xatolik: {res.text}")
                    return
            except Exception as e:
                messagebox.showerror("Xatolik", f"Serverga ulanishda xatolik: {e}")
                return

        # Start job request
        try:
            payload = {
                "source_type": stype,
                "filename": uploaded_filename or "",
                "conf_threshold": self.conf_threshold.get(),
                "min_area_ratio": self.min_area_ratio.get()
            }
            res = requests.post(f"{url}/api/start", data=payload, timeout=10)
            if res.status_code == 200 and res.json().get("ok"):
                self.is_running = True
                self.start_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.NORMAL)
            else:
                err = res.json().get("error", "Noma'lum xatolik")
                messagebox.showerror("Xatolik", f"Serverda jarayon boshlanmadi: {err}")
        except Exception as e:
            messagebox.showerror("Xatolik", f"Serverga so'rov yuborishda xatolik: {e}")

    def stop_processing(self):
        url = self.server_url.get().rstrip("/")
        try:
            res = requests.post(f"{url}/api/stop", timeout=5)
            if res.status_code == 200:
                self.is_running = False
                self.start_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
        except Exception:
            pass

    def poll_server_status(self):
        if self.is_connected:
            url = self.server_url.get().rstrip("/")
            try:
                res = requests.get(f"{url}/api/status", timeout=2)
                if res.status_code == 200:
                    data = res.json()
                    running = data.get("running", False)
                    saved = data.get("saved_count", 0)
                    proc = data.get("processed_frames", 0)
                    total = data.get("total_frames", 0)

                    self.is_running = running
                    if running:
                        self.start_btn.config(state=tk.DISABLED)
                        self.stop_btn.config(state=tk.NORMAL)
                        status_text = f"● Jarayon ishlamoqda | Saqlangan crops: {saved}"
                        if total > 0:
                            percent = int((proc / total) * 100)
                            self.progress_bar["value"] = percent
                            status_text += f" | Freymlar: {proc} / {total} ({percent}%)"
                        else:
                            self.progress_bar["value"] = 0
                            status_text += f" | Freymlar: {proc}"
                        self.stats_info_label.config(text=status_text, fg="#00e676")
                    else:
                        self.start_btn.config(state=tk.NORMAL)
                        self.stop_btn.config(state=tk.DISABLED)
                        if total > 0 and proc >= total:
                            self.progress_bar["value"] = 100
                            self.stats_info_label.config(
                                text=f"✅ Qayta ishlash yakunlandi! Saqlangan crops: {saved} ta",
                                fg="#00e676"
                            )
                        else:
                            self.stats_info_label.config(
                                text=f"Holat: To'xtatilgan | Saqlangan crops: {saved}",
                                fg="#ffcc00"
                            )
            except Exception:
                pass

        self.root.after(1000, self.poll_server_status)

    def fetch_preview_loop(self):
        while True:
            if not self.is_connected:
                time.sleep(0.5)
                continue
            url = self.server_url.get().rstrip("/")
            try:
                ch_param = (self.selected_channel or 0) if self.source_type.get() == "nvr" else 0
                res = requests.get(f"{url}/api/preview_image?channel={ch_param}", timeout=3)
                if res.status_code == 200 and res.content:
                    frame = cv2.imdecode(np.frombuffer(res.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.lock:
                            self.latest_frame = frame
            except Exception:
                pass
            time.sleep(0.08)

    def update_preview_display(self):
        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None

        if frame is not None:
            h, w = frame.shape[:2]
            max_w, max_h = 1350, 580
            scale = min(max_w / w, max_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h))

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=photo)
            self.photo_ref = photo

        self.root.after(config.GUI_UPDATE_MS, self.update_preview_display)

    def on_closing(self):
        self.stop_processing()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = NVRClientApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

