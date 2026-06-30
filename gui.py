from __future__ import annotations

import json
import re
import threading
import tkinter as tk
from tkinter import messagebox
from typing import Optional
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from metrics import MetricsCalculator
from opencv_detector import OpenCVHotspotDetector
from openvino_detector import OpenVINOYOLODetector
from robot_mapper import RobotTargetMapper
from thermal_generator import HotspotShape, ThermalImageGenerator
from yolo_detector import YOLOv8PyTorchDetector

# ── Design tokens ─────────────────────────────────────────────────────────────
C: dict[str, str] = {
    "bg":         "#0F1117",
    "surface":    "#1A1D2B",
    "card":       "#20243A",
    "border":     "#2A2F4A",
    "accent":     "#6366F1",
    "accent_dim": "#4F46E5",
    "success":    "#10B981",
    "warning":    "#F59E0B",
    "error":      "#EF4444",
    "text":       "#F1F5F9",
    "muted":      "#94A3B8",
    "dim":        "#475569",
    "sidebar":    "#131520",
    "header":     "#161929",
}
FF = "Segoe UI"
UI_STATE_FILE = ".physicalai_ui_state.json"


class ThermalHotspotDemo:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PhysicalAI  ·  Thermal Benchmarking Platform")
        self.root.configure(bg=C["bg"])

        self.generator = ThermalImageGenerator(width=320, height=240, noise_std=1.5)
        self.opencv_detector = OpenCVHotspotDetector()
        self.yolo_pytorch_detector: Optional[YOLOv8PyTorchDetector] = None
        self.yolo_openvino_detector: Optional[OpenVINOYOLODetector] = None
        self.robot_mapper = RobotTargetMapper()

        self.current_frame = None
        self.metrics_history: dict = {"opencv": [], "pytorch": [], "openvino": []}

        self._status_var = tk.StringVar(value="Initializing…")
        self._kpi: dict[str, tk.StringVar] = {
            k: tk.StringVar(value="—")
            for k in (
                "opencv_lat", "opencv_fps", "opencv_acc", "opencv_conf",
                "pytorch_lat", "pytorch_fps", "pytorch_acc", "pytorch_conf",
                "openvino_lat", "openvino_fps", "openvino_acc", "openvino_conf",
            )
        }
        self._img_figs: dict[str, Figure] = {}
        self._img_cvs: dict[str, FigureCanvasTkAgg] = {}
        self._sub_vars: dict[str, tk.StringVar] = {}
        self.hotspot_count: Optional[tk.Scale] = None
        self.noise_scale: Optional[tk.Scale] = None
        self.benchmark_samples: Optional[tk.IntVar] = None
        self._dot: Optional[tk.Label] = None
        self.metrics_text: Optional[tk.Text] = None
        self._vis_frame: Optional[tk.Frame] = None

        self._setup_ui()
        self._configure_window_size()
        self._initialize_default_models()

    def _configure_window_size(self) -> None:
        """First launch fits display; later launches restore last geometry."""
        self.root.update_idletasks()
        area_x, area_y, screen_w, screen_h = self._get_display_work_area()

        # Keep margins for taskbar/window chrome to avoid off-screen bottom overflow.
        max_w = max(860, screen_w - 12)
        max_h = max(540, screen_h - 12)
        min_w = min(1100, max(860, int(screen_w * 0.68)))
        min_h = min(700, max(520, int(screen_h * 0.62)))

        state = self._load_ui_state()
        restored = False

        if state and state.get("geometry"):
            parsed = self._parse_geometry(str(state["geometry"]))
            if parsed is not None:
                w, h, x, y = parsed
                w = max(min_w, min(w, max_w))
                h = max(min_h, min(h, max_h))
                x = min(max(area_x, x), max(area_x, area_x + screen_w - w))
                y = min(max(area_y, y), max(area_y, area_y + screen_h - h))
                self.root.geometry(f"{w}x{h}+{x}+{y}")
                restored = True

        if not restored:
            target_w = min(max_w, max(min_w, int(screen_w * 0.88)))
            target_h = min(max_h, max(min_h, int(screen_h * 0.80)))
            x = area_x + max(0, (screen_w - target_w) // 2)
            y = area_y + max(0, (screen_h - target_h) // 2)
            self.root.geometry(f"{target_w}x{target_h}+{x}+{y}")

        self.root.minsize(min_w, min_h)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after_idle(self._enforce_window_fit)

    def _get_display_work_area(self) -> tuple[int, int, int, int]:
        """Return (x, y, width, height) of usable display work area."""
        try:
            import ctypes
            from ctypes import wintypes

            class RECT(ctypes.Structure):
                _fields_ = [
                    ("left", wintypes.LONG),
                    ("top", wintypes.LONG),
                    ("right", wintypes.LONG),
                    ("bottom", wintypes.LONG),
                ]

            rect = RECT()
            ok = ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
            if ok:
                w = int(rect.right - rect.left)
                h = int(rect.bottom - rect.top)
                if w > 0 and h > 0:
                    return int(rect.left), int(rect.top), w, h
        except Exception:
            pass

        return 0, 0, int(self.root.winfo_screenwidth()), int(self.root.winfo_screenheight())

    def _enforce_window_fit(self) -> None:
        """Re-clamp size after widgets are laid out to avoid overflow on startup."""
        self.root.update_idletasks()
        area_x, area_y, area_w, area_h = self._get_display_work_area()
        parsed = self._parse_geometry(self.root.winfo_geometry())
        if parsed is None:
            return

        w, h, x, y = parsed
        max_w = max(860, area_w - 12)
        max_h = max(540, area_h - 12)
        w = min(w, max_w)
        h = min(h, max_h)
        x = min(max(area_x, x), max(area_x, area_x + area_w - w))
        y = min(max(area_y, y), max(area_y, area_y + area_h - h))
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    @staticmethod
    def _parse_geometry(geometry: str) -> Optional[tuple[int, int, int, int]]:
        match = re.match(r"^(\d+)x(\d+)([+-]\d+)([+-]\d+)$", geometry.strip())
        if not match:
            return None
        w, h, x, y = match.groups()
        return int(w), int(h), int(x), int(y)

    def _ui_state_path(self) -> Path:
        return Path(__file__).resolve().parent / UI_STATE_FILE

    def _load_ui_state(self) -> dict:
        path = self._ui_state_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_ui_state(self) -> None:
        path = self._ui_state_path()
        data = {"geometry": self.root.winfo_geometry()}
        try:
            path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_close(self) -> None:
        self._save_ui_state()
        self.root.destroy()

    # ═══════════════════════════════════════════════════════════════════════
    #  UI construction
    # ═══════════════════════════════════════════════════════════════════════

    def _setup_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

    # ── Sidebar ───────────────────────────────────────────────────────────

    def _build_sidebar(self) -> None:
        sb = tk.Frame(self.root, bg=C["sidebar"], width=204)
        sb.grid(row=0, column=0, sticky="ns")
        sb.grid_propagate(False)

        brand = tk.Frame(sb, bg=C["sidebar"])
        brand.pack(fill=tk.X, pady=(20, 0))
        tk.Label(brand, text="⬡", bg=C["sidebar"], fg=C["accent"],
                 font=(FF, 18, "bold")).pack(side=tk.LEFT, padx=(16, 6))
        tk.Label(brand, text="PhysicalAI", bg=C["sidebar"], fg=C["text"],
                 font=(FF, 13, "bold")).pack(side=tk.LEFT)

        tk.Frame(sb, bg=C["border"], height=1).pack(fill=tk.X, pady=14)
        tk.Label(sb, text="PARAMETERS", bg=C["sidebar"], fg=C["dim"],
                 font=(FF, 8, "bold")).pack(anchor=tk.W, padx=16, pady=(0, 6))
        self.hotspot_count = self._sidebar_slider(
            sb, "Hotspot Count", 1, 5, 1, integer=True)
        self.noise_scale = self._sidebar_slider(
            sb, "Noise Level", 0.0, 5.0, 1.5, res=0.1)

        tk.Frame(sb, bg=C["border"], height=1).pack(fill=tk.X, pady=14)
        tk.Label(sb, text="BENCHMARK", bg=C["sidebar"], fg=C["dim"],
                 font=(FF, 8, "bold")).pack(anchor=tk.W, padx=16, pady=(0, 6))
        spf = tk.Frame(sb, bg=C["sidebar"])
        spf.pack(fill=tk.X, padx=16, pady=(0, 10))
        tk.Label(spf, text="Samples", bg=C["sidebar"], fg=C["muted"],
                 font=(FF, 9)).pack(anchor=tk.W)
        self.benchmark_samples = tk.IntVar(value=100)
        tk.Spinbox(spf, from_=10, to=2000, increment=10,
                   textvariable=self.benchmark_samples, width=8,
                   bg=C["card"], fg=C["text"], buttonbackground=C["border"],
                   highlightthickness=1, highlightcolor=C["border"],
                   relief="flat", font=(FF, 10)).pack(fill=tk.X, pady=(2, 0))

        tk.Label(sb, text="v2.0  ·  OpenVINO Platform",
                 bg=C["sidebar"], fg=C["dim"],
                 font=(FF, 8)).pack(side=tk.BOTTOM, pady=12)

    def _sidebar_slider(self, parent: tk.Frame, label: str,
                         from_: float, to: float, default: float,
                         res: float = 1.0,
                         integer: bool = False) -> tk.Scale:
        frame = tk.Frame(parent, bg=C["sidebar"])
        frame.pack(fill=tk.X, padx=16, pady=(0, 10))
        tk.Label(frame, text=label, bg=C["sidebar"], fg=C["muted"],
                 font=(FF, 9)).pack(anchor=tk.W)
        s = tk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL,
                     resolution=res, bg=C["sidebar"], fg=C["text"],
                     troughcolor=C["border"], activebackground=C["accent"],
                     highlightthickness=0, bd=0, length=170,
                     sliderlength=14, font=(FF, 8))
        s.set(default)
        s.pack(fill=tk.X)
        return s

    # ── Main area ─────────────────────────────────────────────────────────

    def _build_main(self) -> None:
        self._main = tk.Frame(self.root, bg=C["bg"])
        self._main.grid(row=0, column=1, sticky="nsew")
        self._main.columnconfigure(0, weight=1)
        self._main.rowconfigure(1, weight=1)
        self._build_header()
        self._build_body()

    def _build_header(self) -> None:
        hdr = tk.Frame(self._main, bg=C["header"], height=58)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        hdr.columnconfigure(1, weight=1)

        tk.Label(hdr, text="Benchmark  /  Thermal Detection",
                 bg=C["header"], fg=C["text"],
                 font=(FF, 12, "bold")).grid(row=0, column=0,
                                             padx=20, pady=16, sticky="w")
        sf = tk.Frame(hdr, bg=C["header"])
        sf.grid(row=0, column=1, sticky="e", padx=8)
        self._dot = tk.Label(sf, text="●", bg=C["header"],
                              fg=C["success"], font=(FF, 10))
        self._dot.pack(side=tk.LEFT)
        tk.Label(sf, textvariable=self._status_var, bg=C["header"],
                 fg=C["muted"], font=(FF, 9)).pack(side=tk.LEFT, padx=(4, 0))

        bf = tk.Frame(hdr, bg=C["header"])
        bf.grid(row=0, column=2, padx=16)
        self._btn(bf, "▶  Detect",
                  self.run_detection).pack(side=tk.LEFT, padx=(0, 6))
        self._btn(bf, "⟳  Benchmark",
                  self.run_benchmark).pack(side=tk.LEFT)

    def _build_body(self) -> None:
        self._body = tk.Frame(self._main, bg=C["bg"])
        self._body.grid(row=1, column=0, sticky="nsew", padx=20, pady=12)
        self._body.columnconfigure(0, weight=1)
        self._body.rowconfigure(0, weight=1)
        self._body.columnconfigure(1, weight=0)
        self._build_vis_and_metrics(self._body)

    def _build_kpi_row(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent, bg=C["bg"])
        row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for i in range(3):
            row.columnconfigure(i, weight=1)

        # Per-model Latency + FPS cards
        for i, (mkey, mlabel, color) in enumerate([
            ("opencv",   "OpenCV",   "#10B981"),
            ("pytorch",  "PyTorch",  "#F59E0B"),
            ("openvino", "OpenVINO", "#EC4899"),
        ]):
            card = tk.Frame(row, bg=C["card"],
                             highlightbackground=C["border"],
                             highlightthickness=1)
            card.grid(row=0, column=i, padx=(0 if i == 0 else 8, 0),
                      sticky="nsew")
            tk.Frame(card, bg=color, height=3).pack(fill=tk.X)
            inner = tk.Frame(card, bg=C["card"])
            inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)
            tk.Label(inner, text=mlabel.upper(), bg=C["card"], fg=C["muted"],
                     font=(FF, 8, "bold")).pack(anchor=tk.W)

            grid = tk.Frame(inner, bg=C["card"])
            grid.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
            grid.columnconfigure(0, weight=1)
            grid.columnconfigure(1, weight=1)
            grid.rowconfigure(0, weight=1)
            grid.rowconfigure(1, weight=1)

            for idx, (title, unit, key_suffix, value_size) in enumerate([
                ("Latency", " ms", "lat", 16),
                ("Frame Rate", " fps", "fps", 16),
                ("Accuracy", " px", "acc", 14),
                ("Confidence", " score", "conf", 14),
            ]):
                r = idx // 2
                c = idx % 2
                cell = tk.Frame(grid, bg="#1B1F33", highlightbackground=C["border"], highlightthickness=1)
                cell.grid(row=r, column=c, sticky="nsew", padx=(0 if c == 0 else 4, 0),
                          pady=(0 if r == 0 else 4, 0))

                tk.Label(cell, text=title, bg="#1B1F33", fg=C["dim"],
                         font=(FF, 7, "bold")).pack(anchor=tk.W, padx=8, pady=(6, 0))
                rowv = tk.Frame(cell, bg="#1B1F33")
                rowv.pack(anchor=tk.W, padx=8, pady=(2, 7))
                tk.Label(rowv, textvariable=self._kpi[f"{mkey}_{key_suffix}"],
                         bg="#1B1F33", fg=C["text"],
                         font=(FF, value_size, "bold")).pack(side=tk.LEFT, anchor=tk.S)
                tk.Label(rowv, text=unit, bg="#1B1F33", fg=C["muted"],
                         font=(FF, 8)).pack(side=tk.LEFT, anchor=tk.S, pady=(0, 2))

    def _build_vis_and_metrics(self, parent: tk.Frame) -> None:
        self._vis_frame = tk.Frame(parent, bg=C["bg"])
        self._vis_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self._vis_frame.columnconfigure((0, 1, 2), weight=1)
        self._vis_frame.rowconfigure((0, 1), weight=1)

        right = tk.Frame(parent, bg=C["surface"],
                          highlightbackground=C["border"],
                          highlightthickness=1, width=268)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_propagate(False)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        tk.Label(right, text="RESULTS", bg=C["surface"], fg=C["dim"],
                 font=(FF, 8, "bold")).grid(row=0, column=0, padx=14,
                                             pady=(12, 4), sticky="w")
        tf = tk.Frame(right, bg=C["surface"])
        tf.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        tf.columnconfigure(0, weight=1)
        tf.rowconfigure(0, weight=1)
        self.metrics_text = tk.Text(
            tf, state=tk.DISABLED, wrap=tk.NONE,
            bg=C["surface"], fg=C["text"],
            font=("Consolas", 9), relief="flat",
            selectbackground=C["accent"])
        self.metrics_text.grid(row=0, column=0, sticky="nsew")

        for tag, fg, bold in [
            ("header", C["text"],    True),
            ("key",    "#818CF8",    True),
            ("val",    C["text"],    False),
            ("good",   C["success"], False),
            ("warn",   C["warning"], False),
            ("muted",  C["muted"],   False),
            ("dim",    C["dim"],     False),
        ]:
            self.metrics_text.tag_configure(
                tag, foreground=fg,
                font=("Consolas", 9, "bold" if bold else "normal"))

        for key, title, r, c, cs, color in [
            ("thermal",  "Input Thermal", 0, 0, 2, "#6366F1"),
            ("mask",     "Ground Truth",  0, 2, 1, "#0EA5E9"),
            ("opencv",   "OpenCV",        1, 0, 1, "#10B981"),
            ("pytorch",  "PyTorch",       1, 1, 1, "#F59E0B"),
            ("openvino", "OpenVINO",      1, 2, 1, "#EC4899"),
        ]:
            self._make_img_card(key, title, r, c, cs, color)

    def _make_img_card(self, key: str, title: str, row: int, col: int,
                        colspan: int, accent: str) -> None:
        card = tk.Frame(self._vis_frame, bg=C["card"],
                         highlightbackground=C["border"],
                         highlightthickness=1)
        card.grid(row=row, column=col, columnspan=colspan,
                  padx=(0 if col == 0 else 8, 0),
                  pady=(0 if row == 0 else 8, 0),
                  sticky="nsew")
        card.columnconfigure(0, weight=1)
        card.rowconfigure(1, weight=1)

        tbar = tk.Frame(card, bg=C["card"])
        tbar.grid(row=0, column=0, sticky="ew")
        tk.Frame(tbar, bg=accent, width=3).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(tbar, text=title, bg=C["card"], fg=C["text"],
                 font=(FF, 9, "bold")).pack(side=tk.LEFT, padx=10, pady=6)
        sub = tk.StringVar(value="—")
        self._sub_vars[key] = sub
        tk.Label(tbar, textvariable=sub, bg=C["card"], fg=C["muted"],
                 font=(FF, 8)).pack(side=tk.RIGHT, padx=10)

        fig = Figure(figsize=(4.2 if colspan == 2 else 2.8, 2.6), dpi=80)
        fig.patch.set_facecolor(C["card"])
        ax = fig.add_subplot(111)
        ax.set_facecolor("#161929")
        ax.text(0.5, 0.5, "—", ha="center", va="center",
                color=C["dim"], fontsize=10, transform=ax.transAxes)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
        fig.tight_layout(pad=0.3)

        cv = FigureCanvasTkAgg(fig, master=card)
        cv.get_tk_widget().configure(bg=C["card"], highlightthickness=0)
        cv.get_tk_widget().grid(row=1, column=0, sticky="nsew",
                                 padx=2, pady=(0, 2))
        cv.draw()
        self._img_figs[key] = fig
        self._img_cvs[key] = cv

        if key in ("opencv", "pytorch", "openvino"):
            metric_panel = tk.Frame(card, bg=C["card"])
            metric_panel.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))
            metric_panel.columnconfigure(0, weight=1)
            metric_panel.columnconfigure(1, weight=1)
            metric_panel.rowconfigure(0, weight=1)
            metric_panel.rowconfigure(1, weight=1)

            for idx, (title_txt, unit, key_suffix, value_size) in enumerate([
                ("Latency", "ms", "lat", 11),
                ("Frame Rate", "fps", "fps", 11),
                ("Accuracy", "px", "acc", 11),
                ("Confidence", "score", "conf", 11),
            ]):
                rr = idx // 2
                cc = idx % 2
                cell = tk.Frame(metric_panel, bg="#1B1F33", highlightbackground=C["border"],
                                highlightthickness=1)
                cell.grid(row=rr, column=cc, sticky="nsew",
                          padx=(0 if cc == 0 else 4, 0),
                          pady=(0 if rr == 0 else 4, 0))

                tk.Label(cell, text=title_txt, bg="#1B1F33", fg=C["dim"],
                         font=(FF, 7, "bold")).pack(anchor=tk.W, padx=6, pady=(4, 0))
                vrow = tk.Frame(cell, bg="#1B1F33")
                vrow.pack(anchor=tk.W, padx=6, pady=(1, 4))
                tk.Label(vrow, textvariable=self._kpi[f"{key}_{key_suffix}"],
                         bg="#1B1F33", fg=C["text"],
                         font=(FF, value_size, "bold")).pack(side=tk.LEFT)
                tk.Label(vrow, text=f" {unit}", bg="#1B1F33", fg=C["muted"],
                         font=(FF, 7)).pack(side=tk.LEFT, pady=(0, 1))

    # ── Helpers ───────────────────────────────────────────────────────────

    def _btn(self, parent: tk.Frame, text: str, cmd,
              primary: bool = True) -> tk.Label:
        bg = C["accent"] if primary else C["surface"]
        hbg = C["accent_dim"] if primary else C["card"]
        b = tk.Label(parent, text=text, bg=bg, fg=C["text"],
                      font=(FF, 9, "bold"), padx=14, pady=6, cursor="hand2")
        b.bind("<Button-1>", lambda e: cmd())
        b.bind("<Enter>",    lambda e: b.config(bg=hbg))
        b.bind("<Leave>",    lambda e: b.config(bg=bg))
        return b

    def _set_status(self, msg: str, busy: bool = False) -> None:
        self._status_var.set(msg)
        if self._dot:
            self._dot.config(fg=C["warning"] if busy else C["success"])

    def _update_kpi(self, results: dict, gt_x: float, gt_y: float) -> None:
        for mkey in ("opencv", "pytorch", "openvino"):
            if mkey in results:
                r = results[mkey]
                lat = r.inference_time_ms
                fps = 1000.0 / lat if lat > 0 else 0.0
                err = MetricsCalculator.localization_error(
                    r.center_x, r.center_y, gt_x, gt_y)
                self._kpi[f"{mkey}_lat"].set(f"{lat:.1f}")
                self._kpi[f"{mkey}_fps"].set(f"{fps:.0f}")
                self._kpi[f"{mkey}_acc"].set(f"{err:.1f}")
                self._kpi[f"{mkey}_conf"].set(f"{r.confidence:.2f}")
            else:
                self._kpi[f"{mkey}_lat"].set("—")
                self._kpi[f"{mkey}_fps"].set("—")
                self._kpi[f"{mkey}_acc"].set("—")
                self._kpi[f"{mkey}_conf"].set("—")

    def _draw(self, key: str, image: np.ndarray,
               cmap: str = "inferno",
               centers=None, result=None,
               subtitle: str = "") -> None:
        fig = self._img_figs[key]
        cv  = self._img_cvs[key]
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor("#161929")
        ax.imshow(image, cmap=cmap, aspect="auto")
        if centers:
            ax.scatter([p[0] for p in centers], [p[1] for p in centers],
                       c="#22D3EE", s=50, marker="*", zorder=5)
        if result is not None:
            ax.scatter([result.center_x], [result.center_y],
                       c="#EF4444", s=55, marker="x", zorder=6, linewidths=2)
            x, y, w, h = result.bbox
            ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                    edgecolor="#FCD34D", linewidth=1.5))
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
        ax.axis("off")
        fig.tight_layout(pad=0.3)
        cv.draw()
        self._sub_vars[key].set(subtitle)

    def _clear_card(self, key: str) -> None:
        fig = self._img_figs[key]
        cv  = self._img_cvs[key]
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor("#161929")
        ax.text(0.5, 0.5, "Run Detection", ha="center", va="center",
                color=C["dim"], fontsize=9, transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout(pad=0.3)
        cv.draw()
        self._sub_vars[key].set("—")

    # ═══════════════════════════════════════════════════════════════════════
    #  Core actions
    # ═══════════════════════════════════════════════════════════════════════

    def generate_new_frame(self) -> None:
        self.generator.noise_std = float(self.noise_scale.get())
        hotspot_count = int(self.hotspot_count.get())
        self.current_frame = self.generator.generate(
            hotspot_count=hotspot_count, shape=HotspotShape.CIRCULAR)
        self._draw("thermal", self.current_frame.image, cmap="inferno",
                    centers=self.current_frame.centers, subtitle="raw input")
        self._draw("mask", self.current_frame.mask, cmap="gray",
                    subtitle="ground truth")
        for k in ("opencv", "pytorch", "openvino"):
            self._clear_card(k)
            self._kpi[f"{k}_lat"].set("—")
            self._kpi[f"{k}_fps"].set("—")
            self._kpi[f"{k}_acc"].set("—")
            self._kpi[f"{k}_conf"].set("—")
        if not self._vis_frame.winfo_ismapped():
            self._vis_frame.grid()
        self._set_status("Frame generated")

    def _initialize_default_models(self) -> None:
        def _load() -> None:
            self.root.after(0, lambda: self._set_status(
                "Loading models…", busy=True))
            try:
                self.yolo_pytorch_detector = YOLOv8PyTorchDetector(device="cpu")
            except Exception as e:
                print(f"PyTorch load failed: {e}")
            self.load_openvino_model(
                model_path_raw="yolov8n_openvino_model/yolov8n.xml",
                show_success=False, show_errors=False)
            self.root.after(0, lambda: self._set_status("Ready"))
        threading.Thread(target=_load, daemon=True).start()

    def load_openvino_model(
        self,
        model_path_raw: Optional[str] = None,
        show_success: bool = True,
        show_errors: bool = True,
    ) -> None:
        from pathlib import Path
        try:
            if model_path_raw is None:
                model_path_raw = "yolov8n_openvino_model/yolov8n.xml"
            inp = Path(model_path_raw.strip())
            proj = Path(__file__).resolve().parent
            candidates = [inp]
            if not inp.is_absolute():
                candidates = [Path.cwd() / inp, proj / inp]
            resolved = next((p for p in candidates if p.exists()), None)
            if resolved is None:
                for fb in [
                    Path.cwd() / "yolov8n_openvino_model" / "yolov8n.xml",
                    proj / "yolov8n_openvino_model" / "yolov8n.xml",
                    Path.cwd() / "models" / "yolov8n.xml",
                    proj / "models" / "yolov8n.xml",
                ]:
                    if fb.exists():
                        resolved = fb
                        break
            if resolved is None:
                if show_errors:
                    messagebox.showerror("Error", "OpenVINO model not found.")
                return
            self.yolo_openvino_detector = OpenVINOYOLODetector(
                model_path=str(resolved.resolve()), device="CPU")
            if show_success:
                messagebox.showinfo("Success",
                                    f"OpenVINO model loaded!\n{resolved}")
        except Exception as e:
            if show_errors:
                messagebox.showerror("Error", str(e))

    def run_detection(self) -> None:
        # Always refresh input before detection so one click completes the full flow.
        self.generate_new_frame()
        self._set_status("Running…", busy=True)
        if not self._vis_frame.winfo_ismapped():
            self._vis_frame.grid()
        threading.Thread(target=self._run_detections_threaded,
                          daemon=True).start()

    def _run_detections_threaded(self) -> None:
        try:
            results = {}
            results["opencv"] = self.opencv_detector.detect(
                self.current_frame.image)
            if self.yolo_pytorch_detector is None:
                try:
                    self.yolo_pytorch_detector = YOLOv8PyTorchDetector(
                        device="cpu")
                except Exception as e:
                    print(f"PyTorch N/A: {e}")
            if self.yolo_pytorch_detector:
                results["pytorch"] = self.yolo_pytorch_detector.detect(
                    self.current_frame.image)
            if self.yolo_openvino_detector is None:
                # Ensure first detection can include OpenVINO even if async startup load is not ready yet.
                self.load_openvino_model(show_success=False, show_errors=False)
            if self.yolo_openvino_detector:
                results["openvino"] = self.yolo_openvino_detector.detect(
                    self.current_frame.image)
            self.root.after(0, lambda: self._display_results(results))
        except Exception as e:
            self.root.after(0,
                lambda: messagebox.showerror("Detection Error", str(e)))

    def run_benchmark(self) -> None:
        n = int(self.benchmark_samples.get())
        if n <= 0:
            messagebox.showwarning("Warning", "Sample count must be > 0")
            return
        self._set_status(f"Benchmarking {n} samples…", busy=True)
        threading.Thread(target=self._run_benchmark_threaded,
                          args=(n,), daemon=True).start()

    def _run_benchmark_threaded(self, sample_count: int) -> None:
        try:
            if self.yolo_pytorch_detector is None:
                try:
                    self.yolo_pytorch_detector = YOLOv8PyTorchDetector(
                        device="cpu")
                except Exception as e:
                    print(f"PyTorch benchmark init failed: {e}")
            if self.yolo_openvino_detector is None:
                self.load_openvino_model(show_success=False, show_errors=False)

            detectors = {
                "opencv":   self.opencv_detector,
                "pytorch":  self.yolo_pytorch_detector,
                "openvino": self.yolo_openvino_detector,
            }
            active_detectors = {k: v for k, v in detectors.items()
                                 if v is not None}
            if not active_detectors:
                self.root.after(0, lambda: messagebox.showerror(
                    "Benchmark Error", "No detector available"))
                return

            stats = {k: {"error": [], "latency": [], "fps": []}
                     for k in active_detectors}
            gen = ThermalImageGenerator(
                width=self.generator.width,
                height=self.generator.height,
                noise_std=float(self.noise_scale.get()),
                seed=20260630)
            hotspot_count = int(self.hotspot_count.get())

            warm_frame = gen.generate(hotspot_count=hotspot_count,
                                       shape=HotspotShape.CIRCULAR)
            for detector in active_detectors.values():
                try:
                    detector.detect(warm_frame.image)
                except Exception:
                    pass

            for i in range(sample_count):
                frame = gen.generate(hotspot_count=hotspot_count,
                                     shape=HotspotShape.CIRCULAR)
                gt_x, gt_y = frame.centers[0]
                for key, detector in active_detectors.items():
                    result = detector.detect(frame.image)
                    error = MetricsCalculator.localization_error(
                        result.center_x, result.center_y, gt_x, gt_y)
                    latency = float(result.inference_time_ms)
                    fps = 1000.0 / latency if latency > 0 else 0.0
                    stats[key]["error"].append(error)
                    stats[key]["latency"].append(latency)
                    stats[key]["fps"].append(fps)
                if (i + 1) % 10 == 0:
                    pct = (i + 1) / sample_count * 100
                    self.root.after(0, lambda p=pct: self._set_status(
                        f"Benchmarking… {p:.0f}%", busy=True))

            report = self._format_benchmark_report(stats, sample_count)
            self.root.after(0, lambda: self._apply_benchmark_report(report))
            self.root.after(0, lambda: self._set_status("Benchmark complete"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Benchmark Error", str(e)))
            self.root.after(0, lambda: self._set_status("Error", busy=True))

    def _format_benchmark_report(self, stats: dict,
                                   sample_count: int) -> str:
        sep = "─" * 40
        lines = [sep, "  BENCHMARK  SUMMARY", sep,
                 f"  Samples  : {sample_count}",
                 "  Stream   : shared (seed=20260630)",
                 "  GT       : frame.centers[0]",
                 sep, ""]
        for key in ("opencv", "pytorch", "openvino"):
            if key not in stats:
                continue
            err = np.asarray(stats[key]["error"],   dtype=np.float32)
            lat = np.asarray(stats[key]["latency"], dtype=np.float32)
            fps = np.asarray(stats[key]["fps"],     dtype=np.float32)
            lines += [
                f"  {key.upper()}",
                f"  Error(px)  mean {np.mean(err):.2f}"
                f"  med {np.median(err):.2f}"
                f"  P95 {np.percentile(err, 95):.2f}",
                f"  Lat(ms)    mean {np.mean(lat):.2f}"
                f"  med {np.median(lat):.2f}"
                f"  P95 {np.percentile(lat, 95):.2f}",
                f"  FPS        mean {np.mean(fps):.1f}"
                f"  med {np.median(fps):.1f}",
                "",
            ]
        lines.append(sep)
        return "\n".join(lines)

    def _apply_benchmark_report(self, report: str) -> None:
        self._vis_frame.grid_remove()
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, report, "val")
        self.metrics_text.config(state=tk.DISABLED)

    def _display_results(self, results: dict) -> None:
        if not self._vis_frame.winfo_ismapped():
            self._vis_frame.grid()
        gt_x, gt_y = self.current_frame.centers[0]
        self._draw("thermal", self.current_frame.image, cmap="inferno",
                    centers=self.current_frame.centers, subtitle="raw input")
        self._draw("mask", self.current_frame.mask, cmap="gray",
                    subtitle="ground truth")
        best_err = min(
            (MetricsCalculator.localization_error(
                r.center_x, r.center_y, gt_x, gt_y)
             for r in results.values()),
            default=float("inf"),
        )
        tagged: list[tuple[str, str]] = [
            ("header", "DETECTION  RESULTS\n\n")]
        for key in ("opencv", "pytorch", "openvino"):
            if key not in results:
                self._clear_card(key)
                continue
            r = results[key]
            err = MetricsCalculator.localization_error(
                r.center_x, r.center_y, gt_x, gt_y)
            fps = 1000.0 / r.inference_time_ms \
                if r.inference_time_ms > 0 else 0.0
            self._draw(key, self.current_frame.image, cmap="inferno",
                        result=r,
                        subtitle=(f"{r.inference_time_ms:.1f}ms  "
                                  f"{err:.1f}px  confidence {r.confidence:.2f}"))
            robot = self.robot_mapper.pixel_to_robot(r.center_x, r.center_y)
            is_best = abs(err - best_err) < 0.01
            tagged += [
                ("key",   f"{key.upper()}\n"),
                ("muted", "  Latency    "),
                ("val",   f"{r.inference_time_ms:.2f} ms"),
                ("muted", f"  Frame Rate {fps:.0f} fps\n"),
                ("muted", "  Error      "),
                ("good" if is_best else "warn", f"{err:.2f} px"),
                ("dim",   " ★ best\n" if is_best else "\n"),
                ("muted", "  Confidence "),
                ("val",   f"{r.confidence:.3f}\n"),
                ("muted", "  Robot  "),
                ("dim",   "X "), ("val", f"{robot.X:.3f}  "),
                ("dim",   "Y "), ("val", f"{robot.Y:.3f}  "),
                ("dim",   "Z "), ("val", f"{robot.Z:.3f}\n\n"),
            ]
        self._update_kpi(results, gt_x, gt_y)
        self._set_status("Detection complete")
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        for tag, text in tagged:
            self.metrics_text.insert(tk.END, text, tag)
        self.metrics_text.config(state=tk.DISABLED)

    def clear_history(self) -> None:
        self.metrics_history = {"opencv": [], "pytorch": [], "openvino": []}


def main() -> None:
    root = tk.Tk()
    app = ThermalHotspotDemo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
