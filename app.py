import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import time
import datetime
from pathlib import Path
from collections import defaultdict

from model    import FabricClassifier
from analyzer import DefectEvent, analyse_session, export_csv

# ── Configuration ──────────────────────────────────────────────────
WEIGHTS_PATH   = "best.pt"
CAMERA_INDEX   = 1
DISPLAY_W      = 640
DISPLAY_H      = 480
INFERENCE_SKIP = 3       # run model every N frames
POLL_MS        = 30      # how often (ms) main thread polls the frame queue
# ───────────────────────────────────────────────────────────────────


class FabricInspectorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Fabric Defect Inspector")
        self.root.resizable(False, False)
        self.root.configure(bg="#1a1a2e")
        
        # ✅ Fix: Add these four lines so they exist before the camera starts
        self.last_probs     = {}
        self.last_is_fabric = True
        self.last_class     = None   
        self.last_conf      = 0.0    

        # Shared state between threads
        self._lock          = threading.Lock()
        self._latest_frame  = None   # annotated BGR frame ready to display
        self._latest_class  = None
        self._latest_conf   = 0.0
        self._fps           = 0.0

        self.is_running     = False
        self.cap            = None
        self.classifier     = None
        self.session_start  = None
        self.frame_count    = 0
        self.events         = []
        self._last_result   = None

        self._build_ui()
        self._load_model()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ─────────────────────────────────────────────────────────
    def _build_ui(self):
        DARK  = "#1a1a2e"
        PANEL = "#16213e"
        ACC   = "#0f3460"
        WHITE = "#e0e0e0"
        GREEN = "#00b894"
        RED   = "#d63031"

        # Top bar
        top = tk.Frame(self.root, bg=ACC, height=48)
        top.pack(fill='x')
        tk.Label(top, text="🧵  Fabric Defect Inspector",
                 bg=ACC, fg=WHITE,
                 font=("Segoe UI", 15, "bold")).pack(
                     side='left', padx=16, pady=8)
        self.status_lbl = tk.Label(top, text="● Idle",
                                   bg=ACC, fg="#b2bec3",
                                   font=("Segoe UI", 11))
        self.status_lbl.pack(side='right', padx=16)

        # Main area
        main = tk.Frame(self.root, bg=DARK)
        main.pack(fill='both', padx=12, pady=10)

        # ── Left: camera ──
        left = tk.Frame(main, bg=DARK)
        left.pack(side='left', padx=(0, 10))

        self.canvas = tk.Canvas(left,
                                width=DISPLAY_W, height=DISPLAY_H,
                                bg="#000", highlightthickness=1,
                                highlightbackground="#444")
        self.canvas.pack()

        # Stats strip
        stats = tk.Frame(left, bg=PANEL)
        stats.pack(fill='x', pady=(4, 0))
        self.fps_lbl    = self._stat(stats, "FPS",     "—")
        self.class_lbl  = self._stat(stats, "Defect",  "—")
        self.conf_lbl   = self._stat(stats, "Conf",    "—")
        self.frames_lbl = self._stat(stats, "Frames",  "0")
        self.dcount_lbl = self._stat(stats, "Defects", "0")

        # ── Right: controls + results ──
        right = tk.Frame(main, bg=DARK, width=320)
        right.pack(side='left', fill='y')
        right.pack_propagate(False)

        # Controls card
        ctrl = tk.LabelFrame(right, text="  Controls  ",
                             bg=PANEL, fg=WHITE,
                             font=("Segoe UI", 10, "bold"),
                             labelanchor='n', bd=0,
                             highlightbackground="#444",
                             highlightthickness=1)
        ctrl.pack(fill='x', pady=(0, 8))

        tk.Label(ctrl, text="Max market rate (₹):",
                 bg=PANEL, fg=WHITE,
                 font=("Segoe UI", 10)).pack(
                     anchor='w', padx=12, pady=(10, 2))
        self.rate_var = tk.StringVar(value="1000")
        tk.Entry(ctrl, textvariable=self.rate_var,
                 font=("Segoe UI", 12, "bold"),
                 bg="#0f3460", fg=WHITE, insertbackground=WHITE,
                 relief='flat', bd=4, width=14).pack(
                     anchor='w', padx=12, pady=(0, 8))

        tk.Label(ctrl, text="Confidence threshold:",
                 bg=PANEL, fg=WHITE,
                 font=("Segoe UI", 10)).pack(anchor='w', padx=12)
        self.conf_var = tk.DoubleVar(value=0.45)   # was 0.70
        ttk.Scale(ctrl, from_=0.4, to=0.95,
                  variable=self.conf_var,
                  orient='horizontal',
                  command=self._on_conf_change).pack(
                      fill='x', padx=12, pady=(2, 0))
        self.conf_lbl2 = tk.Label(ctrl, text="0.70",
                                  bg=PANEL, fg="#74b9ff",
                                  font=("Segoe UI", 10, "bold"))
        self.conf_lbl2.pack(anchor='e', padx=12, pady=(0, 8))

        self.timer_lbl = tk.Label(ctrl, text="00:00",
                                  bg=PANEL, fg="#74b9ff",
                                  font=("Segoe UI", 26, "bold"))
        self.timer_lbl.pack(pady=4)

        btn_row = tk.Frame(ctrl, bg=PANEL)
        btn_row.pack(fill='x', padx=12, pady=(4, 12))

        self.start_btn = tk.Button(
            btn_row, text="▶  Start",
            command=self._start,
            bg=GREEN, fg="white", activebackground="#00cec9",
            font=("Segoe UI", 12, "bold"),
            relief='flat', padx=16, pady=8, cursor='hand2')
        self.start_btn.pack(side='left', expand=True, fill='x', padx=(0, 4))

        self.stop_btn = tk.Button(
            btn_row, text="■  Stop",
            command=self._stop,
            bg=RED, fg="white", activebackground="#e17055",
            font=("Segoe UI", 12, "bold"),
            relief='flat', padx=16, pady=8, cursor='hand2',
            state='disabled')
        self.stop_btn.pack(side='left', expand=True, fill='x')

        # Results card
        res_card = tk.LabelFrame(right, text="  Analysis Results  ",
                                 bg=PANEL, fg=WHITE,
                                 font=("Segoe UI", 10, "bold"),
                                 labelanchor='n', bd=0,
                                 highlightbackground="#444",
                                 highlightthickness=1)
        res_card.pack(fill='both', expand=True)

        self.results_text = tk.Text(
            res_card,
            bg="#0d1117", fg=WHITE,
            font=("Consolas", 10),
            relief='flat', bd=6,
            state='disabled', wrap='word',
            width=36, height=20)
        self.results_text.pack(fill='both', expand=True, padx=4, pady=4)

        self.export_btn = tk.Button(
            right, text="📥  Export CSV Report",
            command=self._export_csv,
            bg=ACC, fg=WHITE,
            font=("Segoe UI", 10, "bold"),
            relief='flat', pady=6, cursor='hand2',
            state='disabled')
        self.export_btn.pack(fill='x', pady=(4, 0))

        self._write_results(
            "Results will appear here\nafter stopping the session.")

    def _stat(self, parent, label, value):
        PANEL = "#16213e"
        WHITE = "#e0e0e0"
        cell  = tk.Frame(parent, bg=PANEL)
        cell.pack(side='left', expand=True, fill='x', padx=1)
        tk.Label(cell, text=label, bg=PANEL, fg="#888",
                 font=("Segoe UI", 8)).pack()
        lbl = tk.Label(cell, text=value, bg=PANEL, fg=WHITE,
                       font=("Segoe UI", 11, "bold"))
        lbl.pack()
        return lbl

    # ── Model ──────────────────────────────────────────────────────
    def _load_model(self):
        if not Path(WEIGHTS_PATH).exists():
            messagebox.showerror("Model not found",
                                 f"Cannot find '{WEIGHTS_PATH}'.\n"
                                 "Place best.pt next to app.py.")
            return
        try:
            self.classifier = FabricClassifier(WEIGHTS_PATH, confidence=self.conf_var.get())
        except Exception as e:
            messagebox.showerror("Model error", str(e))

    # ── Start ──────────────────────────────────────────────────────
    def _start(self):
        if self.classifier is None:
            messagebox.showerror("Error", "Model not loaded.")
            return
        try:
            rate = float(self.rate_var.get())
            assert rate > 0
        except Exception:
            messagebox.showerror("Invalid input",
                                 "Enter a positive number for market rate.")
            return

        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error",
                                 f"Cannot open camera {CAMERA_INDEX}.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)

        # Reset
        self.events       = []
        self.frame_count  = 0
        self._frame_idx   = 0
        self.is_running   = True
        self.session_start = time.time()
        with self._lock:
            self._latest_frame = None
            self._latest_class = None
            self._latest_conf  = 0.0
            self._fps          = 0.0

        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.export_btn.config(state='disabled')
        self._set_status("● Monitoring", "#00b894")
        self._write_results("Session running…")

        # Camera thread — only captures and infers, never touches Tkinter
        threading.Thread(target=self._camera_loop,
                         daemon=True, name="CameraThread").start()

        # Main thread polls for new frames via after()
        self._poll_display()
        self._update_timer()

    # ── Camera thread ──────────────────────────────────────────────
    def _camera_loop(self):
        """
        Runs entirely in a background thread.
        Writes results to shared variables protected by _lock.
        Never calls any Tkinter method.
        """
        fps_timer  = time.time()
        fps_frames = 0
        frame_idx  = 0

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            self.frame_count += 1
            frame_idx        += 1
            fps_frames        += 1

            # Inference every N frames
            cls_name = None
            conf     = 0.0
            if frame_idx % INFERENCE_SKIP == 0:
                cls_name, conf, all_probs, is_fabric = self.classifier.predict(frame)
                self.last_probs      = all_probs
                self.last_is_fabric  = is_fabric
                self.last_class    = cls_name
                self.last_conf     = conf
                self.last_probs    = all_probs
                self.last_is_fabric= is_fabric

                if cls_name:
                    elapsed = time.time() - self.session_start
                    self.events.append(DefectEvent(elapsed, cls_name, conf))

            # Annotate frame
            annotated = self.classifier.annotate(
                frame,
                self.last_class,
                self.last_conf,
                getattr(self, 'last_probs', {}),
                getattr(self, 'last_is_fabric', True)
            )
            annotated = cv2.resize(annotated, (DISPLAY_W, DISPLAY_H))

            # FPS
            now = time.time()
            if now - fps_timer >= 1.0:
                fps = fps_frames / (now - fps_timer)
                fps_frames = 0
                fps_timer  = now
            else:
                fps = self._fps   # reuse last value

            # Write to shared state
            with self._lock:
                self._latest_frame = annotated.copy()
                self._latest_class = cls_name
                self._latest_conf  = conf
                self._fps          = fps

            time.sleep(0.005)   # ~200 fps ceiling, prevents CPU spin

    # ── Main-thread display poller ─────────────────────────────────
    def _poll_display(self):
        """Called by Tkinter's event loop every POLL_MS milliseconds."""
        if not self.is_running:
            return

        with self._lock:
            frame      = self._latest_frame
            cls_name   = self._latest_class
            conf       = self._latest_conf
            fps        = self._fps

        if frame is not None:
            # BGR → RGB → PhotoImage
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tk_img  = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.create_image(0, 0, anchor='nw', image=tk_img)
            self.canvas._img = tk_img      # keep reference — prevents GC

        # Update stats strip
        self.fps_lbl.config(text=f"{fps:.1f}")
        col = "#d63031" if cls_name else "#00b894"
        self.class_lbl.config(text=(cls_name or "Clean"), fg=col)
        self.conf_lbl.config(text=f"{conf:.2f}")
        self.frames_lbl.config(text=str(self.frame_count))
        self.dcount_lbl.config(text=str(len(self.events)))

        # Schedule next poll
        self.root.after(POLL_MS, self._poll_display)

    # ── Timer ──────────────────────────────────────────────────────
    def _update_timer(self):
        if not self.is_running or self.session_start is None:
            return
        elapsed    = int(time.time() - self.session_start)
        mins, secs = divmod(elapsed, 60)
        self.timer_lbl.config(text=f"{mins:02d}:{secs:02d}")
        self.root.after(1000, self._update_timer)

    # ── Stop ───────────────────────────────────────────────────────
    def _stop(self):
        self.is_running = False
        time.sleep(0.1)   # let camera thread finish current iteration
        if self.cap:
            self.cap.release()
            self.cap = None

        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self._set_status("● Analysing…", "#fdcb6e")
        threading.Thread(target=self._run_analysis,
                         daemon=True).start()

    # ── Analysis ───────────────────────────────────────────────────
    def _run_analysis(self):
        try:
            rate = float(self.rate_var.get())
        except Exception:
            rate = 1000.0

        duration = (time.time() - self.session_start
                    if self.session_start else 0.0)

        result = analyse_session(
            self.events, self.frame_count, duration, rate)
        self._last_result = result
        self.root.after(0, lambda: self._display_results(result))

    def _display_results(self, r):
        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "   INSPECTION REPORT",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Duration      : {r.duration_secs}s",
            f"  Total frames  : {r.total_frames}",
            f"  Clean frames  : {r.clean_frames}",
            f"  Defect frames : {r.defect_frames}",
            f"  Defect rate   : {r.defect_rate_pct}%",
            "",
            "  DEFECT BREAKDOWN",
            "  ─────────────────────────────",
        ]

        from analyzer import SEVERITY
        if r.defect_counts:
            for cls, count in r.defect_counts.most_common():
                lines.append(
                    f"  {cls:<16}: {count:>4}  (sev×{SEVERITY.get(cls,5)})")
        else:
            lines.append("  No defects detected ✓")

        lines += [
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Quality score  : {r.quality_score} / 100",
            f"  Quality grade  : {r.quality_grade}",
            "",
            f"  Max rate       : ₹{r.max_market_rate:,.2f}",
            f"  Suggested price: ₹{r.suggested_price:,.2f}",
            "",
            "  Press 'Export CSV' to save log.",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]

        self._write_results('\n'.join(lines))
        self._set_status("● Done", "#74b9ff")
        self.export_btn.config(state='normal')

        messagebox.showinfo(
            "Quality Result",
            f"Grade         : {r.quality_grade}\n"
            f"Quality score : {r.quality_score} / 100\n"
            f"Defect rate   : {r.defect_rate_pct}%\n\n"
            f"Max rate      : ₹{r.max_market_rate:,.2f}\n"
            f"Suggested     : ₹{r.suggested_price:,.2f}")

    # ── Helpers ────────────────────────────────────────────────────
    def _write_results(self, text: str):
        self.results_text.config(state='normal')
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state='disabled')

    def _set_status(self, text: str, color: str = "#b2bec3"):
        self.status_lbl.config(text=text, fg=color)

    def _on_conf_change(self, _=None):
        val = round(self.conf_var.get(), 2)
        self.conf_lbl2.config(text=str(val))
        if self.classifier:
            self.classifier.confidence = val

    def _export_csv(self):
        if not self._last_result:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=(
                f"report_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"))
        if path:
            export_csv(self._last_result, path)
            messagebox.showinfo("Saved", f"Report saved:\n{path}")

    def _on_close(self):
        self.is_running = False
        time.sleep(0.1)
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    FabricInspectorApp(root)
    root.mainloop()