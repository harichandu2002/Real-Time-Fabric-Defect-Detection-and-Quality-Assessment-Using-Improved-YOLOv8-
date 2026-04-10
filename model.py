import cv2
import numpy as np
from ultralytics import YOLO

CLASS_NAMES = ['Broken_Thread', 'Hole', 'Misweave', 'Stain']
SEVERITY = {
    'Hole':          10,
    'Broken_Thread':  7,
    'Misweave':       5,
    'Stain':          3,
}

CLASS_COLORS_BGR = {
    'Hole':          (0,   215, 255),
    'Stain':         (147,  20, 255),
    'Broken_Thread': (0,   140, 255),
    'Misweave':      (209, 206,   0),
}

NO_FABRIC_THRESHOLD = 0.28   # below this = not fabric-like
CLAHE_ENGINE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_for_model(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Apply the EXACT same preprocessing used during TILDA training:
      1. Convert BGR → greyscale  (TILDA images are greyscale)
      2. Apply CLAHE on the greyscale channel
      3. Convert greyscale → 3-channel RGB  (model expects 3 channels)
      4. Letterbox-resize to 640×640 with pad value 114
      (YOLOv8 will then centre-crop to 224×224 internally)
    """
    # Step 1: BGR → greyscale (mimic TILDA greyscale nature)
    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Step 2: CLAHE contrast enhancement (exactly as in training pipeline)
    grey_clahe = CLAHE_ENGINE.apply(grey)

    # Step 3: greyscale → 3-channel (stack same channel 3 times)
    rgb = cv2.cvtColor(grey_clahe, cv2.COLOR_GRAY2RGB)

    # Step 4: letterbox to 640×640 preserving aspect ratio
    h, w = rgb.shape[:2]
    scale  = 640 / max(h, w)
    new_w  = int(w * scale)
    new_h  = int(h * scale)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((640, 640, 3), 114, dtype=np.uint8)
    pad_top  = (640 - new_h) // 2
    pad_left = (640 - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas  # returns RGB


class FabricClassifier:
    def __init__(self, weights_path: str, confidence: float = 0.45):
        self.model      = YOLO(weights_path, task='classify')
        self.confidence = confidence

    def predict(self, frame_bgr: np.ndarray):
        """
        Returns:
            class_name  – detected class name, or None if below threshold
            top_conf    – confidence of top-1 prediction (always returned)
            all_probs   – {class: probability} for all 4 classes
            is_fabric   – True if frame looks like fabric (top_conf >= NO_FABRIC_THRESHOLD)
        """
        try:
            # Apply training-consistent preprocessing
            processed = preprocess_for_model(frame_bgr)

            results = self.model(processed, verbose=False)
            if not results:
                return None, 0.0, {}, False

            probs = results[0].probs
            if probs is None:
                return None, 0.0, {}, False

            top1_id = int(probs.top1)
            top1_cf = float(probs.top1conf)

            # Build full probability dict
            all_probs = {}
            if hasattr(probs, 'data'):
                for i, p in enumerate(probs.data.tolist()):
                    all_probs[CLASS_NAMES[i]] = round(float(p), 3)

            is_fabric = top1_cf >= NO_FABRIC_THRESHOLD

            if top1_cf >= self.confidence:
                return CLASS_NAMES[top1_id], top1_cf, all_probs, is_fabric
            return None, top1_cf, all_probs, is_fabric

        except Exception:
            return None, 0.0, {}, False

    def annotate(self, frame_bgr: np.ndarray,
                 class_name, conf: float,
                 all_probs: dict = None,
                 is_fabric: bool = True) -> np.ndarray:

        frame  = frame_bgr.copy()
        h, w   = frame.shape[:2]
        ap     = all_probs or {}

        if not is_fabric:
            # Orange — not fabric
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 120, 210), -1)
            cv2.putText(frame,
                        "Point camera at woven fabric (close-up)",
                        (8, 33), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (2, 2), (w-2, h-2), (0, 120, 210), 3)

        elif class_name:
            # Defect detected
            color = CLASS_COLORS_BGR.get(class_name, (255, 255, 255))
            cv2.rectangle(frame, (0, 0), (w, 50), color, -1)
            cv2.putText(frame,
                        f"{class_name}   {conf:.2f}",
                        (8, 33), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (15, 15, 15), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (2, 2), (w-2, h-2), color, 3)

        else:
            # Fabric present but below threshold → show best guess
            if ap:
                top_cls  = max(ap, key=ap.get)
                top_p    = ap[top_cls]
                col_hint = CLASS_COLORS_BGR.get(top_cls, (180, 180, 180))
                cv2.rectangle(frame, (0, 0), (w, 50), (40, 170, 80), -1)
                cv2.putText(frame,
                            f"Clean  |  best guess: {top_cls} {top_p:.2f}",
                            (8, 33), cv2.FONT_HERSHEY_SIMPLEX,
                            0.68, (15, 15, 15), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (0, 0), (w, 50), (40, 170, 80), -1)
                cv2.putText(frame, f"Clean  {conf:.2f}", (8, 33),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (15, 15, 15), 2, cv2.LINE_AA)

        # ── Probability bars at bottom of frame ──────────────────
        if ap and is_fabric:
            bar_colors = {
                'Hole':          (0,   215, 255),
                'Stain':         (255,  20, 147),
                'Broken_Thread': (0,   140, 255),
                'Misweave':      (209, 206,   0),
            }
            panel_h = 54
            cv2.rectangle(frame, (0, h - panel_h), (w, h), (15, 15, 15), -1)
            bar_h = 10
            for i, cls in enumerate(CLASS_NAMES):
                p   = ap.get(cls, 0.0)
                bx  = int(i * w / 4) + 4
                bw  = int(w / 4) - 8
                by  = h - panel_h + 8
                # Background
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bar_h),
                              (55, 55, 55), -1)
                # Fill
                fill = max(1, int(bw * p))
                col  = bar_colors.get(cls, (200, 200, 200))
                cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h),
                              col, -1)
                # Threshold marker
                thr_x = bx + int(bw * self.confidence)
                cv2.line(frame, (thr_x, by - 2), (thr_x, by + bar_h + 2),
                         (255, 255, 255), 1)
                # Label
                short = {'Hole':'Hole','Stain':'Stain',
                         'Broken_Thread':'BrkThr','Misweave':'Miswv'}
                cv2.putText(frame,
                            f"{short.get(cls, cls)} {p:.2f}",
                            (bx, by + bar_h + 16),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.40, col, 1, cv2.LINE_AA)

        return frame