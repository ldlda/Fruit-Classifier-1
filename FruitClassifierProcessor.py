import logging
import time
from typing import Optional

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from utils import MODEL_CONFIG, get_preprocess_fn, load_my_labels, load_my_model

logger = logging.getLogger(__name__)


class FruitClassifierProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        # Initial defaults; the page will update these via setter methods.
        self.model_name = "MobileNetV2"
        self.model_config = MODEL_CONFIG[self.model_name]
        self.model = load_my_model(self.model_config["file"])
        self.labels = load_my_labels()
        self.preprocess_fn = get_preprocess_fn(self.model_config["family"])
        self.size = self.model_config["size"]

        # Throttling/polling rate in seconds (default ~5 FPS). Can be overridden via session_state:
        # - st.session_state["realtime_poll_hz"] (e.g., 5 for 5 FPS)
        # - or st.session_state["realtime_poll_ms"] (e.g., 200 for 200 ms)
        self.poll_interval_s = 0.2

        # Maintain last prediction to reuse when skipping frames
        self._last_pred_label: Optional[str] = None
        self._last_pred_conf: Optional[float] = None
        self._last_processed_ts: float = 0.0

        # Static ROI ("hitbox") as relative coords (x, y, w, h) in [0,1].
        # Defaults to centered 60% region.
        self.roi_rel = (0.2, 0.2, 0.6, 0.6)

        # Whether to select the latest frame in the queue (lower latency) vs first (smoother)
        self.pick_latest = True

    # --- public setters called from the Streamlit page thread ---
    def set_model(self, name: str) -> None:
        if name not in MODEL_CONFIG:
            logger.warning("Unknown model %s; keeping %s", name, self.model_name)
            return
        if name != self.model_name:
            self.model_name = name
            self.model_config = MODEL_CONFIG[self.model_name]
            self.model = load_my_model(self.model_config["file"])
            self.preprocess_fn = get_preprocess_fn(self.model_config["family"])
            self.size = self.model_config["size"]

    def set_fps(self, fps: float) -> None:
        if fps and fps > 0:
            self.poll_interval_s = max(1.0 / float(fps), 0.001)

    def set_roi(self, x: float, y: float, w: float, h: float) -> None:
        self.roi_rel = (
            float(np.clip(x, 0.0, 1.0)),
            float(np.clip(y, 0.0, 1.0)),
            float(np.clip(w, 0.0, 1.0)),
            float(np.clip(h, 0.0, 1.0)),
        )

    def set_pick_latest(self, val: bool) -> None:
        self.pick_latest = bool(val)

    def _draw_overlay(self, img_bgr: np.ndarray) -> None:
        """Draws ROI rectangle and last prediction text on the frame in-place."""
        h, w = img_bgr.shape[:2]
        rx, ry, rw, rh = self.roi_rel
        x0 = int(rx * w)
        y0 = int(ry * h)
        x1 = int(min(x0 + rw * w, w - 1))
        y1 = int(min(y0 + rh * h, h - 1))

        # ROI rectangle
        cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (0, 255, 255), 2)

        # Prediction text
        if self._last_pred_label is not None and self._last_pred_conf is not None:
            text = f"{self._last_pred_label} ({self._last_pred_conf:.2f})"
            cv2.putText(
                img_bgr,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    def _process_one_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process a single frame, respecting the polling interval and ROI."""
        # Convert the frame from the stream into a NumPy array (BGR)
        img_bgr = frame.to_ndarray(format="bgr24")

        # No Streamlit calls in this thread to avoid ScriptRunContext warnings.

        now = time.time()
        should_process = (now - self._last_processed_ts) >= self.poll_interval_s

        if should_process:
            # Compute ROI in absolute pixels
            h, w = img_bgr.shape[:2]
            rx, ry, rw, rh = self.roi_rel
            x0 = int(rx * w)
            y0 = int(ry * h)
            # Use exclusive end indices for slicing safety
            x1 = int(min(x0 + rw * w, w))
            y1 = int(min(y0 + rh * h, h))
            # Ensure valid slice
            x0 = max(0, min(x0, w - 1))
            y0 = max(0, min(y0, h - 1))
            x1 = max(x0 + 1, min(x1, w))
            y1 = max(y0 + 1, min(y1, h))

            roi_bgr = img_bgr[y0:y1, x0:x1]
            # --- PREPROCESS THE ROI ---
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            roi_resized = cv2.resize(roi_rgb, self.size)
            img_array = np.expand_dims(roi_resized, axis=0)
            img_preprocessed = self.preprocess_fn(img_array)

            # --- PREDICT ---
            predictions = self.model(img_preprocessed)[0]
            pred_index = int(np.argmax(predictions))
            pred_label = self.labels.get(pred_index, str(pred_index))
            confidence = float(np.max(predictions))

            # Cache results
            self._last_pred_label = pred_label
            self._last_pred_conf = confidence
            self._last_processed_ts = now

        # Draw overlay using latest predictions (new or cached)
        self._draw_overlay(img_bgr)
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    async def recv_queued(self, frames: list[av.VideoFrame]) -> list[av.VideoFrame]:  # type: ignore[override]
        """Process queued frames and return a list of frames.

        We only run the model on one selected frame per batch (first or latest)
        based on `self.pick_latest` and the polling interval, to reduce CPU.
        Other frames are returned unchanged for smooth streaming.
        """
        if not frames:
            return []

        # Pick which frame to run prediction on (if polling allows)
        idx = len(frames) - 1 if self.pick_latest else 0
        selected = frames[idx]

        # Run processing only once for the selected frame
        processed = self._process_one_frame(selected)

        frames[idx] = processed
        return frames

    def recv(
        self, frame: av.VideoFrame
    ) -> av.VideoFrame:  # Fallback if queued API not used
        return self._process_one_frame(frame)

    # Minimal transform for compatibility with VideoProcessorBase
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        return self._process_one_frame(frame).to_ndarray(format="bgr24")
