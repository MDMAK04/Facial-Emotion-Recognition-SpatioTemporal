import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Tuple

class KeyframeSelector:
    def __init__(
        self,
        img_size: Tuple[int, int] = (128, 128),
        ssim_threshold: float = 0.90,
        target_frames: int = 16,
        crop_face: bool = True
    ):
        self.img_size = tuple(img_size)
        self.ssim_threshold = ssim_threshold
        self.target_frames = target_frames
        self.crop_face = crop_face
        self.face_cascade = None

        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
            xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(xml_path):
                self.face_cascade = cv2.CascadeClassifier(xml_path)

    def _get_face_bbox(self, frame_rgb: np.ndarray) -> Tuple[int, int, int, int]:
        h, w, _ = frame_rgb.shape
        if self.face_cascade is not None:
            scale = 0.25
            small_gray = cv2.cvtColor(
                cv2.resize(frame_rgb, (int(w * scale), int(h * scale))),
                cv2.COLOR_RGB2GRAY
            )
            faces = self.face_cascade.detectMultiScale(
                small_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
            )
            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                x, y, fw, fh = int(x / scale), int(y / scale), int(fw / scale), int(fh / scale)
                mw, mh = int(0.15 * fw), int(0.15 * fh)
                return max(0, x - mw), max(0, y - mh), min(w, x + fw + mw), min(h, y + fh + mh)

        # Fallback calibré pour le cadrage studio fixe RAVDESS (1280x720)
        size = int(min(h, w) * 0.42)
        cx, cy = w // 2, int(h * 0.44)
        return max(0, cx - size), max(0, cy - size), min(w, cx + size), min(h, cy + size)

    def extract_keyframes(self, video_path: str) -> Tuple[np.ndarray, dict]:
        cap = cv2.VideoCapture(video_path)
        raw_frames = []
        bbox = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.crop_face:
                if bbox is None:
                    bbox = self._get_face_bbox(frame_rgb)
                x1, y1, x2, y2 = bbox
                cropped = frame_rgb[y1:y2, x1:x2]
            else:
                cropped = frame_rgb
            raw_frames.append(cv2.resize(cropped, self.img_size))
        cap.release()

        total_frames = len(raw_frames)
        if total_frames == 0:
            raise ValueError(f"Vidéo illisible : {video_path}")

        non_redundant = [raw_frames[0]]
        last_gray = cv2.cvtColor(raw_frames[0], cv2.COLOR_RGB2GRAY)

        for frame in raw_frames[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if ssim(last_gray, curr_gray) < self.ssim_threshold:
                non_redundant.append(frame)
                last_gray = curr_gray

        num_selected = len(non_redundant)
        pool = non_redundant if num_selected >= self.target_frames else raw_frames
        indices = np.linspace(0, len(pool) - 1, self.target_frames, dtype=int)
        final_frames = [pool[i] for i in indices]

        stats = {
            "total_raw_frames": total_frames,
            "after_ssim_filtering": num_selected,
            "reduction_ratio": round(1.0 - (num_selected / total_frames), 3)
        }
        return np.array(final_frames, dtype=np.uint8), stats