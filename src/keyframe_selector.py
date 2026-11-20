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
        self.img_size = img_size
        self.ssim_threshold = ssim_threshold
        self.target_frames = target_frames
        self.crop_face = crop_face
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _get_face_bbox(self, frame_rgb: np.ndarray) -> Tuple[int, int, int, int]:
        """Détecte la zone du visage sur une image réduite (ultra-rapide)."""
        h, w, _ = frame_rgb.shape
        scale = 0.25  # Détection sur image 4x plus petite -> 16x plus rapide
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
            margin_w = int(0.15 * fw)
            margin_h = int(0.15 * fh)
            return (
                max(0, x - margin_w),
                max(0, y - margin_h),
                min(w, x + fw + margin_w),
                min(h, y + fh + margin_h)
            )
        else:
            size = min(h, w) // 2
            cx, cy = w // 2, h // 2
            return (max(0, cx - size), max(0, cy - size), min(w, cx + size), min(h, cy + size))

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
            raise ValueError(f"Impossible de lire la vidéo : {video_path}")

        # Filtrage de redondance par SSIM
        non_redundant_frames = [raw_frames[0]]
        last_gray = cv2.cvtColor(raw_frames[0], cv2.COLOR_RGB2GRAY)

        for frame in raw_frames[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            score = ssim(last_gray, curr_gray)
            if score < self.ssim_threshold:
                non_redundant_frames.append(frame)
                last_gray = curr_gray

        num_selected = len(non_redundant_frames)
        if num_selected >= self.target_frames:
            indices = np.linspace(0, num_selected - 1, self.target_frames, dtype=int)
            final_frames = [non_redundant_frames[i] for i in indices]
        else:
            indices = np.linspace(0, total_frames - 1, self.target_frames, dtype=int)
            final_frames = [raw_frames[i] for i in indices]

        stats = {
            "total_raw_frames": total_frames,
            "after_ssim_filtering": num_selected,
            "final_sequence_len": len(final_frames),
            "reduction_ratio": round(1 - (num_selected / total_frames), 3)
        }

        return np.array(final_frames, dtype=np.uint8), stats