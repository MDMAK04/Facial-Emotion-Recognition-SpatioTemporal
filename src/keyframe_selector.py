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

    def _detect_and_crop_face(self, frame_rgb: np.ndarray) -> np.ndarray:
        if not self.crop_face:
            return cv2.resize(frame_rgb, self.img_size)

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            margin_w = int(0.15 * w)
            margin_h = int(0.15 * h)
            x1 = max(0, x - margin_w)
            y1 = max(0, y - margin_h)
            x2 = min(frame_rgb.shape[1], x + w + margin_w)
            y2 = min(frame_rgb.shape[0], y + h + margin_h)
            cropped = frame_rgb[y1:y2, x1:x2]
        else:
            h, w, _ = frame_rgb.shape
            center_x, center_y = w // 2, h // 2
            size = min(h, w) // 2
            cropped = frame_rgb[
                max(0, center_y - size):min(h, center_y + size),
                max(0, center_x - size):min(w, center_x + size)
            ]

        return cv2.resize(cropped, self.img_size)

    def extract_keyframes(self, video_path: str) -> Tuple[np.ndarray, dict]:
        cap = cv2.VideoCapture(video_path)
        raw_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = self._detect_and_crop_face(frame_rgb)
            raw_frames.append(processed_frame)
        cap.release()

        total_frames = len(raw_frames)
        if total_frames == 0:
            raise ValueError(f"Impossible de lire la vidéo : {video_path}")

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