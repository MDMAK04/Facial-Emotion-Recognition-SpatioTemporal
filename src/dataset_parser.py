import os
import glob
import pandas as pd
from pathlib import Path

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

LABEL_TO_IDX = {v: i for i, v in enumerate(EMOTION_MAP.values())}
IDX_TO_LABEL = {i: v for i, v in enumerate(EMOTION_MAP.values())}

def parse_ravdess_directory(root_dir: str, train_actors_max: int = 19) -> pd.DataFrame:
    video_paths = glob.glob(os.path.join(root_dir, "**", "*.mp4"), recursive=True)
    records = []

    for path in sorted(video_paths):
        filename = Path(path).stem
        parts = filename.split("-")
        if len(parts) != 7:
            continue

        modality, vocal_channel, emotion_code, intensity, statement, repetition, actor = parts

        if vocal_channel != "01" or modality not in ["01", "02"]:
            continue

        actor_id = int(actor)
        emotion_name = EMOTION_MAP[emotion_code]
        split = "train" if actor_id <= train_actors_max else "val"

        records.append({
            "video_path": path,
            "filename": filename,
            "modality": modality,
            "emotion_code": emotion_code,
            "emotion": emotion_name,
            "label": LABEL_TO_IDX[emotion_name],
            "intensity": "normal" if intensity == "01" else "strong",
            "statement": int(statement),
            "repetition": int(repetition),
            "actor_id": actor_id,
            "split": split
        })

    df = pd.DataFrame(records)
    print(f"Total vidéos chargées : {len(df)}")
    print(f"Train (Acteurs 01-{train_actors_max}) : {len(df[df['split']=='train'])} vidéos")
    print(f"Val   (Acteurs {train_actors_max+1}-24) : {len(df[df['split']=='val'])} vidéos")
    return df