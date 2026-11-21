import os
import glob
import pandas as pd

EMOTION_MAP = {
    "01": 0,  # neutral
    "02": 1,  # calm
    "03": 2,  # happy
    "04": 3,  # sad
    "05": 4,  # angry
    "06": 5,  # fearful
    "07": 6,  # disgust
    "08": 7,  # surprised
}

EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]

def parse_ravdess_directory(data_dir: str, train_actors_max: int = 19) -> pd.DataFrame:
    video_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.mp4"), recursive=True))
    records = []

    for path in video_files:
        fname = os.path.basename(path).replace(".mp4", "")
        parts = fname.split("-")
        if len(parts) != 7:
            continue

        modality, vocal_channel, emotion, intensity, statement, repetition, actor = parts

        # On ne garde que les vidéos parlées (vocal_channel == '01') avec image ('01' ou '02')
        if vocal_channel != "01" or modality not in ("01", "02"):
            continue
        if emotion not in EMOTION_MAP:
            continue

        actor_id = int(actor)
        # Clé unique d'enregistrement (évite tout doublon entre Full-AV '01' et Video-only '02')
        clip_uid = f"{vocal_channel}-{emotion}-{intensity}-{statement}-{repetition}-{actor}"

        records.append({
            "clip_uid": clip_uid,
            "modality": modality,
            "video_path": path,
            "actor_id": actor_id,
            "label": EMOTION_MAP[emotion],
            "emotion": EMOTION_LABELS[EMOTION_MAP[emotion]],
            "split": "train" if actor_id <= train_actors_max else "val"
        })

    df = pd.DataFrame(records)
    if df.empty:
        raise FileNotFoundError(f"Aucune vidéo RAVDESS trouvée dans : {data_dir}")

    # Priorité à la modalité '01' (Full-AV) s'il y a doublon avec '02' (Video-only)
    df = df.sort_values("modality").drop_duplicates(subset=["clip_uid"], keep="first").reset_index(drop=True)
    return df