import torch
import os
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "sample_rate"           : 16000,
    "clip_duration"         : 10,
    "n_mels"                : 128,
    "n_fft"                 : 1024,
    "hop_length"            : 512,
    "win_length"            : 1024,
    "fmax"                  : 8000,
    "power"                 : 2.0,
    "latent_dim"            : 128,
    "dropout"               : 0.3,
    "anomaly_threshold_pct" : 90,
}

MACHINE_TYPES   = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
MT_TO_IDX       = {mt: i for i, mt in enumerate(MACHINE_TYPES)}
N_MACHINE_TYPES = len(MACHINE_TYPES)

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))