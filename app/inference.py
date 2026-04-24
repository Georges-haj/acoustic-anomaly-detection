import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import librosa

from app.config import CONFIG, MACHINE_TYPES, MT_TO_IDX, MODELS_DIR, device
from app.model import MultiTaskAnomalyModel

MEL_FREQS = librosa.mel_frequencies(
    n_mels=CONFIG["n_mels"], fmin=0, fmax=CONFIG["fmax"]
)


def extract_log_mel(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=CONFIG["sample_rate"], mono=True)
    target_len = CONFIG["sample_rate"] * CONFIG["clip_duration"]
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="reflect")
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=CONFIG["n_mels"], n_fft=CONFIG["n_fft"],
        hop_length=CONFIG["hop_length"], win_length=CONFIG["win_length"],
        fmax=CONFIG["fmax"], power=CONFIG["power"],
    )
    log_mel  = librosa.power_to_db(mel, ref=np.max)
    log_mel -= log_mel.min()
    log_mel /= (log_mel.max() + 1e-8)
    return log_mel.astype(np.float32)


def load_models_and_stats():
    models_dict = {}
    for mt in MACHINE_TYPES:
        model_path = MODELS_DIR / f"mt_{mt}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing: {model_path}")
        model = MultiTaskAnomalyModel()
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
        models_dict[mt] = model
        print(f"  ✅ Loaded: mt_{mt}.pt")

    with open(MODELS_DIR / "reference_stats.json") as f:
        reference_stats = json.load(f)

    with open(MODELS_DIR / "threshold.json") as f:
        threshold = float(json.load(f)["threshold"])

    print(f"✅ All {len(models_dict)} models loaded. Threshold={threshold:.4f}")
    return models_dict, reference_stats, threshold


def predict_single(audio_path, machine_type, models_dict, reference_stats, threshold):
    feat = extract_log_mel(audio_path)
    x    = torch.FloatTensor(feat).unsqueeze(0).unsqueeze(0).to(device)

    model = models_dict[machine_type]
    model.eval()

    with torch.no_grad():
        recon, logits, _ = model(x)
        raw_score  = float(F.mse_loss(recon, x, reduction="none").mean().item())
        ref_mean   = reference_stats[machine_type]["mean"]
        ref_std    = reference_stats[machine_type]["std"]
        norm_score = (raw_score - ref_mean) / ref_std
        mt_probs   = F.softmax(logits, dim=-1).cpu().numpy()[0]

    is_anomaly   = norm_score > threshold
    decision     = "⚠️  CHECK THIS MACHINE" if is_anomaly else "✅ NORMAL"
    mt_prob_dict = {mt: round(float(mt_probs[MT_TO_IDX[mt]]), 4) for mt in MACHINE_TYPES}

    recon_np  = recon.cpu().squeeze().numpy()
    diff      = np.abs(feat - recon_np)
    band_err  = diff.mean(axis=1)
    top_bands = np.argsort(band_err)[-5:][::-1].tolist()
    top_hz    = [round(float(MEL_FREQS[b]), 1) for b in top_bands]

    return {
        "anomaly_score"     : round(float(norm_score), 4),
        "raw_mse"           : round(float(raw_score), 6),
        "threshold"         : round(float(threshold), 4),
        "is_anomaly"        : bool(is_anomaly),
        "decision"          : decision,
        "machine_type"      : machine_type,
        "classifier_output" : mt_prob_dict,
        "explanation": {
            "top_anomalous_mel_bands" : [int(b) for b in top_bands],
            "approx_frequencies_hz"   : top_hz,
            "description": (
                f"Reconstruction error is highest around "
                f"{min(top_hz):.0f}–{max(top_hz):.0f} Hz. "
                f"These frequency bands deviate most from "
                f"learned normal patterns for {machine_type}."
            ),
        },
    }  # ← this closing bracket was missing before