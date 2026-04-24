import os
import gradio as gr
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, RedirectResponse

from app.config import MACHINE_TYPES, N_MACHINE_TYPES, CONFIG, MODELS_DIR
from app.inference import load_models_and_stats, predict_single

# ── Load models once at startup ────────────────────────────────────────────────
print("\n🔄 Loading models...")
mt_models, REFERENCE_STATS, THRESHOLD = load_models_and_stats()
print("🚀 Ready.\n")

# ==============================================================================
# FastAPI Backend
# ==============================================================================
app = FastAPI(
    title       = "🎵 Acoustic Anomaly Detection API",
    description = "First-Shot Unsupervised Anomaly Detection — DCASE 2024 Task 2 | ESIB USJ",
    version     = "2.0.0",
)

@app.get("/")
def root():
    return RedirectResponse(url="/gradio")

@app.get("/health")
def health():
    return {
        "status"        : "ok",
        "models_loaded" : list(mt_models.keys()),
        "n_models"      : len(mt_models),
        "threshold"     : round(THRESHOLD, 4),
    }

@app.get("/config")
def get_config():
    return {
        "sample_rate"    : CONFIG["sample_rate"],
        "clip_duration"  : CONFIG["clip_duration"],
        "n_mels"         : CONFIG["n_mels"],
        "threshold_value": round(float(THRESHOLD), 4),
    }

@app.get("/machines")
def get_machines():
    return {"machine_types": MACHINE_TYPES, "n_types": N_MACHINE_TYPES}


@app.post("/score_auto")
async def score_audio_auto(file: UploadFile = File(...)):
    """
    Auto-detect machine type by running ALL 7 models.
    Uses raw MSE for machine selection (directly comparable across models).
    Uses normalized score for anomaly detection (per-machine threshold).
    """
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files accepted.")

    tmp_path = f"/tmp/{file.filename}"
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        # Run ALL 7 models and collect raw MSE
        all_raw_mse = {}
        all_results = {}
        for mt in MACHINE_TYPES:
            r = predict_single(
                tmp_path, mt, mt_models, REFERENCE_STATS, THRESHOLD
            )
            all_raw_mse[mt] = r["raw_mse"]
            all_results[mt] = r

        # Best machine = lowest raw MSE
        best_mt      = min(all_raw_mse, key=all_raw_mse.get)
        final_result = all_results[best_mt]

        final_result["all_machine_scores"]    = {
            mt: round(float(s), 6) for mt, s in all_raw_mse.items()
        }
        final_result["auto_detected_machine"] = best_mt
        final_result["detection_mode"]        = "auto"

        return JSONResponse(content=final_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/score")
async def score_audio(
    file         : UploadFile = File(...),
    machine_type : str        = Form(...),
):
    """Score with a manually specified machine type."""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files accepted.")
    if machine_type not in MACHINE_TYPES:
        raise HTTPException(status_code=400,
            detail=f"Unknown machine_type. Choose from: {MACHINE_TYPES}")

    tmp_path = f"/tmp/{file.filename}"
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        result = predict_single(
            tmp_path, machine_type, mt_models, REFERENCE_STATS, THRESHOLD
        )
        result["auto_detected_machine"] = machine_type
        result["all_machine_scores"]    = {}
        result["detection_mode"]        = "manual"
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==============================================================================
# Gradio UI
# ==============================================================================
def analyze(wav_file, machine_type):
    if wav_file is None:
        return "", "", "", ""

    try:
        with open(wav_file, "rb") as f:
            if machine_type == "🔍 Auto-Detect":
                response = requests.post(
                    "http://localhost:8000/score_auto",
                    files={"file": f},
                )
            else:
                response = requests.post(
                    "http://localhost:8000/score",
                    files={"file": f},
                    data={"machine_type": machine_type},
                )

        if response.status_code != 200:
            err = response.json().get("detail", "Unknown error")
            return f"❌ {err}", "", "", ""

        r       = response.json()
        best_mt = r["auto_detected_machine"]
        score   = r["anomaly_score"]
        thresh  = r["threshold"]
        gap     = score - thresh
        status  = "🔴  ANOMALY DETECTED" if r["is_anomaly"] else "🟢  MACHINE IS NORMAL"
        mode    = r.get("detection_mode", "auto")

        # ── Panel 1: Detection ─────────────────────────────────────────
        detection = f"""{status}
{"="*40}
Anomaly Score : {score:.4f}
Threshold     : {thresh:.4f}
Gap           : {gap:+.4f}
  (+ = anomaly  |  - = normal)

Raw MSE       : {r["raw_mse"]:.6f}
Mode          : {"🔍 Auto-Detect" if mode == "auto" else "👤 Manual"}
"""

        # ── Panel 2: Machine Recognition ───────────────────────────────
        all_scores = r.get("all_machine_scores", {})

        if all_scores:
            score_lines = "\n".join([
                f"  {'►' if mt == best_mt else ' '} {mt:<12} {s:.6f}"
                for mt, s in sorted(all_scores.items(), key=lambda x: x[1])
            ])
            recognition = f"""🤖  AUTO-DETECTED MACHINE
{"="*40}
  Detected As : {best_mt}
  Method      : Lowest raw reconstruction error
                (lower = better match)

All machine raw MSE scores:
{score_lines}
"""
        else:
            recognition = f"""👤  MANUAL SELECTION
{"="*40}
  Machine     : {best_mt}
  Mode        : Manually selected by user

  The selected model was used directly
  for anomaly detection.
"""

        # ── Panel 3: Explanation ───────────────────────────────────────
        freqs = r["explanation"]["approx_frequencies_hz"]
        desc  = r["explanation"]["description"]

        if r["is_anomaly"]:
            severity = "🔴 HIGH" if score > 3.0 else "🟡 MEDIUM" if score > 2.0 else "🟠 LOW"
            explanation = f"""⚠️  ANOMALY EXPLANATION
{"="*40}
Severity : {severity}

{desc}

Problematic Frequencies:
  {freqs} Hz

Possible causes:
  • Mechanical wear or damage
  • Misalignment or imbalance
  • Abnormal operating condition
  • Foreign object interference
"""
        else:
            explanation = f"""✅  NORMAL OPERATION
{"="*40}
The machine is operating within
its normal learned sound pattern.

Active Frequencies:
  {freqs} Hz

{desc}

No action required.
"""

        # ── Panel 4: Summary ───────────────────────────────────────────
        summary = f"""📋  FULL SUMMARY
{"="*40}
Machine        : {best_mt}
Mode           : {"🔍 Auto-Detect" if mode == "auto" else "👤 Manual"}
Status         : {"ANOMALY ⚠️" if r["is_anomaly"] else "NORMAL ✅"}
Score          : {score:.4f} / {thresh:.4f}

Detection:
  {"⚠️  ANOMALY — inspect machine" if r["is_anomaly"] else "✅ Normal operation"}

Problem frequencies:
  {freqs} Hz
"""
        return detection, recognition, explanation, summary

    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", ""


# ── Build Gradio Interface ─────────────────────────────────────────────────────
with gr.Blocks(title="Acoustic Anomaly Detection", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🎵 Acoustic Anomaly Detection System
    ### First-Shot Unsupervised Anomaly Detection Under Domain Shift
    **DCASE 2024 Task 2 · ESIB, Université Saint-Joseph de Beyrouth**
    Laetitia Daou · Georges-Anthony El Hajj
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📂 Input")
            audio_input = gr.Audio(
                label="Upload Machine Recording (.wav)",
                type="filepath",
            )
            machine_dropdown = gr.Dropdown(
                choices=["🔍 Auto-Detect"] + MACHINE_TYPES,
                value="🔍 Auto-Detect",
                label="Machine Type",
                info="Select manually if you know the machine, or leave as Auto-Detect",
            )
            analyze_btn = gr.Button(
                "🔍 Analyze Audio",
                variant="primary",
                size="lg",
            )
            gr.Markdown(f"""
            ---
            **How it works:**
            1. Upload a 10-second WAV file
            2. Select machine type or leave Auto-Detect
            3. Click **Analyze Audio**

            **Auto-Detect:** runs all 7 models,
            picks best match via reconstruction error.

            **Manual:** uses the selected model
            directly for reliable detection.

            **Threshold:** `{THRESHOLD:.4f}`
            - Score **above** → ⚠️ Anomaly
            - Score **below** → ✅ Normal
            """)

        with gr.Column(scale=2):
            gr.Markdown("## 📊 Results")
            with gr.Row():
                detection_box = gr.Textbox(
                    label="🚦 Anomaly Detection",
                    lines=10, interactive=False,
                )
                recognition_box = gr.Textbox(
                    label="🤖 Machine Type Recognition",
                    lines=10, interactive=False,
                )
            with gr.Row():
                explanation_box = gr.Textbox(
                    label="🔬 Problem Explanation",
                    lines=14, interactive=False,
                )
                summary_box = gr.Textbox(
                    label="📋 Full Summary",
                    lines=14, interactive=False,
                )

    analyze_btn.click(
        fn=analyze,
        inputs=[audio_input, machine_dropdown],
        outputs=[detection_box, recognition_box, explanation_box, summary_box],
    )

# ── Mount Gradio inside FastAPI ────────────────────────────────────────────────
app = gr.mount_gradio_app(app, demo, path="/gradio")
