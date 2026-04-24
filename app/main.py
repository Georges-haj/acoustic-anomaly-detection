import os
import gradio as gr
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
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
    """Auto-detect machine type and score."""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files accepted.")

    tmp_path = f"/tmp/{file.filename}"
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        # Run one model just to get classifier output
        result = predict_single(
            tmp_path, MACHINE_TYPES[0], mt_models, REFERENCE_STATS, THRESHOLD
        )
        # Pick machine type with highest classifier confidence
        clf = result["classifier_output"]
        best_mt = max(clf, key=clf.get)

        # Re-run with the detected machine type
        final_result = predict_single(
            tmp_path, best_mt, mt_models, REFERENCE_STATS, THRESHOLD
        )
        return JSONResponse(content=final_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/score")
async def score_audio(
    file         : UploadFile = File(...),
    machine_type : str        = None,
):
    """Score with optional machine type (kept for API compatibility)."""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files accepted.")

    tmp_path = f"/tmp/{file.filename}"
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        if not machine_type or machine_type not in MACHINE_TYPES:
            # Auto-detect
            result = predict_single(
                tmp_path, MACHINE_TYPES[0], mt_models, REFERENCE_STATS, THRESHOLD
            )
            clf    = result["classifier_output"]
            machine_type = max(clf, key=clf.get)

        result = predict_single(
            tmp_path, machine_type, mt_models, REFERENCE_STATS, THRESHOLD
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==============================================================================
# Gradio UI — No dropdown, fully automatic
# ==============================================================================
def analyze(wav_file):
    if wav_file is None:
        return "", "", "", ""

    try:
        with open(wav_file, "rb") as f:
            response = requests.post(
                "http://localhost:8000/score_auto",
                files={"file": f},
            )

        if response.status_code != 200:
            err = response.json().get("detail", "Unknown error")
            return f"❌ {err}", "", "", ""

        r = response.json()

        # ── Panel 1: Detection ─────────────────────────────────────────
        status = "🔴  ANOMALY DETECTED" if r["is_anomaly"] else "🟢  MACHINE IS NORMAL"
        score  = r["anomaly_score"]
        thresh = r["threshold"]
        gap    = score - thresh

        detection = f"""{status}
{"="*40}
Anomaly Score : {score:.4f}
Threshold     : {thresh:.4f}
Gap           : {gap:+.4f}
  (+ = anomaly  |  - = normal)

Raw MSE       : {r["raw_mse"]:.6f}
"""

        # ── Panel 2: Machine Recognition ───────────────────────────────
        clf     = r["classifier_output"]
        top_mt  = max(clf, key=clf.get)
        top_pct = clf[top_mt] * 100

        clf_lines = "\n".join([
            f"  {'►' if mt == top_mt else ' '} {mt:<12} {prob*100:5.1f}%"
            for mt, prob in sorted(clf.items(), key=lambda x: -x[1])
        ])

        recognition = f"""🤖  IDENTIFIED MACHINE
{"="*40}
  Machine    : {top_mt}
  Confidence : {top_pct:.1f}%

All Probabilities:
{clf_lines}
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
Auto-Detected  : {top_mt} ({top_pct:.1f}%)
Status         : {"ANOMALY ⚠️" if r["is_anomaly"] else "NORMAL ✅"}
Score          : {score:.4f} / {thresh:.4f}

Detection:
  {"⚠️  ANOMALY — inspect machine" if r["is_anomaly"] else "✅ Normal operation"}

Recognition:
  {"Confident" if top_pct > 90 else "Uncertain"} about machine type

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
            analyze_btn = gr.Button(
                "🔍 Analyze Audio",
                variant="primary",
                size="lg",
            )
            gr.Markdown(f"""
            ---
            **How it works:**
            1. Upload a 10-second WAV file
            2. Click **Analyze Audio**
            3. The system auto-detects the machine type!

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
        inputs=[audio_input],          # ← no dropdown anymore
        outputs=[detection_box, recognition_box, explanation_box, summary_box],
    )

# ── Mount Gradio inside FastAPI ────────────────────────────────────────────────
app = gr.mount_gradio_app(app, demo, path="/gradio")