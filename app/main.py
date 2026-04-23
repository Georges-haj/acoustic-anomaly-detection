import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from app.config import MACHINE_TYPES, N_MACHINE_TYPES, CONFIG
from app.inference import load_models_and_stats, predict_single

print("\n🔄 Loading models...")
mt_models, REFERENCE_STATS, THRESHOLD = load_models_and_stats()
print("🚀 Ready.\n")

app = FastAPI(
    title       = "🎵 Acoustic Anomaly Detection API",
    description = "First-Shot Unsupervised Anomaly Detection — DCASE 2024 Task 2 | ESIB USJ",
    version     = "2.0.0",
)

@app.get("/")
def root():
    return {
        "service"       : "Acoustic Anomaly Detection",
        "version"       : "2.0.0",
        "machine_types" : MACHINE_TYPES,
        "docs"          : "/docs",
    }

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

@app.post("/score")
async def score_audio(
    file         : UploadFile = File(...),
    machine_type : str        = Form(...),
):
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
        result = predict_single(tmp_path, machine_type,
                                mt_models, REFERENCE_STATS, THRESHOLD)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)