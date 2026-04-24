\# 🎵 Acoustic Anomaly Detection System

\*\*First-Shot Unsupervised Anomaly Detection Under Domain Shift\*\*  

DCASE 2024 Task 2 · ESIB, Université Saint-Joseph de Beyrouth  

Laetitia Daou · Georges-Anthony El Hajj



\---



\## 🚀 How to Run (Docker — Recommended)



\### Prerequisites

\- Install Docker Desktop: https://www.docker.com/products/docker-desktop/

\- No Python or GPU required — everything is inside the image



\### Step 1 — Pull the image

```bashdocker pull laetitiadaou123/acoustic-anomaly-detection:latest



\### Step 2 — Run the container

```bashdocker run -p 8000:8000 laetitiadaou123/acoustic-anomaly-detection:latest



\### Step 3 — Open in browser

\- \*\*Gradio UI:\*\* http://localhost:8000/gradio

\- \*\*API docs:\*\* http://localhost:8000/docs

\- \*\*Health check:\*\* http://localhost:8000/health



\---



\## 🎯 How to Use

1\. Go to http://localhost:8000/gradio

2\. Upload a 10-second WAV file of a machine sound

3\. Click \*\*Analyze Audio\*\*

4\. The system automatically:

&#x20;  - Identifies the machine type

&#x20;  - Detects if it is normal or anomalous

&#x20;  - Explains which frequencies are problematic



\---



\## 🤖 Supported Machine Types

ToyCar · ToyTrain · bearing · fan · gearbox · slider · valve



\---



\## 📡 API Endpoints

| Endpoint | Method | Description |

|----------|--------|-------------|

| /gradio | GET | Gradio web UI |

| /docs | GET | FastAPI docs |

| /health | GET | System health |

| /score\_auto | POST | Auto-detect + score WAV |

| /machines | GET | List machine types |



\---



\## 📊 Model Performance

| Model | Scoring | Overall AUC | Overall pAUC |

|-------|---------|-------------|--------------|

| Baseline AE | MSE | 0.5031 | 0.0371 |

| Multi-Task (Creative) | MSE | 0.5051 | 0.0391 |

| Multi-Task + k-NN ★ Best | k-NN | 0.5069 | 0.0560 |



\---



\## 🛠️ Tech Stack

\- PyTorch — CNN Autoencoder + Multi-Task Model

\- Librosa — Log-mel spectrogram extraction

\- FastAPI — REST API backend

\- Gradio — Web UI

\- Docker — Containerized deployment

