# User Profiling System (Flask + Streamlit)

A demo-grade hybrid profiling system combining a rules engine and GNN embeddings. Backend runs on Flask with a persistent asyncio runner; frontend is a multi-page Streamlit app. Includes ops endpoints for ingestion/incremental loop control, graph refresh, fusion training, and graceful shutdown.

## Features
- Flask backend with async services via a background event loop
- Mock realtime data + ingestion cache + incremental learner (PyTorch Geometric)
- RuleEngine + FusionCore gating for hybrid scores
- SHAP-based explanations
- Streamlit frontend with status/ops panels
- CPU/GPU switch: `UPS_DEVICE=cpu|gpu` (default gpu)

## Quick Start

1) Backend (Python 3.9 recommended)
- cd backend
- pip install -r requirements.txt
- For CPU: set `UPS_DEVICE=cpu` (Windows PowerShell: `$env:UPS_DEVICE = "cpu"`)
- Start: `python -m app.main` (listens on http://localhost:5000)

2) Frontend
- cd frontend
- pip install -r requirements.txt
- streamlit run app.py
- Configure backend URL via `.streamlit/secrets.toml`:

```
[general]
backend_base_url = "http://localhost:5000"
```

or environment variable `BACKEND_BASE_URL`.

## Endpoints
- GET `/health` — overall status and device info
- Ops (prefix `/api/v1/operations`): status, ingestion start/stop, incremental start/stop, fusion train, rules refresh, explainer clear, shutdown
- Profiling (prefix `/api/v1`): user, recommendation, explain, rules CRUD, graph refresh

## Notes
- Torch/PyG must match your CUDA. For no CUDA, use CPU mode.
- Generated data and rules live under `backend/app/data_source/generated_data/`.

See `ARCHITECTURE.md` for full design and flows.
