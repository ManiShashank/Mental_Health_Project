# Multimodal Analysis of Communication and Language Patterns for Personalized Mental Health Risk Screening

[![System Status](https://img.shields.io/badge/System-Healthy-emerald)](#)
[![Milestone](https://img.shields.io/badge/Milestone-1%20Foundation-blue)](#)
[![License](https://img.shields.io/badge/License-Academic%20Research-purple)](#)

> **IMPORTANT RESEARCH NOTICE**:  
> This platform is an **analytical and pre-screening research system**. It does **NOT** provide a medical or psychiatric diagnosis, clinical assessment, or emergency healthcare intervention. The final output describes observed communication, language, and acoustic patterns and deviations from an individual's personal historical baseline.

---

## Architecture Overview

This project implements a multi-tiered research architecture for analyzing dyadic communication across text and voice modalities:

```
[ React + TypeScript Frontend ] <====== WebSocket / REST ======> [ FastAPI Backend Core ]
                                                                          |
                                                                          v
                                                          [ Multimodal Analysis Engine ]
                                                          - NLP & Linguistic Stats
                                                          - Dual-Path Voice (ASR + Acoustic)
                                                          - Interaction Turn Dynamics
                                                          - Longitudinal Baseline (Z-Score)
                                                          - Explainable Profile Generator
```

---

## Quick Start (Milestone 1)

### 1. Backend Setup
```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
- Interactive API Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/api/v1/health`

### 2. Frontend Setup
```powershell
cd frontend
npm install
npm run dev
```
- Application UI: `http://localhost:5173`

### 3. Run Backend Tests
```powershell
cd backend
pytest tests -v
```

---

## Documentation Suite

- [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md): Comprehensive project specifications and mathematical formulation.
- [`docs/ALGORITHM_REGISTRY.md`](docs/ALGORITHM_REGISTRY.md): Comprehensive registry of all analytical algorithms and lifecycle statuses.
- [`docs/SYSTEM_ARCHITECTURE.md`](docs/SYSTEM_ARCHITECTURE.md): Architectural layout and module separation.
- [`docs/DEVELOPMENT_ROADMAP.md`](docs/DEVELOPMENT_ROADMAP.md): Milestone progress and roadmap.
- [`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md): REST and WebSocket API specifications.
- [`docs/TESTING.md`](docs/TESTING.md): Testing guidelines and test case matrices.
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): Research experiment logs.
