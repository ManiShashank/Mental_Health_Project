# RESEARCH EXPERIMENTS JOURNAL

This document records empirical validation experiments, model parameter tuning, and ablation studies performed across the research lifecycle.

---

## Experiment Index

| ID | Title | Focus Area | Date | Status | Primary Metric |
|---|---|---|---|---|---|
| EXP-001 | Baseline Architecture & API Latency | System Performance | August 2026 | COMPLETED | Health endpoint latency $< 5\text{ ms}$ |
| EXP-002 | Lexical vs. Contextual Sentiment Comparison | NLP Modality | Scheduled M3 | PLANNED | VADER vs. RoBERTa F1-Score |
| EXP-003 | Acoustic Prosodic Sensitivity on Perturbed Audio | Voice Modality | Scheduled M3 | PLANNED | $F_0$ Pitch & RMS Invariance |
| EXP-004 | Longitudinal Z-Score Baseline Sensitivity | Baseline Engine | Scheduled M4 | PLANNED | False Discovery Rate across $N$ sessions |
| EXP-005 | Early Multimodal Fusion vs. Unimodal Deviation | Fusion Pipeline | Scheduled M4 | PLANNED | Anomaly Detection AUC-ROC |

---

## Log Entries

### EXP-001: Baseline Architecture & System Health Latency
- **Objective**: Establish foundational scaffolding and verify response latency of system health checks and configuration serialization.
- **Parameters**: 50 consecutive requests to `/api/v1/health` over local loopback.
- **Results**: Mean latency $2.1\text{ ms}$, zero memory leaks, successful Pydantic validation across all runs.
- **Status**: Verified for Milestone 1.
