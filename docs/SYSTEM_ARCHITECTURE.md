# SYSTEM ARCHITECTURE

## 1. High-Level Architecture

The system is structured as a decoupled, asynchronous micro-architecture comprising an interactive React single-page frontend, a high-throughput Python FastAPI backend, native WebSocket message brokers, modular analytical pipelines, and a structured relational data model.

```
+-------------------------------------------------------------------------------+
|                                React Frontend                                 |
|                                                                               |
|  +---------------------+  +----------------------+  +----------------------+  |
|  |  Dyadic Chat &      |  | Research Analytics   |  | Explainable Profile  |  |
|  |  Audio Recorder UI  |  | Dashboard & Visuals  |  | & Deviation Report   |  |
|  +----------+----------+  +----------+-----------+  +----------+-----------+  |
+-------------|------------------------|-------------------------|--------------+
              | (WebSocket / REST)     | (REST API)              | (REST API)
              v                        v                         v
+-------------------------------------------------------------------------------+
|                             FastAPI Backend Core                              |
|                                                                               |
|  +---------------------+  +----------------------+  +----------------------+  |
|  | Connection Manager  |  | REST Routing Layer   |  | Pydantic Validation  |  |
|  | (WebSockets Engine) |  | (/api/v1/...)        |  | & Config Management  |  |
|  +----------+----------+  +----------+-----------+  +----------+-----------+  |
+-------------|------------------------|-------------------------|--------------+
              |                        |                         |
              +-------------------+----+-------------------------+
                                  |
                                  v
+-------------------------------------------------------------------------------+
|                         Analytical Processing Pipeline                        |
|                                                                               |
|  +------------------------+  +------------------------+  +-----------------+  |
|  | NLP & Language Engine  |  | Dual-Path Voice Engine |  | Interaction &   |  |
|  | - Preprocessing        |  | - Path A: Whisper ASR  |  | Turn-Taking     |  |
|  | - TF-IDF & BERT        |  | - Path B: Librosa      |  | Dynamics Engine |  |
|  | - VADER & Transformer  |  |   (F0, RMS, MFCCs,     |  | (Cadence, Turn  |  |
|  | - LDA & Linguistic Stat|  |    Speech Rate)        |  |  Ratio, Latency)|  |
|  +-----------+------------+  +-----------+------------+  +--------+--------+  |
|              |                           |                        |           |
|              +--------------------+------+------------------------+           |
|                                   |                                           |
|                                   v                                           |
|              +----------------------------------------+                       |
|              |   Multimodal Feature Fusion Engine     |                       |
|              |   (Standardized Alignment & Vector)    |                       |
|              +--------------------+-------------------+                       |
|                                   |                                           |
|                                   v                                           |
|              +----------------------------------------+                       |
|              | Personal Baseline & Deviation Engine   |                       |
|              | (Longitudinal Z-Score & Moving Avgs)   |                       |
|              +--------------------+-------------------+                       |
|                                   |                                           |
|                                   v                                           |
|              +----------------------------------------+                       |
|              | Explainable Pre-Screening Profile Gen  |                       |
|              | (Transparent Rule-Based Provenance)    |                       |
|              +--------------------+-------------------+                       |
+-----------------------------------|-------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------------+
|                      Storage & Persistence Layer (PostgreSQL)                 |
|                                                                               |
|  - Users (Anonymized Participant IDs)     - Extracted Feature Vectors         |
|  - Conversations & Sessions               - Personal Baseline Statistics      |
|  - Text Messages & Voice Metadata         - Analytical Profile Reports        |
+-------------------------------------------------------------------------------+
```

---

## 2. WebSocket Real-Time Message Flow & Persistence (Milestone 2)

```
[ Browser A: Alice ]            [ FastAPI WebSocket Broker ]            [ Browser B: Bob ]
        |                                    |                                  |
        |--- 1. WS Connect (Room TEST001) -->|                                  |
        |<-- 2. History & user_joined -------|                                  |
        |                                    |<-- 3. WS Connect (TEST001) ------|
        |<-- 4. Broadcast: Bob Joined -------|--- 4. History & user_joined ---->|
        |                                    |                                  |
        |--- 5. Send: "Hello Bob" ---------->|                                  |
        |                                    |--- (Async DB Save Message)       |
        |<-- 6. Broadcast: "Hello Bob" ------|--- 6. Broadcast: "Hello Bob" --->|
        |                                    |                                  |
        |                                    |<-- 7. Send: "Hello Alice" -------|
        |                                    |--- (Async DB Save Message)       |
        |<-- 8. Broadcast: "Hello Alice" ----|--- 8. Broadcast: "Hello Alice" ->|
        |                                    |                                  |
        |--- 9. Empty Message ("  ") ------->|                                  |
        |<-- 10. Direct Error (Rejected) ----|  (No DB Save, No Broadcast)      |
```

---

## 3. Directory Structure & Module Separation

```
Mental_Health_Project/
├── docs/                                # Central documentation suite
│   ├── PROJECT_SPEC.md                  # Comprehensive academic specifications
│   ├── ALGORITHM_REGISTRY.md            # Traceable algorithm & feature registry
│   ├── SYSTEM_ARCHITECTURE.md           # Architecture and design specifications
│   ├── DEVELOPMENT_ROADMAP.md           # Phased milestone roadmap
│   ├── API_DOCUMENTATION.md             # REST and WebSocket API specifications
│   ├── TESTING.md                       # Test plan and verification guidelines
│   └── EXPERIMENTS.md                   # Research experiment log
├── backend/                             # Python FastAPI Backend
│   ├── app/
│   │   ├── api/                         # API routing and WebSocket endpoints
│   │   │   ├── v1/                      # Version 1 REST controllers
│   │   │   │   ├── health.py            # Health check & system info
│   │   │   │   ├── chat.py              # Messaging & history endpoints (Milestone 2)
│   │   │   │   ├── analysis.py          # Analytical triggers & retrieval (Milestone 3+)
│   │   │   │   └── dashboard.py         # Aggregated visual dashboard feeds
│   │   │   └── router.py                # Main v1 router aggregator
│   │   ├── core/                        # Core system utilities
│   │   │   ├── config.py                # Environment configuration via Pydantic
│   │   │   └── security.py              # Participant token/anonymization helpers
│   │   ├── schemas/                     # Pydantic data schemas (DTOs)
│   │   │   ├── health.py                # Health response contracts
│   │   │   ├── message.py               # Message and chat schemas (Milestone 2)
│   │   │   └── analysis.py              # Analytical feature and profile schemas
│   │   ├── analysis/                    # Isolated ML/NLP/Voice algorithms (Milestone 3+)
│   │   │   ├── nlp/                     # Text processing & linguistic models
│   │   │   ├── voice/                   # Audio & acoustic feature extractors
│   │   │   ├── communication/           # Interaction & turn dynamics
│   │   │   ├── baseline/                # Personal baseline & deviation engine
│   │   │   ├── fusion/                  # Multimodal feature fusion
│   │   │   ├── temporal/                # Time series & trend analysis
│   │   │   └── experimental/            # Isolated experimental models (Isolation Forest)
│   │   ├── db/                          # Database models & sessions (Milestone 2)
│   │   │   ├── session.py
│   │   │   └── models/
│   │   ├── services/                    # Business logic orchestration
│   │   │   ├── chat_manager.py          # WebSocket room broker
│   │   │   ├── speech_to_text/          # Speech-to-Text service & provider engine (Milestone 5)
│   │   │   │   ├── models.py
│   │   │   │   ├── exceptions.py
│   │   │   │   ├── service.py
│   │   │   │   └── providers/           # Base, Whisper, and Mock STT providers
│   │   │   ├── acoustic_features/       # Acoustic feature extraction engine (Milestone 6)
│   │   │   │   ├── models.py            # Typed metrics & 59-d vector builder
│   │   │   │   ├── exceptions.py
│   │   │   │   ├── service.py
│   │   │   │   └── extractors/          # Base, Librosa DSP, and Mock extractors
│   │   │   ├── nlp_features/            # NLP & linguistic extraction engine (Milestone 7)
│   │   │   │   ├── models.py            # Typed linguistic metrics & 66-d vector builder
│   │   │   │   ├── exceptions.py
│   │   │   │   ├── service.py
│   │   │   │   └── providers/           # Base, RuleLexical, Transformer, Mock providers
│   │   │   ├── multimodal_fusion/       # Multimodal feature fusion orchestrator (Milestone 8)
│   │   │   │   ├── models.py            # 140-d raw & normalized vector models
│   │   │   │   ├── exceptions.py
│   │   │   │   └── service.py           # Alignment, concatenation & canonical scaler
│   │   │   ├── personal_baseline/       # Personal baseline & temporal trend engine (Milestone 9)
│   │   │   │   ├── models.py            # Canonical registry & typed deviation/trend models
│   │   │   │   ├── engine.py            # Pure vectorized math (Bessel std, Z-score, OLS, EMA, SMA, RoC)
│   │   │   │   ├── exceptions.py
│   │   │   │   └── service.py           # Leakage-safe coordinator & snapshot persistence
│   │   │   └── analytical_report/       # Longitudinal analytical report engine (Milestone 10)
│   │   │       ├── models.py            # Analytical report DTOs & canonical domain models
│   │   │       ├── engine.py            # Pure mathematical aggregation & profiling engine
│   │   │       ├── formatters.py        # Multi-format report export (JSON, Markdown, HTML)
│   │   │       ├── exceptions.py
│   │   │       └── service.py           # Report orchestrator, authorization & snapshot cache
│   │   └── main.py                      # FastAPI application entrypoint
│   ├── storage/                         # Persistent local storage (Milestone 4)
│   │   └── audio/                       # Uploaded audio recordings (.webm, .wav, .ogg)
│   ├── tests/                           # Pytest automated test suite (132 tests)
│   │   ├── conftest.py
│   │   ├── test_health.py
│   │   ├── test_chat_ws.py
│   │   ├── test_communication_metrics.py
│   │   ├── test_voice_messages.py
│   │   ├── test_speech_to_text.py
│   │   ├── test_acoustic_features.py
│   │   ├── test_nlp_features.py
│   │   ├── test_multimodal_fusion.py
│   │   ├── test_personal_baseline.py
│   │   └── test_analytical_report.py
│   ├── requirements.txt
│   ├── pytest.ini
│   └── .env.example
├── frontend/                            # React + TypeScript + Vite Frontend
│   ├── src/
│   │   ├── components/                  # Reusable UI components
│   │   │   ├── Navbar.tsx
│   │   │   ├── Footer.tsx
│   │   │   ├── HealthBadge.tsx
│   │   │   ├── DisclaimerModal.tsx
│   │   │   ├── VoiceRecorder.tsx        # MediaRecorder audio controller (Milestone 4)
│   │   │   └── CommunicationAnalysisView.tsx # Step-by-step metrics breakdown
│   │   ├── pages/                       # Application views
│   │   │   ├── Home.tsx                 # Academic landing page
│   │   │   ├── SystemStatus.tsx         # Backend connectivity & health monitor
│   │   │   ├── Documentation.tsx        # In-app architecture & registry viewer
│   │   │   ├── ChatRoom.tsx             # Two-user dyadic chat with M9 baseline drawer & M10 report modal
│   │   │   └── CommunicationAnalysis.tsx # Dedicated metrics analysis view (Milestone 3)
│   │   ├── services/                    # Axios API client & WebSocket services
│   │   │   ├── api.ts
│   │   │   └── websocket.ts
│   │   ├── types/                       # TypeScript interface definitions
│   │   │   └── index.ts
│   │   ├── App.tsx                      # Route declarations
│   │   └── main.tsx                     # React DOM bootstrap
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
└── README.md
```

---

## 3. Component Boundaries & Isolation Principles

1. **Analytical Engine Decoupling**: All feature extraction, fusion, personal baseline calibration, temporal trend calculation, and analytical aggregation logic reside in pure, testable service layers (`backend/app/services/`) with vectorized functional interfaces free from HTTP or DB dependencies.
2. **Strict Downstream Immutability**: Each milestone operates strictly downstream from its predecessors:
   $$\text{M1/M2} \to \text{M3} \to \text{M4} \to \text{M5} \to \text{M6} \to \text{M7} \to \text{M8} \to \text{M9} \to \text{M10}$$
   Upstream records (raw audio, transcripts, acoustic records, NLP records, fused vectors, baseline snapshots) are strictly immutable.
3. **Point-in-Time Historical Reproducibility**: Milestone 10 reports store full snapshot payloads in `analytical_report_records` at generation time, ensuring that future observations added to the system never mutate historical analytical findings.
4. **Anonymized Identity Boundary**: User identity is represented exclusively via cryptographically generated anonymous Participant IDs (`PID-XXXXX`).
5. **Non-Diagnostic Research Boundary**: The platform operates strictly as a descriptive behavioral research tool. It enforces a strict non-diagnostic boundary prohibiting psychiatric diagnosis, clinical risk triage, or treatment recommendation, embedding mandatory verbatim research disclaimers in all exports and UI views.
