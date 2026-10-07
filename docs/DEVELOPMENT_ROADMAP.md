# DEVELOPMENT ROADMAP

This roadmap details the phased, incremental milestones for the **Multimodal Mental Health Pre-Screening Research Platform**.

---

## Milestone 1: Project Foundation
- **Deliverables**:
  - React + TypeScript + Vite frontend scaffolding.
  - FastAPI backend scaffolding with modular routing.
  - Health check (`/api/v1/health`) and System Info (`/api/v1/info`) endpoints.
  - Centralized Pydantic configuration (`app/core/config.py`).
  - Documentation suite: `PROJECT_SPEC.md`, `ALGORITHM_REGISTRY.md`, `SYSTEM_ARCHITECTURE.md`, `DEVELOPMENT_ROADMAP.md`, `API_DOCUMENTATION.md`, `TESTING.md`, `EXPERIMENTS.md`.
  - Academic landing page highlighting research objectives and non-diagnostic boundaries.
  - Automated tests for health and configuration.
- **Constraints**: No NLP, no voice audio processing, no DB, no ML models.
- **Status**: `COMPLETED & VALIDATED`

---

## Milestone 2: Communication Infrastructure & Persistence [COMPLETED]
- **Deliverables**:
  - Relational database schema with SQLAlchemy async ORM (`User`, `Room`, `Message`, `RoomParticipant`).
  - Two-user real-time WebSocket communication broker with room multiplexing.
  - Server-side message validation (empty message rejection, payload integrity).
  - UTC millisecond timestamping and conversation history retrieval.
  - React two-user chat interface with room lobby, online/reconnecting status indicators, distinct sender/receiver styling, and leave room controls.
  - Pytest automated test suite (11/11 tests passing) and live two-browser dyadic simulation script.
- **Constraints**: Strictly communication and persistence; no NLP, ML, voice extraction, or screening logic.
- **Status**: `COMPLETED & READY FOR AUDIT`

---

## Milestone 3: Conversation Data Management & Analysis-Ready Storage [COMPLETED]
- **Deliverables**:
  - Enriched conversation lifecycle management (`started_at`, `ended_at`, `status`, participant counts).
  - Deterministic communication metrics engine (`backend/app/analysis/communication/metrics.py`) computing total messages, messages by participant, message ratio, average/min/max lengths, duration, average/min/max turn latency, communication gaps ($\ge 60\text{s}$), and text/voice message counts.
  - REST analysis endpoints: `GET /api/v1/chat/rooms/{room_id}/analysis`, `GET /api/v1/chat/conversations/{room_id}`, `POST /api/v1/chat/conversations/{room_id}/end`.
  - Frontend Communication Analytics drawer integrated into the chat interface.
  - Automated tests covering all 12 test scenarios (14/14 tests passing).
- **Constraints**: Strictly communication and descriptive statistics; no NLP, BERT, VADER, RoBERTa, Whisper, acoustic feature extraction, or machine learning pre-screening.
- **Status**: `COMPLETED & READY FOR AUDIT`

---

## Milestone 4: Voice Data Capture & Persistence [COMPLETED]
- **Deliverables**:
  - Frontend voice recording controller (`VoiceRecorder.tsx`) supporting 7 discrete lifecycle states (Idle, Permission Request, Recording with Timer, Preview Player, Sending, Sent, Error).
  - Secure local audio storage pipeline (`backend/storage/audio/`) with MIME format validation (`audio/webm`, `audio/wav`, `audio/ogg`, `audio/mp4`, `audio/aac`, `audio/mpeg`) and size limit enforcement (15 MB).
  - Extended relational schema for `Message` table with audio metadata (`audio_url`, `audio_duration_seconds`, `audio_format`, `audio_size_bytes`).
  - REST endpoints: `POST /api/v1/chat/rooms/{room_id}/voice` (multipart upload and real-time broadcast) and `GET /api/v1/chat/audio/{filename}` (safe audio streaming).
  - Extended WebSocket message event protocol broadcasting voice messages in real-time to dyadic partners.
  - Seamless M3 communication metrics integration (tracking `text_message_count` and `voice_message_count`).
  - Comprehensive automated pytest suite (18/18 tests passing) and automated live 2-client simulation script.
- **Constraints**: Strictly audio capture, persistence, and playback; no acoustic feature extraction, Whisper transcription, or ML pre-screening.
- **Status**: `COMPLETED & VALIDATED`

---

## Milestone 5: Speech-to-Text (STT) [COMPLETED]
- **Deliverables**:
  - Modular Speech-to-Text service layer (`backend/app/services/speech_to_text/`) with provider abstraction (`BaseSTTProvider`, `WhisperSTTProvider`, `MockSTTProvider`).
  - Offline Whisper speech recognition pipeline transcribing persisted M4 audio recordings into text.
  - Database schema extension on `Message` (`transcript_text`, `transcription_status`, `transcription_model`, `transcription_language`, `transcription_error`, `transcribed_at`) with SQLite auto-migrations.
  - REST endpoint: `POST /api/v1/chat/messages/{message_id}/transcribe` for on-demand speech-to-text inference.
  - Real-time WebSocket event dispatch (`transcription_completed`) broadcasting completed transcripts to all room participants.
  - Frontend ChatRoom UI integration displaying transcripts, transcription statuses, and manual transcribe controls within voice cards.
  - Pytest automated test suite (25/25 tests passing) and automated live 2-client simulation script.
- **Constraints**: Strictly speech-to-text transcription; no acoustic feature extraction (M6), no NLP sentiment/emotion/topic analysis (M7), no clinical diagnoses.
- **Status**: `COMPLETED & VALIDATED`

---

## Milestone 6: Acoustic Feature Extraction [COMPLETED]
- **Deliverables**:
  - Modular Acoustic Features service layer (`backend/app/services/acoustic_features/`) with provider abstraction (`BaseAcousticExtractor`, `LibrosaAcousticExtractor`, `MockAcousticExtractor`).
  - Signal processing DSP algorithms for 6 acoustic feature families:
    1. Pitch ($F_0$) Statistics: mean, std, min, max, median, range, voicing fraction, local jitter via `librosa.pyin`.
    2. RMS Energy & Shimmer: mean, std, min, max, dynamic range, local shimmer via `librosa.feature.rms`.
    3. Estimated Speech Rate & Syllable Rhythm: vocalic peak detection via `scipy.signal.find_peaks`.
    4. Pause Dynamics: $-30\text{ dBFS}$ thresholding with minimum $150\text{ ms}$ silence run filtering.
    5. MFCC Representation: 13 cepstral coefficient means and standard deviations ($26$-d) via `librosa.feature.mfcc`.
    6. Spectral Descriptors: spectral centroid, bandwidth, rolloff ($85\%$), flatness, zero-crossing rate ($10$-d).
  - Standardized immutable raw feature vector (`v1.0-59d`) with exactly 59 finite numerical elements in indices 00–58.
  - Relational database entity `AcousticFeatureRecord` (`acoustic_features` table) and repository operations with idempotent upserts.
  - REST endpoints: `POST /api/v1/chat/messages/{id}/analyze-acoustic`, `GET /api/v1/chat/messages/{id}/acoustic-features`, and `GET /api/v1/chat/rooms/{room_id}/acoustic-features`.
  - WebSocket event dispatch (`acoustic_analysis_completed`) notifying dyadic clients in real time.
  - Frontend ChatRoom integration featuring collapsible 59-d acoustic breakdown with non-diagnostic research disclaimers.
  - Pytest automated test suite (34/34 tests passing) and automated live 2-client verification script.
- **Constraints**: Strictly physical/descriptive signal feature extraction; no NLP (M7), no multimodal fusion (M8), no personal baseline (M9), no medical/psychiatric diagnoses.
- **Status**: `COMPLETED & VALIDATED`

---

## Milestone 7: NLP & Language Pattern Analysis [COMPLETED]
- **Deliverables**:
  - Modular NLP & Linguistic Features service layer (`backend/app/services/nlp_features/`) with provider abstraction (`BaseNLPProvider`, `RuleLexicalNLPProvider`, `TransformerNLPProvider`, `MockNLPProvider`).
  - Linguistic processing algorithms for 6 feature families:
    1. Lexical & Syntactic Diversity: Character count, word count, sentence count, type-token ratio (TTR), average word length, and 4-category pronoun counts (1st singular, 1st plural, 2nd person, 3rd person) and pronoun ratio (10 features).
    2. Sentiment Polarity Dynamics: VADER compound score, positive/neutral/negative lexicon proportions, contextual positive/neutral/negative probabilities, and polarity score (8 features).
    3. Affective Emotion Probabilities: 7-class normalized probability distribution across Joy, Sadness, Anger, Fear, Surprise, Disgust, and Neutral (7 features).
    4. Lexical TF-IDF Projections: 20 canonical n-gram TF-IDF weights (20 features).
    5. Semantic Embedding Summary: 16 dense semantic axes via deterministic projection / mean pooling (16 features).
    6. Thematic LDA Topic Proportions: 5 latent Dirichlet allocation topic probabilities (5 features).
  - Standardized immutable raw feature vector (`v1.0-66d`) with exactly 66 finite numerical elements in indices 00–65.
  - Relational database entity `NLPFeatureRecord` (`nlp_features` table) and repository operations with idempotent upserts.
  - REST endpoints: `POST /api/v1/chat/messages/{id}/analyze-nlp`, `GET /api/v1/chat/messages/{id}/nlp-features`, and `GET /api/v1/chat/rooms/{room_id}/nlp-features`.
  - WebSocket event dispatch (`nlp_analysis_completed`) broadcasting to dyadic clients in real time.
  - Dual text & speech input: seamlessly analyzes raw text messages and Milestone 5 speech transcripts from voice messages.
  - Frontend ChatRoom integration featuring collapsible 66-d linguistic & emotional descriptor breakdown with non-diagnostic research disclaimers.
  - Comprehensive automated pytest suite (46/46 tests passing) and automated live 2-client verification script (`scratch/m7_live_verification.py`).
- **Constraints**: Strictly descriptive linguistic and affective features; no multimodal fusion (M8), no personal baseline (M9), no medical/psychiatric diagnoses.
- **Status**: `COMPLETED & VALIDATED`

---

## Milestone 8: Multimodal Feature Fusion [COMPLETED]
- **Deliverables**:
  - Modular Multimodal Feature Fusion service layer (`backend/app/services/multimodal_fusion/`) orchestrating M3 communication, M6 acoustic, and M7 NLP modalities.
  - Standardized immutable 140-dimensional feature space (`v1.0-140d`) with exact index mapping:
    1. `[000..009]`: M3 Communication Dynamics (10 dims: turn index, sender ratio, latency, gap, duration, char/word lengths, session averages, turn switch).
    2. `[010..068]`: M6 Acoustic Descriptors (59 dims: pitch F0, RMS energy & shimmer, speech rate, pause dynamics, 13 MFCC means & stds, spectral descriptors).
    3. `[069..134]`: M7 NLP & Linguistic Descriptors (66 dims: lexical diversity & pronouns, VADER and contextual sentiment, 7 affective emotions, 20 TF-IDF, 16 embeddings, 5 LDA topics).
    4. `[135..139]`: Modality Availability & Type Mask (5 dims: `has_comm`, `has_acoustic`, `has_nlp`, `is_voice`, `is_text`).
  - Dual vector outputs: `raw_fused_vector` and `normalized_fused_vector` with deterministic canonical min-max scaling strictly bounded in $[0.0, 1.0]$.
  - Deterministic missing modality imputation: 59 zeros for absent acoustic (text messages); 66 neutral defaults for untranscribed voice.
  - Relational database entity `MultimodalFeatureRecord` (`multimodal_features` table) and repository operations with idempotent upserts.
  - REST endpoints: `POST /api/v1/chat/messages/{id}/fuse`, `GET /api/v1/chat/messages/{id}/multimodal-features`, and `GET /api/v1/chat/rooms/{room_id}/multimodal-features`.
  - WebSocket event dispatch (`multimodal_fusion_completed`) broadcasting to dyadic clients in real time.
  - Frontend ChatRoom integration featuring collapsible 140-d multimodal drawer with modality badges, 4-block breakdown, normalized vector summary, and non-diagnostic research disclaimer.
  - Comprehensive automated pytest suite (60/60 tests passing, 14 M8 tests) and automated live 2-client verification script (`scratch/m8_live_verification.py`).
- **Constraints**: Strictly aligned statistical representation; no personal baseline (M9), no diagnostic/psychiatric labeling (M10).
- **Status**: `COMPLETED & VALIDATED`

---

## Milestone 9: Personal Baseline & Temporal Trend Engine [COMPLETED]
- **Deliverables**:
  - Modular Personal Baseline & Temporal Trend service layer (`backend/app/services/personal_baseline/`) implementing idiographic normative modeling on 140-d fused vectors (`v1.0-140d`).
  - Longitudinal baseline engine computing Bessel sample variance $s_i^2 = \frac{1}{N_i - 1} \sum m_i (x_i - \mu_i)^2$ and std across historical windows (`all_time`, `rolling_10`, `rolling_30`), requiring $N_{\text{prior}} \ge 3$ for baseline establishment.
  - Two-branch regularized $z$-score deviation engine with $\epsilon = 10^{-6}$, clipping in $[-5.0, +5.0]$, and modality-mask-aware validity filters (76 valid dimensions for text; 135 for transcribed voice).
  - Aggregate deviation metrics: `mean_absolute_z` ($\frac{1}{D_{\text{valid}}} \sum |z_i|$), `rms_z` ($\sqrt{\text{mean}(z^2)}$), and `standardized_euclidean_distance` ($\sqrt{\sum z^2}$).
  - Longitudinal temporal dynamics profiler: SMA-3, SMA-5, EMA-3 ($\alpha=0.5$), EMA-5 ($\alpha=1/3$), OLS regression trajectory slope $\beta_{1, i}$, and Rate of Change (RoC).
  - Strict leakage prevention: evaluation of message $M_k$ uses strictly $t_{\text{prev}} < t_k$; $M_k$ is never in its own baseline.
  - Point-in-time historical reproducibility: persisted baseline snapshot statistics (`baseline_snapshot_mean`, `baseline_snapshot_std`) guarantee past evaluations remain immutable upon addition of future turns.
  - Database schema: `PersonalBaselineRecord` (`personal_baselines` table) and `TemporalTrendRecord` (`temporal_trend_records` table) with concurrency-safe idempotent upserts.
  - REST endpoints:
    - `POST /api/v1/chat/baselines/users/{user_id}/recompute`
    - `GET /api/v1/chat/baselines/users/{user_id}`
    - `POST /api/v1/chat/temporal/messages/{message_id}/evaluate`
    - `GET /api/v1/chat/temporal/messages/{message_id}`
    - `GET /api/v1/chat/temporal/users/{user_id}/trajectory`
    - `GET /api/v1/chat/temporal/rooms/{room_id}`
  - WebSocket event dispatch (`personal_baseline_updated`, `temporal_deviation_computed`) broadcasting real-time evaluation updates.
  - Frontend ChatRoom integration:
    - Baseline Health Status badge in header (`Calibrating (N/3)` vs `Active (N)`).
    - Interactive 140-d Z-score deviation drawer on both voice and text messages displaying valid dimension counts, aggregate deviation metrics, top 5 deviations, and research non-diagnostic disclaimers.
  - Pytest automated test suite (35/35 new tests passing, 95/95 total backend tests passing).
  - Live 2-client 7-turn verification script (`scratch/m9_live_verification.py`) executing end-to-end with real-time WebSocket validation.
- **Constraints**: Descriptive within-subject statistical deviations only. Strictly non-diagnostic; no clinical categorization or DSM/ICD triage (M10). M3-M8 data completely immutable.
- **Status**: `COMPLETED & VALIDATED`

---

## Milestone 10: Final Analytical Report Engine [COMPLETED]
- **Deliverables**:
  - Modular Analytical Report Service layer (`backend/app/services/analytical_report/`) orchestrating M9 personal baseline and temporal trend records into an interpretable longitudinal research report.
  - Pure statistical aggregation engine (`AnalyticalReportEngine`) computing:
    1. Multi-modality completeness & coverage (communication, acoustic, NLP counts & percentages).
    2. Aggregate statistical variation summaries: Mean $|Z|$, RMS $Z$, and Standardized Euclidean Distance (mean, median, min, max, std).
    3. Research-bounded statistical variation classification (Stable $<1.0\sigma$, Noticeable $1.0-2.0\sigma$, Significant $\ge 2.0\sigma$).
    4. Largest behavioral feature shifts ranked deterministically by mean $|Z|$ desc, max $|Z|$ desc, index asc.
    5. Longitudinal temporal dynamics: OLS trajectory slopes, trend directions, SMA-3, SMA-5, EMA-3, EMA-5, and Rate of Change.
    6. Longitudinal trajectory time-series structures for frontend charting.
    7. Automated methodological research limitations.
    8. Prominent mandatory non-diagnostic research disclaimer.
  - Multi-format report exporter (`ReportFormatter`) supporting JSON, structured Markdown, and styled HTML.
  - Database schema: `AnalyticalReportRecord` (`analytical_report_records` table) persisting complete point-in-time report snapshots to guarantee historical immutability.
  - REST endpoints:
    - `POST /api/v1/chat/analytical/users/{user_id}/report`
    - `GET /api/v1/chat/analytical/users/{user_id}/report`
    - `GET /api/v1/chat/analytical/reports/{report_id}`
    - `GET /api/v1/chat/analytical/users/{user_id}/reports`
    - `GET /api/v1/chat/analytical/rooms/{room_id}`
    - `GET /api/v1/chat/analytical/reports/{report_id}/export`
  - WebSocket event dispatch (`analytical_report_generated`) broadcasting real-time notification to connected participants.
  - Security & Authorization: Strict user isolation preventing cross-participant report access (403 Forbidden).
  - Frontend ChatRoom integration:
    - Header "Analytical Report" trigger button.
    - Comprehensive interactive Analytical Report Modal with executive summary cards, modality progress bars, aggregate deviation statistics, feature shifts table, temporal trends table, limitations, export toolbar (JSON/MD/HTML), and mandatory disclaimer.
  - Pytest automated test suite: 37/37 new tests passing (T10-01 to T10-37), bringing full backend regression suite to **132/132 tests passing with 0 regressions**.
  - Live 2-client verification script (`scratch/m10_live_verification.py`) executing 7 dyadic turns, report generation, multi-format export, and snapshot immutability verification.
- **Constraints**: Strictly within-subject descriptive statistical analysis. Absolutely non-diagnostic; no psychiatric diagnosis, DSM/ICD triage, suicide/risk scoring, or clinical recommendations. M1–M9 analytical data completely immutable.
- **Status**: `COMPLETED & VALIDATED`

