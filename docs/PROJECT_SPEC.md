# PROJECT SPECIFICATION

**Project Title:** Multimodal Analysis of Communication and Language Patterns for Personalized Mental Health Risk Screening  
**Domain:** Computational Linguistics, Acoustic Signal Processing, Human-Computer Interaction, Affective Computing  
**System Classification:** Analytical & Pre-Screening Research Platform (Non-Diagnostic)

---

## 1. Executive Summary & Research Scope

This project implements an academic research platform designed to capture, process, and analyze multi-turn dyadic communication across text and voice modalities. The objective is to identify statistical deviations in an individual's communicative behavior relative to their **own personalized historical baseline**, rather than generic population-level thresholds.

### 1.1 Non-Diagnostic Research Boundary
> **CRITICAL ETHICAL & CLINICAL DIRECTIVE**:  
> This platform is an **analytical research and pre-screening decision-support tool**. It does **NOT** provide a medical diagnosis, clinical prognosis, or clinical assessment of any mental health disorder (e.g., Major Depressive Disorder, Generalized Anxiety Disorder, Bipolar Disorder). All system outputs describe solely **observed statistical patterns, linguistic shifts, acoustic metrics, and baseline deviation scores**.

---

## 2. Core Functional Capabilities

1. **Two-User Dyadic Web Communication**: Secure, authenticated, session-managed real-time messaging interface between participant pairs.
2. **Real-Time Text Transmission**: Low-latency bi-directional messaging with precise millisecond-level timestamping.
3. **Voice Recording & Audio Transmission**: High-fidelity in-browser audio capture (Opus/WAV), transmission, playback, and waveform visualization.
4. **Traceable Conversation History**: Complete session logging retaining message sequence, timing intervals, and sender metadata.
5. **Text & Language Analysis Pipeline**: Preprocessing, TF-IDF lexical modeling, contextual BERT embeddings, VADER & Transformer sentiment, transformer emotion classification, LDA topic modeling, and syntactic/lexical statistics.
6. **Dual-Path Voice Analysis Pipeline**:
   - **Path A (Speech-to-Text)**: Transcript generation using Whisper ASR and subsequent processing through the NLP pipeline.
   - **Path B (Acoustic Feature Extraction)**: Extraction of prosodic, spectral, and cepstral features (pitch/F0, energy/RMS, speech rate, pause durations, MFCCs, spectral flux/centroid).
7. **Communication & Interaction Dynamics**: Turn-taking latency, message cadence, session duration, response delay distribution, sender/receiver ratio, and initiation patterns.
8. **Personal Baseline Analysis**: Longitudinal rolling statistics ($z$-scores, personal mean $\mu$, personal standard deviation $\sigma$) establishing individual-specific normative ranges.
9. **Temporal Trend Analysis**: Moving averages, rolling windows, and deviation trajectory tracking across longitudinal sessions.
10. **Multimodal Early Feature Fusion**: Standardized concatenation of lexical, acoustic, and behavioral feature vectors with explicit scaling and missing-value imputation.
11. **Explainable Pre-Screening Profile**: Transparent, rule-based dimensional indicator matrix (e.g., Stable / Noticeable Change / Significant Change) with full feature provenance.
12. **Interactive Research Dashboard**: Visual audit trail containing time-series plots, feature distributions, baseline comparisons, and exportable analytical reports.

---

## 3. Modality Specifications

```
                       ┌───────────────────────────────┐
                       │   Dyadic Interaction Input    │
                       └───────────────┬───────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                │                                             │
      ┌─────────▼─────────┐                         ┌─────────▼─────────┐
      │   Text Messages   │                         │  Voice Messages   │
      └─────────┬─────────┘                         └─────────┬─────────┘
                │                                             │
                │                                    ┌────────┴────────┐
                │                                    ▼                 ▼
                │                             ┌─────────────┐   ┌─────────────┐
                │                             │   Path A    │   │   Path B    │
                │                             │ Whisper ASR │   │  Acoustic   │
                │                             │ (Transcript)│   │  Features   │
                │                             └──────┬──────┘   └──────┬──────┘
                │                                    │                 │
                └──────────────────┬─────────────────┘                 │
                                   ▼                                   │
                     ┌───────────────────────────┐                     │
                     │    NLP & Text Analysis    │                     │
                     │  - Lexical (TF-IDF, POS)  │                     │
                     │  - Embeddings (BERT)      │                     │
                     │  - Sentiment & Emotion    │                     │
                     │  - LDA Topics & Stats     │                     │
                     └─────────────┬─────────────┘                     │
                                   │                                   │
                                   └─────────────────┬─────────────────┘
                                                     ▼
                                     ┌───────────────────────────────┐
                                     │  Communication & Interaction  │
                                     │  (Timing, Turn-taking, Ratio) │
                                     └───────────────┬───────────────┘
                                                     ▼
                                     ┌───────────────────────────────┐
                                     │   Multimodal Feature Fusion   │
                                     │  & Personal Baseline Engine   │
                                     └───────────────┬───────────────┘
                                                     ▼
                                     ┌───────────────────────────────┐
                                     │   Explainable Pre-Screening   │
                                     │   Profile & Visual Dashboard  │
                                     └───────────────────────────────┘
```

---

## 4. Personal Baseline & Statistical Formulation

For any extracted numerical feature $x_{i,t}$ for participant $i$ at session $t$:

$$\mu_i = \frac{1}{N} \sum_{k=1}^N x_{i,k}$$

$$\sigma_i = \sqrt{\frac{1}{N-1} \sum_{k=1}^N (x_{i,k} - \mu_i)^2}$$

The standardized deviation score $z_{i,t}$ is given by:

$$z_{i,t} = \frac{x_{i,t} - \mu_i}{\sigma_i + \epsilon}$$

Where $\epsilon = 10^{-6}$ prevents division by zero in zero-variance features.

### Deviation Classification Thresholds
- **Stable / Typical**: $|z| < 1.0$
- **Noticeable Change**: $1.0 \le |z| < 2.0$
- **Significant Change**: $|z| \ge 2.0$

---

## 5. Communication Protocol & Database Schema (Milestone 2)

### 5.1 Real-Time WebSocket Protocol
- **Transport**: Native WebSocket protocol (`WS /api/v1/chat/ws/{room_id}/{user_id}?display_name={display_name}`)
- **Event Multiplexing**: JSON envelopes containing `event`, `data`, and `timestamp` (UTC ISO-8601).
- **Core Events**:
  - `join` / `user_joined`: Client registration and active room membership broadcast.
  - `history`: Server dispatch of prior timestamped messages upon connection.
  - `message`: Bidirectional real-time text payload broadcast and DB persistence.
  - `leave` / `user_left`: Graceful or network-induced disconnection notice.
  - `error`: Server-side validation error reporting (e.g. empty messages).
  - `ping` / `pong`: Connection heartbeat maintenance.

### 5.2 Database Entities & Relationships
- **ORM**: Async SQLAlchemy 2.0 (SQLite / PostgreSQL compatible)
- **Entities**:
  - `User`: `id` (PK, String), `display_name`, `created_at` (UTC).
  - `Room`: `id` (PK, String), `name`, `status`, `is_active`, `created_at`, `started_at`, `ended_at` (UTC).
  - `Message`: `id` (PK, UUID String), `room_id` (FK $\to$ `rooms.id`), `sender_id` (FK $\to$ `users.id`), `receiver_id` (Nullable String), `message_text` (Text), `message_type` (`text` | `voice` | `system`), `audio_url` (Nullable String), `audio_duration_seconds` (Nullable Float), `audio_format` (Nullable String), `audio_size_bytes` (Nullable Integer), `transcript_text` (Nullable Text), `transcription_status` (Nullable String: `pending` | `processing` | `completed` | `failed`), `transcription_model` (Nullable String), `transcription_language` (Nullable String), `transcription_error` (Nullable Text), `transcribed_at` (Nullable DateTime UTC), `timestamp` (DateTime UTC, Indexed).
  - `AcousticFeatureRecord`: `id` (PK), `message_id` (FK $\to$ `messages.id`, Unique), `room_id` (FK), `sender_id` (FK), `status` (String), `engine` (String), `duration_seconds` (Float), `sample_rate` (Int), `channels` (Int), `pitch_features` (JSON), `energy_features` (JSON), `speech_rate_features` (JSON), `pause_features` (JSON), `mfcc_features` (JSON), `spectral_features` (JSON), `feature_vector_version` (String: `v1.0-59d`), `raw_feature_vector` (JSON: exact 59 floats), `has_voiced_frames` (Boolean), `extraction_error` (Nullable Text), `created_at` (DateTime UTC), `extracted_at` (DateTime UTC).
  - `NLPFeatureRecord`: `id` (PK), `message_id` (FK $\to$ `messages.id`, Unique), `room_id` (FK), `sender_id` (FK), `source_type` (`text_message` | `voice_transcript`), `status` (String), `engine` (String), `text_content` (Text), `lexical_features` (JSON), `sentiment_features` (JSON), `emotion_features` (JSON), `tfidf_features` (JSON), `embedding_features` (JSON), `topic_features` (JSON), `feature_vector_version` (String: `v1.0-66d`), `raw_feature_vector` (JSON: exact 66 floats), `extraction_error` (Nullable Text), `created_at` (DateTime UTC), `extracted_at` (DateTime UTC).
  - `RoomParticipant`: `id` (PK), `room_id` (FK), `user_id` (FK), `joined_at`, `last_seen_at`.

---

## 6. Technology Stack & Architectural Principles

- **Frontend**: React 18, TypeScript, Vite, TailwindCSS, Chart.js / Plotly.js, Lucide Icons.
- **Backend**: Python 3.13 / 3.11+, FastAPI, Uvicorn, Pydantic v2 Settings.
- **Real-Time Communication**: Native WebSocket protocol with JSON payload dispatching.
- **Database**: SQLite (local) / PostgreSQL with Async SQLAlchemy 2.0 and auto-migrations.
- **Speech Processing (M5 & M6)**: OpenAI Whisper STT, Librosa DSP, SoundFile, SciPy signal processing.
- **NLP & Future Milestones**: HuggingFace Transformers, PyTorch, NLTK/spaCy, scikit-learn.
