# API DOCUMENTATION

This document defines the REST and WebSocket API contracts for the Multimodal Mental Health Pre-Screening System.

Base URL: `http://127.0.0.1:8000/api/v1`  
Interactive OpenAPI UI: `http://127.0.0.1:8000/docs`  
ReDoc UI: `http://127.0.0.1:8000/redoc`

---

## 1. System & Health Endpoints (Milestone 1)

### `GET /health`
Verifies backend operational status and returns runtime environment metadata.

- **Request**:
  ```http
  GET /api/v1/health HTTP/1.1
  Host: 127.0.0.1:8000
  ```

- **Response (200 OK)**:
  ```json
  {
    "status": "healthy",
    "timestamp": "2026-08-30T12:54:00.000000Z",
    "version": "1.0.0",
    "environment": "development"
  }
  ```

- **Schema**:
  | Field | Type | Description |
  |---|---|---|
  | `status` | string | Health indicator (`"healthy"`, `"degraded"`, `"unhealthy"`) |
  | `timestamp` | string (ISO-8601) | Server current UTC timestamp |
  | `version` | string | API version string |
  | `environment` | string | Environment name (`"development"`, `"testing"`, `"production"`) |

---

### `GET /info`
Returns academic project metadata and non-diagnostic regulatory disclaimers.

- **Request**:
  ```http
  GET /api/v1/info HTTP/1.1
  Host: 127.0.0.1:8000
  ```

- **Response (200 OK)**:
  ```json
  {
    "project_title": "Multimodal Analysis of Communication and Language Patterns for Personalized Mental Health Risk Screening",
    "system_type": "Analytical Pre-Screening Research Platform",
    "disclaimer": "This system is an analytical pre-screening research platform. It does NOT provide clinical diagnoses or replace professional medical advice.",
    "version": "1.0.0",
    "modalities_supported": [
      "Text (Lexical, Contextual, Affective, Linguistic)",
      "Voice (ASR Transcript, Prosodic, Acoustic, MFCC)",
      "Communication (Turn Dynamics, Cadence, Interaction Ratios)"
    ],
    "baseline_paradigm": "Personal Longitudinal Z-Score Deviation"
  }
  ```

---

## 2. Communication & WebSocket Endpoints (Milestone 2)

### `GET /chat/rooms`
Returns a list of all active chat rooms.

- **Response (200 OK)**:
  ```json
  [
    {
      "id": "TEST001",
      "name": "Room TEST001",
      "is_active": true,
      "created_at": "2026-08-30T13:16:48.000000Z"
    }
  ]
  ```

---

### `POST /chat/rooms`
Creates a new conversation room or retrieves an existing one.

- **Request Body**:
  ```json
  {
    "id": "TEST001",
    "name": "Dyadic Dyad Beta"
  }
  ```
- **Response (201 Created)**: Returns the `RoomResponse` object.

---

### `GET /chat/rooms/{room_id}/messages`
Retrieves historical timestamped messages persisted in the database.

- **Query Parameters**:
  - `limit` (int, default: 100)
  - `offset` (int, default: 0)
- **Response (200 OK)**:
  ```json
  {
    "room_id": "TEST001",
    "messages": [
      {
        "message_id": "2d2db172-06cb-4dda-9777-6f16054c1827",
        "conversation_id": "TEST001",
        "room_id": "TEST001",
        "sender_id": "PID-ALICE",
        "sender_name": "Alice",
        "receiver_id": "PID-BOB",
        "message_text": "Hello Bob",
        "timestamp": "2026-08-30T13:16:48.988460Z",
        "message_type": "text"
      }
    ],
    "total_messages": 1
  }
  ```

---

### `GET /chat/rooms/{room_id}/participants`
Returns the active participants currently connected via WebSockets in the room.

---

### `GET /chat/rooms/{room_id}/analysis`
Computes descriptive communication metrics (message count, length, ratio, response latency, gaps) directly from persisted database messages.

- **Query Parameters**:
  - `gap_threshold` (float, default: `60.0` seconds): Inactivity latency threshold defining a communication gap.
- **Response (200 OK)**:
  ```json
  {
    "conversation_id": "LIVE_DEMO_ROOM",
    "room_id": "LIVE_DEMO_ROOM",
    "status": "active",
    "started_at": "2026-08-30T17:57:28.167030",
    "ended_at": null,
    "participants": ["PID-ALICE-DEMO", "PID-BOB-DEMO"],
    "participant_names": {
      "PID-ALICE-DEMO": "Alice",
      "PID-BOB-DEMO": "Bob"
    },
    "message_count": 5,
    "text_message_count": 5,
    "voice_message_count": 0,
    "messages_by_participant": {
      "PID-ALICE-DEMO": 3,
      "PID-BOB-DEMO": 2
    },
    "message_ratio": {
      "PID-ALICE-DEMO": 0.6,
      "PID-BOB-DEMO": 0.4
    },
    "average_message_length": 62.6,
    "average_message_length_words": 10.8,
    "min_message_length": 50,
    "max_message_length": 69,
    "conversation_duration_seconds": 5.37,
    "average_response_time_seconds": 1.344,
    "min_response_time_seconds": 1.338,
    "max_response_time_seconds": 1.352,
    "number_of_communication_gaps": 0,
    "gap_threshold_seconds": 60.0,
    "participant_breakdown": [
      {
        "user_id": "PID-ALICE-DEMO",
        "display_name": "Alice",
        "message_count": 3,
        "message_percentage": 0.6,
        "average_message_length_chars": 62.0,
        "average_message_length_words": 10.67,
        "total_words": 32
      }
    ]
  }
  ```

---

### `GET /chat/conversations/{room_id}`
Retrieves session metadata, status, participant counts, and lifecycle timestamps.

---

### `POST /chat/rooms/{room_id}/voice` (Milestone 4)
Uploads, stores, persists, and broadcasts a voice message.

- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `file`: Raw audio binary (`.webm`, `.wav`, `.ogg`, `.mp4`, `.aac`, `.mp3`)
  - `user_id`: Sender participant ID
  - `duration_seconds`: (Optional float) Voice recording length in seconds
  - `receiver_id`: (Optional string) Recipient participant ID
- **Response (201 Created)**:
  ```json
  {
    "message": {
      "message_id": "893d56f1-2856-4c48-8df0-7d315f69f2e1",
      "conversation_id": "M4_DEMO_001",
      "room_id": "M4_DEMO_001",
      "sender_id": "PID-ALICE",
      "sender_name": "Alice",
      "receiver_id": "PID-BOB",
      "message_text": "[Voice Message]",
      "timestamp": "2026-08-31T01:00:00Z",
      "message_type": "voice",
      "audio_url": "/api/v1/chat/audio/M4_DEMO_001_893d56f1.webm",
      "audio_duration_seconds": 4.2,
      "audio_format": "audio/webm",
      "audio_size_bytes": 48200
    },
    "detail": "Voice message uploaded, persisted, and broadcast successfully."
  }
  ```

---

### `GET /chat/audio/{filename}` (Milestone 4)
Securely streams stored audio recordings with HTTP Range request support and content-type detection.

- **Response (200 OK)**: Binary audio stream (`audio/webm`, `audio/wav`, `audio/ogg`, etc.)

---

### `POST /chat/messages/{message_id}/transcribe` (Milestone 5)
Executes Speech-to-Text inference on a stored voice message and updates its record in the database.

- **URL Parameter**: `message_id` (string UUID)
- **Query Parameter**: `language` (optional string, e.g. `"en"`)
- **Response (200 OK)**:
  ```json
  {
    "message_id": "893d56f1-2856-4c48-8df0-7d315f69f2e1",
    "room_id": "M5_DEMO_001",
    "transcript_text": "Hello Bob, I am recording a voice message for speech-to-text analysis.",
    "transcription_status": "completed",
    "transcription_model": "whisper-base",
    "transcription_language": "en",
    "transcription_error": null,
    "transcribed_at": "2026-08-31T18:59:00Z",
    "detail": "Voice message transcribed successfully."
  }
  ```

---

### `POST /chat/messages/{message_id}/analyze-acoustic`
Extracts 6 physical acoustic feature families and standardizes them into the exact 59-dimensional feature vector (`v1.0-59d`).

- **Parameters**: `message_id` (string path param)
- **Response (200 OK)**:
  ```json
  {
    "message_id": "27e2fdf0-c9c6-438a-9d64-381df9a2634f",
    "room_id": "M6_LIVE_001",
    "sender_id": "PID-ALICE",
    "status": "completed",
    "engine": "librosa-v1.0.0",
    "duration_seconds": 3.0,
    "sample_rate": 16000,
    "channels": 1,
    "pitch_features": {
      "f0_mean_hz": 200.5,
      "f0_std_hz": 12.3,
      "f0_min_hz": 180.2,
      "f0_max_hz": 220.1,
      "f0_median_hz": 200.1,
      "f0_range_hz": 39.9,
      "voicing_fraction": 0.904,
      "f0_jitter_local": 0.0152
    },
    "energy_features": {
      "rms_mean": 0.3509,
      "rms_std": 0.1205,
      "rms_min": 0.0,
      "rms_max": 0.5432,
      "rms_energy_range": 0.5432,
      "shimmer_local": 0.0312
    },
    "speech_rate_features": {
      "estimated_syllable_count": 15,
      "speech_duration_seconds": 2.71,
      "estimated_speaking_rate_syllables_per_sec": 5.0,
      "estimated_articulation_rate_syllables_per_sec": 5.54
    },
    "pause_features": {
      "pause_count": 1,
      "total_pause_duration_seconds": 0.29,
      "mean_pause_duration_seconds": 0.29,
      "max_pause_duration_seconds": 0.29,
      "pause_to_speech_ratio": 0.107
    },
    "mfcc_features": {
      "mfcc_means": [-120.4, 45.2, "... (13 values)"],
      "mfcc_stds": [12.1, 8.4, "... (13 values)"],
      "mfcc_dimension": 26
    },
    "spectral_features": {
      "spectral_centroid_mean": 289.0,
      "spectral_centroid_std": 45.2,
      "spectral_bandwidth_mean": 412.0,
      "spectral_bandwidth_std": 32.1,
      "spectral_rolloff_mean": 510.0,
      "spectral_rolloff_std": 40.2,
      "spectral_flatness_mean": 0.0012,
      "spectral_flatness_std": 0.0003,
      "zcr_mean": 0.045,
      "zcr_std": 0.012
    },
    "feature_vector_version": "v1.0-59d",
    "raw_feature_vector": [200.5, "... (exact 59 finite float elements)"],
    "has_voiced_frames": true,
    "extraction_error": null,
    "extracted_at": "2026-08-31T19:40:00Z",
    "detail": "Acoustic features extracted and persisted successfully."
  }
  ```

---

### `GET /chat/messages/{message_id}/acoustic-features`
Retrieves persisted acoustic features for a specific voice message.

---

### `GET /chat/rooms/{room_id}/acoustic-features`
Retrieves all completed acoustic feature records for all voice messages in a conversation room.

---

### `POST /chat/conversations/{room_id}/end`
Marks an active conversation as completed and records the UTC `ended_at` timestamp.

---

### `WS /chat/ws/{room_id}/{user_id}`
Real-time bidirectional WebSocket transport for dyadic messaging.

- **Query Parameters**: `display_name` (string, optional)
- **Inbound Text Event Schema**:
  ```json
  {
    "event": "message",
    "data": {
      "message_text": "Hello Bob",
      "receiver_id": "PID-BOB"
    }
  }
  ```
- **Outbound Event Types**:
  - `history`: Prior room messages (both text and voice) dispatched upon connection.
  - `user_joined`: Broadcasted to room participants when a user connects.
  - `message`: Real-time broadcasted message payload (includes `audio_url` and audio metadata for voice messages).
  - `transcription_completed`: Real-time STT completion notification with transcript text.
  - `acoustic_analysis_completed`: Real-time acoustic feature extraction completion notification.
  - `user_left`: Broadcasted when a participant disconnects.
  - `error`: Server-side error payload (e.g. empty message validation failure).
  - `ping` / `pong`: Connection heartbeat maintenance.

---

## 3. Upcoming Endpoints (Roadmap)

### Milestone 2: Communication Endpoints
- `POST /api/v1/sessions/create`: Initialize a dyadic interaction session.
- `GET /api/v1/conversations/{id}/messages`: Retrieve timestamped conversation logs.
- `POST /api/v1/voice/upload`: Upload in-browser recorded audio chunk/message.
- `WS /api/v1/ws/chat/{session_id}/{user_id}`: Real-time bidirectional WebSocket transport.

### Milestone 3: Analytical Endpoints
- `POST /api/v1/analysis/session/{session_id}/process`: Trigger multimodal analysis pipeline on session data.
- `GET /api/v1/analysis/session/{session_id}/features`: Retrieve extracted multimodal feature vectors.

### Milestone 4 & 5: Baseline & Dashboard Endpoints
- `GET /api/v1/baseline/user/{user_id}`: Retrieve personal baseline mean $\mu$ and standard deviation $\sigma$.
- `GET /api/v1/profile/user/{user_id}/session/{session_id}`: Retrieve explainable pre-screening deviation profile.
- `GET /api/v1/dashboard/overview/{user_id}`: Retrieve aggregated time-series trends and visual feeds.
