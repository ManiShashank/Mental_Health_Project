# TESTING & VERIFICATION STRATEGY

## 1. Testing Philosophy & Compliance

To maintain academic rigor and software stability, testing follows strict validation rules:
1. **Rule 9**: Every algorithm must have dedicated unit test coverage.
2. **Rule 10**: Every major module must have documented sample input/output fixtures.
3. **Rule 12**: Report all test results and failures honestly.
4. **Rule 13**: Code execution alone does not constitute algorithmic validation; mathematical and statistical invariants must be explicitly asserted.

---

## 2. Test Suite Organization

```
backend/tests/
├── conftest.py                  # Pytest fixtures and test clients
├── test_health.py               # Health & system info endpoint tests (Milestone 1)
├── test_nlp_*.py                # NLP & lexical algorithm tests (Milestone 3)
├── test_voice_*.py              # Acoustic extraction & ASR tests (Milestone 3)
├── test_communication_*.py      # Turn dynamics & cadence tests (Milestone 3)
├── test_baseline_*.py           # Personal baseline & z-score tests (Milestone 4)
├── test_fusion_*.py             # Feature fusion tests (Milestone 4)
└── test_profile_*.py            # Pre-screening rule engine tests (Milestone 4)
```

---

## 3. Test Execution Commands

### Running Backend Unit & Integration Tests:
```powershell
pytest backend/tests -v
```

### Running Tests with Coverage Report:
```powershell
pytest backend/tests --cov=app --cov-report=term-missing
```

### Running Frontend Tests & Type Checking:
```powershell
npm --prefix frontend run build
```

---

## 4. Test Cases & Verification Matrix

### Milestone 1: System Health & Foundation
| Test Function | Target File | Verification Criteria | Status |
|---|---|---|---|
| `test_root_endpoint` | `tests/test_health.py` | Root endpoint returns documentation pointers and non-diagnostic disclaimer | PASS |
| `test_health_check_endpoint` | `tests/test_health.py` | `GET /api/v1/health` returns status code `200` with `status: "healthy"` | PASS |
| `test_system_info_endpoint` | `tests/test_health.py` | `GET /api/v1/info` returns non-diagnostic disclaimer and project metadata | PASS |
| `test_cors_headers` | `tests/test_health.py` | Validates CORS origins and response headers | PASS |
| `test_settings_configuration` | `tests/test_health.py` | Verifies environment variables and default configuration fallbacks | PASS |

### Milestone 2: Communication Infrastructure & WebSockets
| Test Function | Target File | Verification Criteria | Status |
|---|---|---|---|
| `test_rest_create_and_list_rooms` | `tests/test_chat_ws.py` | REST API creates and lists conversation rooms in DB | PASS |
| `test_websocket_single_user_connection` | `tests/test_chat_ws.py` | Single user connects, receives `history` and `user_joined` | PASS |
| `test_two_users_communication_and_persistence` | `tests/test_chat_ws.py` | Alice and Bob connect to same room, exchange messages, and persist to DB | PASS |
| `test_empty_message_rejection` | `tests/test_chat_ws.py` | Empty or whitespace-only messages rejected with validation error | PASS |
| `test_invalid_json_payload_handling` | `tests/test_chat_ws.py` | Malformed JSON handled gracefully without dropping WebSocket | PASS |
| `test_disconnect_and_user_left_broadcast` | `tests/test_chat_ws.py` | Disconnection unregisters user and broadcasts `user_left` | PASS |
| `test_two_clients_live` | `tests/manual_two_browser_test.py` | End-to-end two-browser live network simulation (Alice & Bob in TEST001) | PASS |

### Milestone 3: Conversation Data Management & Communication Metrics
| Test Function | Target File | Verification Criteria | Status |
|---|---|---|---|
| `test_unit_metrics_engine_empty_conversation` | `tests/test_communication_metrics.py` | Empty conversations handle zero messages gracefully without division by zero | PASS |
| `test_unit_metrics_engine_deterministic_calculations` | `tests/test_communication_metrics.py` | Exact assertions for counts, lengths (chars/words), duration, response latency, and gaps | PASS |
| `test_api_conversation_lifecycle_and_analysis` | `tests/test_communication_metrics.py` | REST endpoints for room analysis, conversation details, and session ending | PASS |

### Milestone 4: Voice Data Capture & Persistence
| Test Function | Target File | Verification Criteria | Status |
|---|---|---|---|
| `test_voice_upload_validation_and_persistence` | `tests/test_voice_messages.py` | Valid audio upload, storage writing, schema validation, and DB persistence | PASS |
| `test_invalid_audio_upload_rejection` | `tests/test_voice_messages.py` | Rejection of 0-byte audio and invalid file formats with 400 Bad Request | PASS |
| `test_nonexistent_audio_file_404` | `tests/test_voice_messages.py` | Safe path traversal protection and 404 response on missing audio | PASS |
| `test_voice_and_text_m3_metrics_integration` | `tests/test_voice_messages.py` | Verification that 3 text + 1 voice = total 4 (text: 3, voice: 1) in M3 analysis | PASS |
| `run_m4_live_verification` | `scratch/m4_live_verification.py` | Live 2-client simulation (Alice & Bob exchanging text and voice in M4_DEMO_001) | PASS |

### Milestone 5: Speech-to-Text (STT)
| Test Function | Target File | Verification Criteria | Status |
|---|---|---|---|
| `test_stt_service_initialization_and_provider_switching` | `tests/test_speech_to_text.py` | STT service initialization and dynamic provider switching | PASS |
| `test_valid_voice_message_transcription_and_persistence` | `tests/test_speech_to_text.py` | Speech transcription, DB update, API retrieval, and audio playback preservation | PASS |
| `test_text_message_transcription_rejection` | `tests/test_speech_to_text.py` | Rejection of text messages for transcription with 400 Bad Request | PASS |
| `test_nonexistent_message_transcription_404` | `tests/test_speech_to_text.py` | 404 response when transcribing non-existent message ID | PASS |
| `test_missing_audio_file_transcription_error` | `tests/test_speech_to_text.py` | 404 response when underlying audio recording is missing on disk | PASS |
| `test_failed_transcription_handling_and_status_update` | `tests/test_speech_to_text.py` | Safe handling of engine failures and persistence of `transcription_status='failed'` | PASS |
| `test_m3_metrics_invariance_after_transcription` | `tests/test_speech_to_text.py` | Verification that M3 communication metrics remain unaffected by transcription | PASS |
| `run_m5_live_verification` | `scratch/m5_live_verification.py` | Live 2-client simulation (Alice & Bob with live STT and WS broadcast in M5_DEMO_001) | PASS |

### Milestone 6: Acoustic Feature Extraction
| Test Function | Target File | Verification Criteria | Status |
|---|---|---|---|
| `test_extractor_switching_and_service_initialization` | `tests/test_acoustic_features.py` | Service initialization and dynamic extractor switching between Librosa & Mock | PASS |
| `test_normal_speech_acoustic_extraction` | `tests/test_acoustic_features.py` | Extraction of 6 feature families, voiced frames detection, and exact 59-dim vector | PASS |
| `test_pure_silence_handling_and_null_pitch` | `tests/test_acoustic_features.py` | Pure silence audio produces null pitch/jitter/shimmer, 0.0 voicing, no crashes | PASS |
| `test_very_short_audio_handling` | `tests/test_acoustic_features.py` | Audio < 150ms handled without framing/STFT crashes | PASS |
| `test_text_message_rejection_for_acoustic_analysis` | `tests/test_acoustic_features.py` | Rejection of text messages for acoustic analysis with 400 Bad Request | PASS |
| `test_nonexistent_message_404` | `tests/test_acoustic_features.py` | 404 response when analyzing non-existent message ID | PASS |
| `test_missing_audio_file_404` | `tests/test_acoustic_features.py` | 404 response when underlying audio file is missing on disk | PASS |
| `test_persistence_retrieval_and_idempotent_upsert` | `tests/test_acoustic_features.py` | Idempotent updates, GET endpoints, and full vector persistence | PASS |
| `test_m3_m4_m5_invariance_after_acoustic_analysis` | `tests/test_acoustic_features.py` | Verification that M3 metrics, M4 audio stream, and M5 transcripts are unaffected | PASS |
| `run_m6_live_verification` | `scratch/m6_live_verification.py` | Live 2-client simulation (Alice & Bob with live acoustic extraction in M6_LIVE_001) | PASS |
