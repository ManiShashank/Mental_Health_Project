import io
import math
import os
import uuid
import pytest
from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf

from app.core.config import settings
from app.db.session import init_db, async_session_maker
from app.db.repository import ChatRepository
from app.services.nlp_features import (
    nlp_service,
    MockNLPProvider,
    RuleLexicalNLPProvider,
    TransformerNLPProvider,
    NLPFeaturesResult,
    EmptyTextError
)
from app.services.speech_to_text import stt_service, MockSTTProvider
from app.services.acoustic_features import acoustic_service, MockAcousticExtractor


@pytest.fixture(autouse=True)
def setup_test_env():
    import asyncio
    asyncio.run(init_db())
    os.makedirs(settings.AUDIO_UPLOAD_DIR, exist_ok=True)
    stt_service.set_provider(MockSTTProvider(default_transcript="I feel very happy today and we are ready to work on our project."))
    acoustic_service.set_extractor(MockAcousticExtractor())
    nlp_service.set_provider(RuleLexicalNLPProvider())


def create_test_wav_bytes(duration_s: float = 2.0, sample_rate: int = 16000) -> bytes:
    n_samples = int(duration_s * sample_rate)
    samples = (0.3 * np.sin(2 * np.pi * 220.0 * np.linspace(0, duration_s, n_samples, endpoint=False))).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format='WAV', subtype='PCM_16')
    return buf.getvalue()


def test_nlp_provider_switching():
    """T-01: Test provider switching between RuleLexical, Transformer, and Mock."""
    mock_prov = MockNLPProvider()
    nlp_service.set_provider(mock_prov)
    assert nlp_service.provider.provider_name == "mock-nlp-provider-v1"

    trans_prov = TransformerNLPProvider()
    nlp_service.set_provider(trans_prov)
    assert nlp_service.provider.provider_name == "transformer-v1.0.0"

    rule_prov = RuleLexicalNLPProvider()
    nlp_service.set_provider(rule_prov)
    assert nlp_service.provider.provider_name == "rule-lexical-v1.0.0"


def test_text_message_nlp_extraction(client: TestClient):
    """
    T-02: Test NLP extraction on standard text message:
    - All 6 feature families populated
    - Exact 66 finite floats in raw_feature_vector
    - Correct lexical counts, positive valence compound, and valid emotion distribution.
    """
    room_id = f"M7_TEXT_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/PID-ALICE?display_name=Alice") as ws:
        _ = ws.receive_json()
        _ = ws.receive_json()
        ws.send_text('{"event": "message", "data": {"message_text": "I love working on this project! We are making wonderful progress today."}}')
        echo = ws.receive_json()
        msg_id = echo["data"]["message_id"]

    res = client.post(f"/api/v1/chat/messages/{msg_id}/analyze-nlp")
    assert res.status_code == 200
    data = res.json()

    assert data["message_id"] == msg_id
    assert data["source_type"] == "text_message"
    assert data["status"] == "completed"
    assert data["feature_vector_version"] == "v1.0-66d"

    # Lexical verification
    lex = data["lexical_features"]
    assert lex["word_count"] >= 10
    assert lex["first_person_singular_count"] >= 1  # "I"
    assert lex["first_person_plural_count"] >= 1    # "We"
    assert lex["type_token_ratio_ttr"] > 0.5

    # Sentiment verification ("love", "wonderful" -> high positive compound)
    sent = data["sentiment_features"]
    assert sent["vader_compound"] > 0.3
    assert sent["sentiment_pos_prob"] > sent["sentiment_neg_prob"]
    assert sent["sentiment_polarity_score"] > 0.0

    # Emotion verification (joy dominant)
    emo = data["emotion_features"]
    assert emo["emotion_joy"] > emo["emotion_sadness"]
    total_emo = sum(emo.values())
    assert 0.99 <= total_emo <= 1.01  # Sums to 1.0

    # Exact 66-dimensional vector
    raw_vec = data["raw_feature_vector"]
    assert len(raw_vec) == 66
    assert all(math.isfinite(val) for val in raw_vec)


def test_voice_transcript_nlp_extraction(client: TestClient):
    """
    T-03: Test NLP extraction consuming M5 Speech-to-Text transcripts from voice messages.
    """
    room_id = f"M7_VOICE_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    # 1. Upload voice message (M4)
    audio_bytes = create_test_wav_bytes(duration_s=2.0)
    files = {"file": ("voice_m7.wav", io.BytesIO(audio_bytes), "audio/wav")}
    upload_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "PID-BOB", "duration_seconds": 2.0}
    )
    v_msg_id = upload_res.json()["message"]["message_id"]

    # 2. Transcribe voice message (M5)
    t_res = client.post(f"/api/v1/chat/messages/{v_msg_id}/transcribe")
    assert t_res.status_code == 200
    assert t_res.json()["transcription_status"] == "completed"

    # 3. Trigger NLP analysis (M7)
    nlp_res = client.post(f"/api/v1/chat/messages/{v_msg_id}/analyze-nlp")
    assert nlp_res.status_code == 200
    data = nlp_res.json()

    assert data["message_id"] == v_msg_id
    assert data["source_type"] == "voice_transcript"
    assert data["status"] == "completed"
    assert len(data["raw_feature_vector"]) == 66


def test_untranscribed_voice_message_rejection(client: TestClient):
    """
    T-04: Test that an untranscribed voice message is rejected with 400 Bad Request.
    """
    room_id = f"M7_UNTRANSCRIBED_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    audio_bytes = create_test_wav_bytes(duration_s=1.5)
    files = {"file": ("voice_raw.wav", io.BytesIO(audio_bytes), "audio/wav")}
    upload_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "PID-ALICE", "duration_seconds": 1.5}
    )
    v_msg_id = upload_res.json()["message"]["message_id"]

    # Attempt NLP extraction before STT transcription
    res = client.post(f"/api/v1/chat/messages/{v_msg_id}/analyze-nlp")
    assert res.status_code == 400
    assert "has not been transcribed yet" in res.json()["detail"].lower()


def test_empty_and_whitespace_text_rejection(client: TestClient):
    """
    T-05: Empty or whitespace-only messages rejected with 400 Bad Request.
    """
    room_id = f"M7_EMPTY_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    import asyncio
    async def create_whitespace_msg():
        async with async_session_maker() as db:
            await ChatRepository.get_or_create_user(db, "PID-TEST", "Tester")
            return await ChatRepository.save_message(
                session=db,
                room_id=room_id,
                sender_id="PID-TEST",
                message_text="   \n\t  ",
                message_type="text"
            )

    msg = asyncio.run(create_whitespace_msg())
    res = client.post(f"/api/v1/chat/messages/{msg.id}/analyze-nlp")
    assert res.status_code == 400
    assert "empty or whitespace" in res.json()["detail"].lower()


def test_short_single_word_message_handling(client: TestClient):
    """
    T-06: Short single-word messages ("Yes", "Hello") process cleanly without divide-by-zero errors.
    """
    room_id = f"M7_SHORT_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/PID-BOB?display_name=Bob") as ws:
        _ = ws.receive_json()
        _ = ws.receive_json()
        ws.send_text('{"event": "message", "data": {"message_text": "Yes"}}')
        echo = ws.receive_json()
        msg_id = echo["data"]["message_id"]

    res = client.post(f"/api/v1/chat/messages/{msg_id}/analyze-nlp")
    assert res.status_code == 200
    data = res.json()
    assert data["lexical_features"]["word_count"] == 1
    assert data["lexical_features"]["type_token_ratio_ttr"] == 1.0
    assert len(data["raw_feature_vector"]) == 66


def test_pronoun_counting_and_ratios(client: TestClient):
    """
    T-07: Test accurate identification and distribution of pronouns across all 4 categories.
    """
    room_id = f"M7_PRONOUNS_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    text = "I told myself that we should help you and your friend because they need them."
    with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/PID-ALICE?display_name=Alice") as ws:
        _ = ws.receive_json()
        _ = ws.receive_json()
        ws.send_text(f'{{"event": "message", "data": {{"message_text": "{text}"}}}}')
        echo = ws.receive_json()
        msg_id = echo["data"]["message_id"]

    res = client.post(f"/api/v1/chat/messages/{msg_id}/analyze-nlp")
    assert res.status_code == 200
    lex = res.json()["lexical_features"]

    assert lex["first_person_singular_count"] == 2  # "I", "myself"
    assert lex["first_person_plural_count"] == 1    # "we"
    assert lex["second_person_count"] == 2          # "you", "your"
    assert lex["third_person_count"] == 2           # "they", "them"
    assert lex["pronoun_ratio"] > 0.4


def test_sentiment_polarity_and_emotion_bounds(client: TestClient):
    """
    T-08: Negative text validation:
    - High negative valence compound
    - Sadness/fear/anger higher than joy
    - All emotion probabilities in [0.0, 1.0] and sum to 1.0.
    """
    room_id = f"M7_NEG_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    neg_text = "I feel awful, very sad and exhausted today. Nothing is going well and I am stressed."
    with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/PID-BOB?display_name=Bob") as ws:
        _ = ws.receive_json()
        _ = ws.receive_json()
        ws.send_text(f'{{"event": "message", "data": {{"message_text": "{neg_text}"}}}}')
        echo = ws.receive_json()
        msg_id = echo["data"]["message_id"]

    res = client.post(f"/api/v1/chat/messages/{msg_id}/analyze-nlp")
    assert res.status_code == 200
    data = res.json()

    sent = data["sentiment_features"]
    assert sent["vader_compound"] < -0.3
    assert sent["sentiment_neg_prob"] > sent["sentiment_pos_prob"]
    assert sent["sentiment_polarity_score"] < 0.0

    emo = data["emotion_features"]
    assert emo["emotion_sadness"] > emo["emotion_joy"]
    assert emo["emotion_fear"] > 0.05
    assert 0.99 <= sum(emo.values()) <= 1.01


def test_nonexistent_message_nlp_404(client: TestClient):
    """
    T-09: 404 response on non-existent message ID.
    """
    fake_id = str(uuid.uuid4())
    res = client.post(f"/api/v1/chat/messages/{fake_id}/analyze-nlp")
    assert res.status_code == 404


def test_nlp_persistence_retrieval_and_idempotent_upsert(client: TestClient):
    """
    T-10: Test persistence, idempotent repeated extraction, and GET endpoints.
    """
    room_id = f"M7_PERSIST_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/PID-ALICE?display_name=Alice") as ws:
        _ = ws.receive_json()
        _ = ws.receive_json()
        ws.send_text('{"event": "message", "data": {"message_text": "Testing database persistence and idempotency."}}')
        echo = ws.receive_json()
        msg_id = echo["data"]["message_id"]

    # 1. First extraction
    res1 = client.post(f"/api/v1/chat/messages/{msg_id}/analyze-nlp")
    assert res1.status_code == 200

    # 2. Second extraction (idempotent upsert)
    res2 = client.post(f"/api/v1/chat/messages/{msg_id}/analyze-nlp")
    assert res2.status_code == 200
    assert res2.json()["message_id"] == msg_id

    # 3. GET /messages/{msg_id}/nlp-features
    get_res = client.get(f"/api/v1/chat/messages/{msg_id}/nlp-features")
    assert get_res.status_code == 200
    assert get_res.json()["message_id"] == msg_id
    assert len(get_res.json()["raw_feature_vector"]) == 66

    # 4. GET /rooms/{room_id}/nlp-features
    room_res = client.get(f"/api/v1/chat/rooms/{room_id}/nlp-features")
    assert room_res.status_code == 200
    assert room_res.json()["total_analyzed"] == 1


def test_exact_66d_vector_ordering_and_determinism(client: TestClient):
    """
    T-11: Validate exact 66-dimensional vector ordering and determinism.
    """
    room_id = f"M7_VECTOR_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    text = "We are very glad to share our preliminary results."
    with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/PID-ALICE?display_name=Alice") as ws:
        _ = ws.receive_json()
        _ = ws.receive_json()
        ws.send_text(f'{{"event": "message", "data": {{"message_text": "{text}"}}}}')
        echo = ws.receive_json()
        msg_id = echo["data"]["message_id"]

    res1 = client.post(f"/api/v1/chat/messages/{msg_id}/analyze-nlp").json()
    res2 = client.post(f"/api/v1/chat/messages/{msg_id}/analyze-nlp").json()

    v1 = res1["raw_feature_vector"]
    v2 = res2["raw_feature_vector"]

    assert len(v1) == 66
    assert len(v2) == 66
    assert v1 == v2  # Exactly deterministic


def test_m3_m4_m5_m6_invariance_after_nlp_analysis(client: TestClient):
    """
    T-12: Strict invariance verification:
    - M3 communication metrics remain identical
    - M4 voice audio streaming remains bit-for-bit identical
    - M5 STT transcript remains intact
    - M6 acoustic feature record remains intact
    """
    room_id = f"M7_INVARIANCE_{uuid.uuid4().hex[:8]}"
    client.post("/api/v1/chat/rooms", json={"id": room_id})

    # 1. Upload M4 voice message
    audio_bytes = create_test_wav_bytes(duration_s=2.5)
    files = {"file": ("inv_voice.wav", io.BytesIO(audio_bytes), "audio/wav")}
    upload_res = client.post(
        f"/api/v1/chat/rooms/{room_id}/voice",
        files=files,
        data={"user_id": "PID-ALICE", "duration_seconds": 2.5}
    )
    v_msg = upload_res.json()["message"]
    v_msg_id = v_msg["message_id"]

    # 2. M5 Speech-to-Text
    t_res = client.post(f"/api/v1/chat/messages/{v_msg_id}/transcribe")
    expected_transcript = t_res.json()["transcript_text"]

    # 3. M6 Acoustic Analysis
    a_res = client.post(f"/api/v1/chat/messages/{v_msg_id}/analyze-acoustic")
    assert a_res.status_code == 200
    expected_a_vector = a_res.json()["raw_feature_vector"]

    # 4. Send 2 Text Messages
    with client.websocket_connect(f"/api/v1/chat/ws/{room_id}/PID-BOB?display_name=Bob") as ws:
        _ = ws.receive_json()
        _ = ws.receive_json()
        ws.send_text('{"event": "message", "data": {"message_text": "Text Message 1"}}')
        _ = ws.receive_json()
        ws.send_text('{"event": "message", "data": {"message_text": "Text Message 2"}}')
        _ = ws.receive_json()

    # Capture M3 metrics BEFORE NLP
    m3_before = client.get(f"/api/v1/chat/rooms/{room_id}/analysis").json()

    # Perform M7 NLP Analysis on voice transcript and text message
    nlp_res1 = client.post(f"/api/v1/chat/messages/{v_msg_id}/analyze-nlp")
    assert nlp_res1.status_code == 200

    # 1. Verify M3 Metrics Invariance
    m3_after = client.get(f"/api/v1/chat/rooms/{room_id}/analysis").json()
    assert m3_after["message_count"] == m3_before["message_count"] == 3
    assert m3_after["text_message_count"] == m3_before["text_message_count"] == 2
    assert m3_after["voice_message_count"] == m3_before["voice_message_count"] == 1

    # 2. Verify M4 Voice Playback Invariance
    audio_filename = os.path.basename(v_msg["audio_url"])
    stream_res = client.get(f"/api/v1/chat/audio/{audio_filename}")
    assert stream_res.status_code == 200
    assert stream_res.content == audio_bytes

    # 3. Verify M5 Transcript Invariance
    hist_res = client.get(f"/api/v1/chat/rooms/{room_id}/messages")
    retrieved_v_msg = [m for m in hist_res.json()["messages"] if m["message_id"] == v_msg_id][0]
    assert retrieved_v_msg["transcript_text"] == expected_transcript

    # 4. Verify M6 Acoustic Features Invariance
    m6_check = client.get(f"/api/v1/chat/messages/{v_msg_id}/acoustic-features").json()
    assert m6_check["raw_feature_vector"] == expected_a_vector
