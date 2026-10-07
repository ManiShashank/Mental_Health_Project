class SpeechToTextError(Exception):
    """Base exception for Speech-to-Text service failures."""
    pass


class AudioFileNotFoundError(SpeechToTextError):
    """Raised when the specified audio file cannot be found on disk."""
    pass


class InvalidAudioFormatError(SpeechToTextError):
    """Raised when the audio format is corrupt or unsupported."""
    pass


class NonVoiceMessageError(SpeechToTextError):
    """Raised when attempting to transcribe a non-voice message."""
    pass


class TranscriptionFailedError(SpeechToTextError):
    """Raised when speech recognition inference fails."""
    pass
