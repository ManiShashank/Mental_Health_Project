class AcousticExtractionError(Exception):
    """Base exception for acoustic feature extraction domain."""
    pass


class AudioFileNotFoundError(AcousticExtractionError):
    """Raised when an audio recording file is not found on the local filesystem."""
    pass


class InvalidAudioFormatError(AcousticExtractionError):
    """Raised when an audio file has an unsupported format or is corrupt (e.g. 0 bytes)."""
    pass


class NonVoiceMessageError(AcousticExtractionError):
    """Raised when an acoustic extraction request is made on a non-voice message."""
    pass


class AudioSignalProcessingError(AcousticExtractionError):
    """Raised when a signal processing error occurs during feature extraction."""
    pass
