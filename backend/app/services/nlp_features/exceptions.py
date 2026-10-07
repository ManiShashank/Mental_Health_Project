class NLPException(Exception):
    """Base domain exception for Milestone 7 NLP & Linguistic Feature Extraction."""
    pass


class EmptyTextError(NLPException):
    """Raised when the text payload or transcript is empty or whitespace-only."""
    pass


class MissingTranscriptError(NLPException):
    """Raised when attempting to analyze a voice message that has not yet been transcribed (M5)."""
    pass


class InvalidMessageTypeError(NLPException):
    """Raised when message type cannot undergo NLP analysis."""
    pass


class NLPProcessingError(NLPException):
    """Raised when an internal error occurs during NLP/linguistic extraction."""
    pass
