"""
Multimodal Feature Fusion Domain Exceptions
"""

class MultimodalFusionException(Exception):
    """Base exception for all multimodal feature fusion operations."""
    pass


class MessageNotFoundError(MultimodalFusionException):
    """Raised when the specified message does not exist."""
    pass


class MissingModalityError(MultimodalFusionException):
    """Raised when a required modality is missing and cannot be imputed."""
    pass


class FusionProcessingError(MultimodalFusionException):
    """Raised when an internal error occurs during multimodal vector fusion."""
    pass
