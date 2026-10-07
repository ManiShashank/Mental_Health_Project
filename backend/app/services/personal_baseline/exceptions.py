"""
Milestone 9: Domain Exceptions for Personal Baseline and Temporal Trend Engine
"""


class PersonalBaselineException(Exception):
    """Base exception for all Personal Baseline and Temporal Trend operations."""
    pass


class InsufficientHistoryError(PersonalBaselineException):
    """Raised when an operation requires minimum prior historical observations that are not met."""
    pass


class UnsupportedFeatureVectorError(PersonalBaselineException):
    """Raised when an incompatible or unversioned feature vector is supplied."""
    pass


class DataLeakageError(PersonalBaselineException):
    """Raised when an observation is detected leaking into its own reference baseline."""
    pass


class UserAccessDeniedError(PersonalBaselineException):
    """Raised when unauthorized access to a user's personal baseline is detected."""
    pass


class BaselineCalculationError(PersonalBaselineException):
    """Raised when mathematical or statistical calculation fails."""
    pass
