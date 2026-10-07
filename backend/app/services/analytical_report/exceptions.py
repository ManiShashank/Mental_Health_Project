"""
Milestone 10: Analytical Report Engine Exceptions Hierarchy
"""


class AnalyticalReportException(Exception):
    """Base exception for analytical report engine errors."""
    pass


class InsufficientReportDataError(AnalyticalReportException):
    """Raised when there are no valid observations in the requested reporting period."""
    pass


class UnsupportedFeatureVersionError(AnalyticalReportException):
    """Raised when an unsupported feature vector version is encountered."""
    pass


class InvalidReportPeriodError(AnalyticalReportException):
    """Raised when period_start > period_end or invalid time range is supplied."""
    pass


class ReportNotFoundError(AnalyticalReportException):
    """Raised when a requested analytical report ID is not found."""
    pass


class UserAccessDeniedError(AnalyticalReportException):
    """Raised when a user attempts unauthorized access to another participant's analytical report."""
    pass


class ReportGenerationError(AnalyticalReportException):
    """Raised when an internal error prevents report assembly or serialization."""
    pass
