"""
# Error Types

This module defines error types that are raised by the CL SDK.
"""

class ControlRequestError(RuntimeError):
    """Raised when an attempt to take control is rejected. This implies that another process has control."""

class ControlRequiredError(RuntimeError):
    """Raised when a method is called that requires control, but control has not been taken."""

class TransactionRejected(RuntimeError):
    """Raised when a stimulation plan is rejected by the system."""

class ChannelQueueFull(TransactionRejected):
    """Raised when a stimulation plan is rejected because a channel has too many operations queued."""

class SyncLimitExceeded(TransactionRejected):
    """Raised when a stimulation plan is rejected because too many sync operations are in flight."""

class RunTimestampOrderError(TransactionRejected):
    """Raised when a stimulation plan is rejected because the supplied timestamp to run at is not newer than an already queued plan."""

class DeferredInterruptLimitExceeded(TransactionRejected):
    """Raised when a stimulation plan is rejected because too many deferred channel interruptions are in flight."""

class RecordingFailedError(RuntimeError):
    """Raised when a recording unexpectedly fails."""

class WsApiError(RuntimeError):
    """Raised when an error occurs in the generic WebSocket API client API."""

class UnsafeOperationError(Exception):
    """Raised when opening a recording containing potentially unsafe objects."""
