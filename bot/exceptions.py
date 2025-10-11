"""Custom exceptions for Telegram Bot"""


class BotException(Exception):
    """Base exception for bot errors"""
    pass


class FileTooLargeError(BotException):
    """Raised when uploaded file exceeds size limit"""

    def __init__(self, file_size_mb: float, max_size_mb: int):
        self.file_size_mb = file_size_mb
        self.max_size_mb = max_size_mb
        super().__init__(
            f"File size ({file_size_mb:.1f} MB) exceeds maximum allowed ({max_size_mb} MB)"
        )


class InvalidFileTypeError(BotException):
    """Raised when file is not a PDF"""

    def __init__(self, file_type: str):
        self.file_type = file_type
        super().__init__(
            f"Invalid file type: {file_type}. Only PDF files are supported."
        )


class PDFParsingError(BotException):
    """Raised when PDF parsing fails"""

    def __init__(self, original_error: Exception):
        self.original_error = original_error
        super().__init__(
            f"Failed to parse PDF: {str(original_error)}"
        )


class ExtractionError(BotException):
    """Raised when entity extraction fails"""

    def __init__(self, original_error: Exception):
        self.original_error = original_error
        super().__init__(
            f"Failed to extract entities: {str(original_error)}"
        )


class SVGGenerationError(BotException):
    """Raised when SVG generation fails"""

    def __init__(self, original_error: Exception):
        self.original_error = original_error
        super().__init__(
            f"Failed to generate SVG: {str(original_error)}"
        )


class RateLimitExceededError(BotException):
    """Raised when user exceeds rate limit"""

    def __init__(self, user_id: int, limit: int, time_window: str = "hour"):
        self.user_id = user_id
        self.limit = limit
        self.time_window = time_window
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {time_window}"
        )


class ProcessingTimeoutError(BotException):
    """Raised when processing takes too long"""

    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Processing timeout after {timeout_seconds} seconds"
        )


class ConfigurationError(BotException):
    """Raised when bot configuration is invalid"""

    def __init__(self, message: str):
        super().__init__(f"Configuration error: {message}")
