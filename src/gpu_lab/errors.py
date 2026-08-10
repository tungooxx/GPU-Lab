from pydantic import BaseModel


class GPUError(Exception):
    def __init__(self, error_type: str, message: str, retryable: bool = False):
        self.error_type, self.message, self.retryable = error_type, message, retryable
        super().__init__(message)

    def response(self) -> dict:
        return {
            "error": {"type": self.error_type, "message": self.message, "retryable": self.retryable}
        }


class ErrorResponse(BaseModel):
    error: dict
