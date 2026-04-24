class PipelineServiceError(Exception):
    def __init__(self, stage: str, message: str, status_code: int = 502, details: dict | None = None):
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def build_error_detail(error: PipelineServiceError) -> dict:
    return {
        "error": {
            "type": "pipeline_error",
            "stage": error.stage,
            "message": error.message,
            "details": error.details,
        }
    }
