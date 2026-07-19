class CyExMError(Exception):
    """Base exception for expected Cy-ExM failures."""


class ThreadConfigurationError(CyExMError):
    """Native thread settings cannot be applied safely."""


class InputValidationError(CyExMError):
    """An input path, image, or spatial setting is invalid."""


class CacheValidationError(CyExMError):
    """A cached stage is incomplete, corrupt, or incompatible."""


class RegistrationStageError(CyExMError):
    """A named registration stage failed."""


class ChunkRegistrationError(RegistrationStageError):
    """A serial chunk failed."""

    def __init__(self, chunk_id: int, message: str) -> None:
        self.chunk_id = chunk_id
        super().__init__(f"Chunk {chunk_id}: {message}")
