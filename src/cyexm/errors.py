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


class ManifestDecodeError(CyExMError):
    """A YAML document cannot be decoded safely."""


class ConfigurationValidationError(CyExMError):
    """A scientific manifest or execution profile violates its schema."""


class WorkflowPlanError(CyExMError):
    """Resolved workflow inputs or outputs cannot form a safe plan."""


class ExecutorStartupError(CyExMError):
    """The requested execution backend did not become usable."""


class WorkerPreflightError(ExecutorStartupError):
    """One or more workers failed environment preflight."""


class RoundExecutionError(CyExMError):
    """A round task failed after execution began."""


class WorkflowExecutionError(CyExMError):
    """One or more submitted round tasks failed."""

    def __init__(self, failed_rounds: tuple[int, ...], run_record: str) -> None:
        self.failed_rounds = failed_rounds
        self.run_record = run_record
        rounds = ", ".join(str(value) for value in failed_rounds)
        super().__init__(f"Workflow rounds failed: {rounds}; run record: {run_record}")
