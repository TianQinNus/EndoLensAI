"""Background task worker boundary."""


class AnalysisWorker:
    """Adapt the application runtime to the worker-facing lifecycle contract."""

    def __init__(self, runtime):
        self.runtime = runtime

    def close(self):
        self.runtime.close()
