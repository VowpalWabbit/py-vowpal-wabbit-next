from typing import List, Optional
import vowpalwabbit_next._core


class _LearnerBase:
    _workspace: vowpalwabbit_next._core.Workspace

    def __init__(self, command_line: List[str], *, model_data: Optional[bytes] = None):
        self._workspace = vowpalwabbit_next._core.Workspace(
            command_line, model_data=model_data
        )
