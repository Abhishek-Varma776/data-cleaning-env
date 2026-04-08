from typing import Any
from openenv import HTTPEnvClient
from models import DataCleaningAction, DataCleaningObservation


class DataCleaningClient(HTTPEnvClient[DataCleaningAction, DataCleaningObservation]):
    """
    Client wrapper for DataCleaningEnvironment.
    Handles communication with the HF Space / local server.
    """

    def _step_payload(self, action: DataCleaningAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> Any:
        return payload
