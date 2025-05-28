from abc import ABC

from pydantic import BaseModel, ConfigDict


class TrainingModeConfig(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid")
