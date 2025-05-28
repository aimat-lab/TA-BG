from pydantic import BaseModel, ConfigDict, PositiveInt


class CheckpointingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    write_checkpoint_every: PositiveInt
