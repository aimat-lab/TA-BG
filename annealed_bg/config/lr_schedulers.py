from abc import ABC, abstractmethod
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, NonNegativeInt


class LRSchedulerConfig(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def create_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        pass


class CosineAnnealingLRSchedulerConfig(LRSchedulerConfig):
    scheduler_type: Literal["cosine"]
    T_max: NonNegativeInt
    eta_min: float

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.T_max, eta_min=self.eta_min
        )


class CosineAnnealingWarmRestartsLRSchedulerConfig(LRSchedulerConfig):
    scheduler_type: Literal["cosine_warm_restarts"]
    T_0: NonNegativeInt
    eta_min: float

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=self.T_0,
            eta_min=self.eta_min,
        )


class StepLRSchedulerConfig(LRSchedulerConfig):
    scheduler_type: Literal["step"]
    step_size: NonNegativeInt
    gamma: float

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
        )
