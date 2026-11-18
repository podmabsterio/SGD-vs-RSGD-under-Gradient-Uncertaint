from abc import ABC, abstractmethod


class BaseMetric(ABC):
    name = ""

    def __init__(self, save_per_iteration=False):
        self.save_per_iteration = save_per_iteration

    @abstractmethod
    def __call__(self, **kwargs):
        pass
