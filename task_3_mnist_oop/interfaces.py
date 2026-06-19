from abc import ABC, abstractmethod


class DigitClassificationInterface(ABC):
    @abstractmethod
    def predict(self):
        pass

