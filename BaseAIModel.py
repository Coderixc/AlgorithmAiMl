from abc import ABC , abstractmethod

class BaseAIModel(ABC):

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, X_train, y_train):
        pass

    @abstractmethod
    def predictModel(self, X_train, y_train):
        pass