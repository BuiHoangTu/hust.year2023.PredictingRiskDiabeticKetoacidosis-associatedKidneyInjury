from collections.abc import Iterable
import numpy as np


def combineProbas(probas: Iterable[float]):
    probas = list(probas)
    confidences = np.abs(np.array(probas) - 0.5)
    # Normalize the confidences to sum to 1
    if np.sum(confidences) == 0:
        weights = np.ones_like(confidences) / len(confidences)
    else:
        weights = confidences / np.sum(confidences)
    finalProbas = np.sum(weights * probas)
    
    return finalProbas

class ModelWrapper:
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder

    def predict_proba(self, X):
        X = self.encoder.transform(X)
        return self.model.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return [1 if p > 0.5 else 0 for p in proba]

class Voter:
    def __init__(self, models: list[ModelWrapper]):
        self.models = models

    def predict_proba(self, X, methodCalledForPredictProba="predict_proba"):
        predictions = [getattr(model, methodCalledForPredictProba)(X) for model in self.models]
        return np.mean(predictions, axis=0)

    def predict(self, X, methodCalledForPredictProba="predict_proba"):
        proba = self.predict_proba(X, methodCalledForPredictProba)
        return [1 if p > 0.5 else 0 for p in proba]