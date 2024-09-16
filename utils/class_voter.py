from collections.abc import Iterable
import numpy as np


def combineProbas(probas: Iterable[float]):
    """ combine probabilities using the weighted average of the logit space

    Args:
        probas (Iterable[float]): probabilities to combine, ranged [0, 1]

    Raises:
        ValueError: if any probability is not in the range [0, 1]

    Returns:
        float: combined probability
    """
    
    probas = np.array(list(probas))
    
    if len(probas) == 0:
        return 0.5

    if np.any(probas < 0) or np.any(probas > 1):
        raise ValueError("All probabilities must be in the range [0, 1]")

    if np.any(probas == 0) or np.any(probas == 1):
        res = np.mean(probas[(probas == 0) | (probas == 1)])
        if res == 0.5:
            return combineProbas(probas[(probas != 0) & (probas != 1)])
        return res

    distances = np.minimum(probas, 1 - probas)
    confidences = -np.log(distances)

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
