import numpy as np


class Voter:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X, methodCalledForPredictProba="predict_proba"):
        predictions = [getattr(model, methodCalledForPredictProba)(X) for model in self.models]
        return np.mean(predictions, axis=0)

    def predict(self, X, methodCalledForPredictProba="predict_proba"):
        proba = self.predict_proba(X, methodCalledForPredictProba)
        return [1 if p > 0.5 else 0 for p in proba]