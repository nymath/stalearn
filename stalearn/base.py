from abc import ABCMeta, abstractmethod


class ClassifierMixin:

    _estimator_type = "classifier"
    
    def score(self, X, y, sample_weight=None):
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    
    def _more_tags(self):
        return {"requires_y": True}
    
class RegressorMixin:
    
    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):

        from .metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def _more_tags(self):
        return {"requires_y": True}

