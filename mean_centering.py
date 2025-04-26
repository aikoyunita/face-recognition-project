from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class MeanCentering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X, y=None):
        return X - self.mean_
