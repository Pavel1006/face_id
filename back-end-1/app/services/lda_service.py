from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from joblib import dump, load
import numpy as np

class LDAService:
    def __init__(self, n_components=None):
        self.lda = None
        self.n_components = n_components

    def fit_transform(self, data, labels):
        # Set n_components dynamically based on data
        n_classes = len(np.unique(labels))
        n_features = data.shape[1]
        
        # Ensure n_components is within allowable bounds
        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)
        
        self.lda = LDA(n_components=self.n_components)
        return self.lda.fit_transform(data, labels)

    def transform(self, data):
        return self.lda.transform(data)

    def save(self, path):
        dump(self.lda, path)

    def load(self, path):
        self.lda = load(path)
