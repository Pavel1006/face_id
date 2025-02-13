from sklearn.decomposition import PCA
from joblib import dump, load


class PCAService:
    def __init__(self, n_components=100):
        self.pca = PCA(n_components=n_components)

    def fit_transform(self, data):
        return self.pca.fit_transform(data)

    def transform(self, data):
        if self.pca is None:
            raise ValueError("PCA model is not trained. Load or train the model first.")
        return self.pca.transform(data)

    def save(self, path):
        dump(self.pca, path)

    def load(self, path):
        self.pca = load(path)
