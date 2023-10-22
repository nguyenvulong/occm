
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class SGD:
    def __init__(self, X, y):
        self.clf = self.train(X, y)

    def train(self, X, y):
        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
        clf.fit(X, y)
        return clf

    def predict(self, X):
        return self.clf.predict(X)

    def evaluate(self, X, y):
        return self.clf.score(X, y)
