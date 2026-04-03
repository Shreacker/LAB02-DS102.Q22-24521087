import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class SoftmaxRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr

        self.losses = []
        self.w = None

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        return -(y * np.log(y_hat + 1e-15)).sum(axis=1).mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape
        C = y.shape[1]
        self.w = np.zeros((d, C), dtype=np.float64)
        e = 0
        with tqdm(range(self.epoch)) as pbar:
            for _ in pbar:
                y_hat = self.predict_proba(X)
                delta_y = (y_hat - y)
                
                gradient = X.T @ delta_y / N
                self.w -= gradient * self.lr

                l = self.loss_fn(y, y_hat)
                self.losses.append(l)

                pbar.set_postfix({
                    "loss": l
                })

                e += 1

        return self.losses

    def softmax(self, X: np.ndarray):
        X = X - np.max(X, axis=-1, keepdims=True)
        exp = np.exp(X)
        return exp / exp.sum(axis=-1, keepdims=True)

    def predict_proba(self, X: np.ndarray):
        y_hat = X @ self.w
        return self.softmax(y_hat)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def evaluate(self, y, y_hat) -> dict:
        y_true = np.argmax(y, axis=1)
        y_pred = np.argmax(y_hat, axis=1)

        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        return {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }