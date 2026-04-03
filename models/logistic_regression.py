import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class LogisticRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr

        self.losses = []
        self.w = None

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        l = y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15)
        return -l.mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape
        self.w = np.zeros((d, ), dtype=np.float64)
        e = 0
        with tqdm(range(self.epoch)) as pbar:
            for _ in pbar:
                y_hat = self.predict(X)
                delta_y = (y_hat - y)
                
                gradient = (delta_y.T / N) @ X
                self.w -= gradient.T * self.lr

                l = self.loss_fn(y, y_hat)
                self.losses.append(l)

                pbar.set_postfix({
                    "loss": l
                })

                e += 1

        return self.losses

    def sigmoid(self, X: np.ndarray):
        return np.where(
            X >= 0,
            1. / (1. + np.exp(-X)),
            np.exp(X) / (1 + np.exp(X))
        )

    def predict(self, X: np.ndarray):
        y_hat = X @ self.w
        return self.sigmoid(y_hat)
    
    def evaluate(self, y, y_hat) -> dict:
        precision = precision_score(y, y_hat)
        recall = recall_score(y, y_hat)
        f1 = f1_score(y, y_hat)

        return {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }