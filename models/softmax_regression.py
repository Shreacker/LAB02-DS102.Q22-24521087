import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class SoftmaxRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr

        self.losses = []
        self.w = None

    def loss_fn(self, y: np.ndarray, log_y_hat: np.ndarray) -> float:
        return -(y * log_y_hat).sum(axis=1).mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape
        C = y.shape[1]
        self.w = np.zeros((d, C), dtype=np.float64)
        
        with tqdm(range(self.epoch)) as pbar:
            for _ in pbar:
                log_y_hat = self.predict_log_proba(X)
                y_hat = np.exp(log_y_hat)
                delta_y = (y_hat - y)
                
                gradient = X.T @ delta_y / N
                self.w -= gradient * self.lr

                l = self.loss_fn(y, log_y_hat)
                self.losses.append(l)

                pbar.set_postfix({
                    "loss": l
                })

        return self.losses

    def logsumexp(self, X: np.ndarray):
        a = X.max(-1)
        return a + np.log(np.exp(X - a[:, None]).sum(-1))
    
    def log_softmax(self, X: np.ndarray):
        return X - self.logsumexp(X)[:, None]

    def predict_log_proba(self, X: np.ndarray):
        return self.log_softmax(X @ self.w)

    def predict_proba(self, X: np.ndarray):
        return np.exp(self.predict_log_proba(X))
    
    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)
    
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