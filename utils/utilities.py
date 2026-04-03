import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def filter_data(data, condition):
    images, labels = data

    new_images = images[labels == condition]
    new_labels = labels[labels == condition]

    return new_images, new_labels

def onehot_encoder(labels: np.ndarray):
    N = labels.shape[0] # (N, )
    total_classes = labels.max() + 1
    oh_labels = np.zeros((N, total_classes))

    for i, pos in enumerate(labels):
        oh_labels[i][pos] = 1

    return oh_labels

def evaluate(y: np.ndarray, y_hat: np.ndarray, average: str ='binary') -> dict:
    precision = precision_score(y, y_hat, average=average)
    recall = recall_score(y, y_hat, average=average)
    f1 = f1_score(y, y_hat, average=average)

    return {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }