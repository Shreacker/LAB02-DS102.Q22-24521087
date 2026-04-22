import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import idx2numpy

from models.logistic_regression import LogisticRegression
from utils.utilities import *

train_images = idx2numpy.convert_from_file('data/mnist_train.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('data/mnist_train_label.idx1-ubyte')
train_data = (train_images, train_labels)

test_images = idx2numpy.convert_from_file('data/mnist_test.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('data/mnist_test_label.idx1-ubyte')
test_data = (test_images, test_labels)

train_images_0, train_labels_0 = filter_data(train_data, 0)
train_images_1, train_labels_1 = filter_data(train_data, 1)
train_images = np.concatenate(
    [train_images_0, train_images_1],
    axis=0
)
train_labels = np.concatenate(
    [train_labels_0, train_labels_1],
    axis=0
)

test_images_0, test_labels_0 = filter_data(test_data, 0)
test_images_1, test_labels_1 = filter_data(test_data, 1)
test_images = np.concatenate(
    [test_images_0, test_images_1],
    axis=0
)
test_labels = np.concatenate(
    [test_labels_0, test_labels_1],
    axis=0
)

# Flatten images
N = train_images.shape[0]
train_images = train_images.reshape(N, -1)
N = test_images.shape[0]
test_images = test_images.reshape(N, -1)

# Normalization
train_mean = train_images.mean()
train_std = train_images.std()
train_images = (train_images - train_mean) / train_std
test_images = (test_images - train_mean) / train_std

# Shuffle dataset
train_idx = np.random.permutation(train_images.shape[0])
test_idx = np.random.permutation(test_images.shape[0])
train_images = train_images[train_idx]
train_labels = train_labels[train_idx]
test_images = test_images[test_idx]
test_labels = test_labels[test_idx]

# Training Model
logReg = LogisticRegression(epoch=50, lr=0.1, solver='saga')
losses = logReg.fit(train_images, train_labels)

# Evaluation
predictions = logReg.predict(test_images)
metrics = logReg.evaluate(test_labels, predictions)
for metric in metrics.items():
    print(metric[0], "=", metric[1])

# Visualize Losses
plot_path = 'data/LossFnPlot/LogisticRegLoss'
plt.plot(losses, color='maroon')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Function', size=27)
plt.tight_layout()
# plt.savefig(plot_path)
plt.show()