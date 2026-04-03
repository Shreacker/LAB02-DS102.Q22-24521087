import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import idx2numpy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils.utilities import *

'''
LOGISTIC REGRESSION
'''
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

# Training Model
print('---LOGISTIC REGRESSION MODEL (Binary Classification)---')
logReg = LogisticRegression(random_state=21, max_iter=50)
logReg.fit(train_images, train_labels)

# Evaluation
preds = logReg.predict(test_images)
metrics = evaluate(test_labels, preds)
for metric in metrics.items():
    print(metric[0], "=", metric[1])

'''
SOFTMAX REGRESSION
'''
train_images = idx2numpy.convert_from_file('data/mnist_train.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('data/mnist_train_label.idx1-ubyte')

test_images = idx2numpy.convert_from_file('data/mnist_test.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('data/mnist_test_label.idx1-ubyte')

# Flatten images
N = train_images.shape[0]
train_images = train_images.reshape(N, -1)
N = test_images.shape[0]
test_images = test_images.reshape(N, -1)

# Standardization
scaler = StandardScaler()
train_images = scaler.fit_transform(train_images)
test_images = scaler.transform(test_images)

# Training Model
print('---SOFTMAX REGRESSION MODEL (Multiclass Classification)---')
smReg = LogisticRegression(random_state=21, max_iter=500, solver='saga')
smReg.fit(train_images, train_labels)

# Evaluation
preds = logReg.predict(test_images)
print(np.unique(preds))
metrics = evaluate(test_labels, preds, average='macro')
for metric in metrics.items():
    print(metric[0], "=", metric[1])