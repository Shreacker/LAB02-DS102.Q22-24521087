import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import idx2numpy

from models.softmax_regression import SoftmaxRegression
from utils.utilities import *

train_images = idx2numpy.convert_from_file('data/mnist_train.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('data/mnist_train_label.idx1-ubyte')

test_images = idx2numpy.convert_from_file('data/mnist_test.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('data/mnist_test_label.idx1-ubyte')

train_labels = onehot_encoder(train_labels)
test_labels = onehot_encoder(test_labels)

# Flatten images
N = train_images.shape[0]
train_images = train_images.reshape(N, -1)
N = test_images.shape[0]
test_images = test_images.reshape(N, -1)

# Training Model
smReg = SoftmaxRegression(epoch=500, lr=0.1)
losses = smReg.fit(train_images, train_labels)

# Evaluation
predictions = smReg.predict_proba(test_images)
metrics = smReg.evaluate(test_labels, predictions)
for metric in metrics.items():
    print(metric[0], "=", metric[1])

# Visualize Losses
plot_path = 'data/LossFnPlot/SoftmaxRegLoss'
plt.plot(losses, color='maroon')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Function', size=27)
plt.tight_layout()
plt.savefig(plot_path)
plt.show()