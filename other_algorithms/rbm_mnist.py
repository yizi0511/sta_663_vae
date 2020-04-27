import numpy as np
import mnist
from sklearn.neural_network import BernoulliRBM


X_train = mnist.train_images().reshape(-1, 784) / 255
X_test = mnist.test_images().reshape(-1, 784) / 255

rbm = BernoulliRBM(n_components=500, learning_rate=0.001, random_state=0, n_iter=200, batch_size=100, verbose=True)
rbm.fit(X_train)


