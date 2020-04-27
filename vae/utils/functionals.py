import numpy as npy

try:
    import cupy as np
except ImportError:
    import numpy as np


###### Loss Functions ######

def BCE(x, y):
    """
    Calculate binary cross entropy loss for reconstruction loss in the ELBO.
    """

    loss = np.sum(-y * np.log(x + 10e-8) - (1 - y) * np.log(1 - x + 10e-8))

    return loss

###### Activation Functions ######

def sigmoid(x, requires_grad=False):
    """
    Transform the input with a sigmoid function.
    If requires_grad = True, then compute the derivative of sigmoid function.
    """

    res = 1 / (1 + np.exp(-x))
    if requires_grad:
        return res * (1 - res)
    return res

def relu(x, requires_grad=False):
    """
    Transform the input with a relu function.
    If requires_grad = True, then compute the derivative of relu function.
    """

    if requires_grad:
        return 1.0 * (x > 0)
    else:
        return x * (x > 0)   
    
def lrelu(x, alpha=0.01, requires_grad=False):
    """
    Transform the input with a lrelu function.
    If requires_grad = True, then compute the derivative of lrelu function.
    """

    if requires_grad:
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx
    else:
        return np.maximum(x, x * alpha, x)

def tanh(x, requires_grad=False):
    """
    Transform the input with a tanh function.
    If requires_grad = True, then compute the derivative of tanh function.
    """

    if requires_grad:
        return 1.0 - np.tanh(x) ** 2
    return np.tanh(x)




###### Weights initialization ######

def initialize_weight(dim_in, dim_out):
    """
    Initialize the weight matrix of a neural network layer.
        - dim_in: input size of the layer
        - dim_out: output size of the layer
    """

    W = np.random.randn(dim_in, dim_out).astype(np.float32) * np.sqrt(2.0/(dim_in))
    return W


def initialize_bias(dim_out):
    """
    Initialize the bias of a neural network layer.
        - dim_in: input size of the layer
        - dim_out: output size of the layer
    """

    b = np.zeros(dim_out).astype(np.float32)
    return b


###### Data Loaders ######

def load_mnist(data_path, label_path):
    """
    Load the MNIST data set from given file paths.
    """

    f = open(data_path)
    images = npy.fromfile(file=f, dtype=np.uint8)
    X = images[16:].reshape((60000, 28, 28, 1)).astype(np.float32) / 255  # Normalize to [0,1]  


    f = open(label_path)
    labels = npy.fromfile(file=f, dtype=np.uint8)
    Y = labels[8:].reshape((60000)).astype(np.int32)

    return np.array(X), Y, len(X) 


