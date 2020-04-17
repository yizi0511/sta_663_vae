import os
import numpy as npy

try:
    import cupy as np
except ImportError:
    import numpy as np
    print("GPU not enabled on this machine.")

from vae.models.vae_model import VAE
from vae.utils.functionals import load_mnist, BCE    

np.random.seed(663)


def train(model, n_epoch=20):
    """
    Train variational auto-encoder on MNIST data set.
    """

    # Load training data
    train_data, _, train_size = load_mnist(data_path = './vae/data/train-images-idx3-ubyte',
                                        label_path = './vae/data/train-labels-idx1-ubyte')
    # shuffle training data
    np.random.shuffle(train_data)
    
    # Batch training setup
    batch_size = model.n_batch
    batch_idx = train_size // batch_size
    
    # Loss setup
    total_loss = 0
    total_rec_loss = 0
    total_kl = 0
    total_iter = 0
    
    for epoch in range(n_epoch):

        for idx in range(batch_idx):

            # Divide training data into mini-batches
            train_batch = train_data[idx * batch_size: idx * batch_size + batch_size]

            # Ignore a batch if insufficient observations 
            if train_batch.shape[0] != batch_size:
                break
            
            ###### Forward Pass ######
            
            xhat, mu, logvar = model.forward(train_batch)
            
            # Calculate reconstruction Loss
            rec_loss = BCE(xhat, train_batch)
            
            # Calculate KL Divergence
            kl = -.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar))
            
            #Loss record keeping
            total_rec_loss += rec_loss / batch_size
            total_kl += kl / batch_size
            total_loss = total_rec_loss + total_kl
            total_iter += 1


            ###### Backpropagation ######

            model.backward(train_batch, xhat) 

            #model.img = np.squeeze(xhat, axis=3) * 2 - 1

        print("Epoch [%d/%d] RC Loss:%.4f  KL Loss:%.4f  Total Loss: %.4f"%(
                epoch, n_epoch, total_rec_loss/total_iter, total_kl/total_iter, total_loss/total_iter))
            


            
if __name__ == '__main__':

    # Set up model params
    input_size = 784
    latent_size = 20
    hidden_size = 400 
    batch_size = 64 
    learning_rate = 0.0001
    beta1 = 0.9
    beta2 = 0.999
    tolerance = 1e-8

    # Instantiate model
    model = VAE(input_size, latent_size, hidden_size, 
                batch_size, learning_rate, beta1, beta2, tolerance)

    # Train model
    train(model, n_epoch=20)
    
