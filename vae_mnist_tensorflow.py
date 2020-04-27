import argparse
import numpy as np

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))

import mnist

from vae.models.vae_model_tf import VAE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--input_size", type=int, default=784)
    parser.add_argument("--latent_size", type=int, default=20)
    parser.add_argument("--hidden_size", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--use_cpu", type=int, default=1)
    return parser.parse_args()

args = parse_args()

def use_device(use_cpu = 1):
    """
    Choose available CPU or GPU devices to use.
    """
    if (use_cpu == 1):
        device = '/device:CPU:0'
    else:
        device = '/device:GPU:0'
    return device


def calculate_elbo(model, X, recon_X):
    """
    Compute the ELBO of the model with reconstruction error and KL divergence..
    """
    rec_loss = - np.sum(X * np.log(1e-8 + recon_X)
                           + (1 - X) * np.log(1e-8 + 1 - recon_X), 1)
    mu, logvar = model.transform(X)
    kl = -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar), 1)
    elbo = np.mean(rec_loss + kl)
    return elbo


def train_mnist(network_architecture, train_images, test_images,
                learning_rate=0.001, batch_size=100, n_epoch=10, use_cpu=True):
    """
    Train the VAE model on the MNIST data set.
    """

    # Instantiate the model
    vae = VAE(network_architecture, learning_rate=learning_rate, batch_size=batch_size, use_device=use_device(use_cpu))

    # Load mnist data
    train_size = len(train_images)
    train_data = train_images.reshape((train_size, 784)) / 255 # normalize to [0,1]
    test_size = len(test_images)
    test_data = test_images.reshape((test_size, 784)) / 255 # normalize to [0,1]

    # Record keeping
    train_elbo_lst = []
    test_elbo_lst = []

    for epoch in range(n_epoch):

        train_loss = 0.
        test_loss = 0.
        n_train_batch = train_size // batch_size
        n_test_batch = test_size // batch_size

        # Train in mini-batches
        for idx in range(n_train_batch):
            train_batch = train_data[idx * batch_size:idx * batch_size + batch_size]
            loss = vae.train(train_batch)
            train_loss += loss
    
        # Test in mini-batches
        for idx in range(n_test_batch):
            test_batch = test_data[idx * batch_size:idx * batch_size + batch_size]
            recon_batch = vae.reconstruct(test_batch)
            loss = calculate_elbo(vae, test_batch, recon_batch)
            test_loss += loss
        
        train_elbo_lst.append(train_loss / n_train_batch)
        test_elbo_lst.append(test_loss / n_test_batch)

        print("Epoch:", "%d/%d" % (epoch+1, n_epoch),
              "Train Loss =", "{:.4f}".format(train_loss / n_train_batch),
              "Test Loss =", "{:.4f}".format(test_loss / n_test_batch))

    return vae, train_elbo_lst, test_elbo_lst



if __name__ == '__main__':

    print(args.use_cpu)
    print(use_device(args.use_cpu))

    network_architecture = dict(n_hidden_enc_1 = args.hidden_size,
                                 n_hidden_enc_2 = args.hidden_size,
                                 n_hidden_dec_1 = args.hidden_size,
                                 n_hidden_dec_2 = args.hidden_size,
                                 n_input = args.input_size,
                                 n_z = args.latent_size)

    vae, train_elbo, test_elbo = train_mnist(network_architecture, mnist.train_images(), 
            mnist.test_images(), learning_rate=args.learning_rate, n_epoch = args.epoch, use_cpu = args.use_cpu)

    #np.save('train_elbo.npy', np.array(train_elbo))
    #np.save('test_elbo.npy', np.array(test_elbo))


