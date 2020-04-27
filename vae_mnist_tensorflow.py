import numpy as np

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import mnist

def xavier_init(channel_in, channel_out, constant = 1):
    """
    Xavier initialization of network weights
    """

    low = -constant * np.sqrt(6.0 / (channel_in + channel_out))
    high = constant * np.sqrt(6.0 / (channel_in + channel_out))
    return tf.random_uniform((channel_in, channel_out), minval = low, maxval = high, dtype=tf.float32)



class VAE(object):

    def __init__(self, network_architecture, learning_rate=0.001, batch_size=100):
        """
        Set up the VAE model.
        """

        # Set model parameters
        self.network_architecture = network_architecture
        self.lr = learning_rate
        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

        # Forward pass
        self.forward()

        # Backward pass
        self.backward()

        # Initialize the variables and launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def initialize_weights(self, n_hidden_enc_1, n_hidden_enc_2,
                           n_hidden_dec_1, n_hidden_dec_2, n_input, n_z):
        """
        Initialize weights of the network layers.
        """

        network_weights = dict()
        network_weights['encoder_weights'] = {
            'W1': tf.Variable(xavier_init(n_input, n_hidden_enc_1)),
            'W2': tf.Variable(xavier_init(n_hidden_enc_1, n_hidden_enc_2)),
            'W_mu': tf.Variable(xavier_init(n_hidden_enc_2, n_z)),
            'W_logvar': tf.Variable(xavier_init(n_hidden_enc_2, n_z))}
        network_weights['encoder_bias'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_enc_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_enc_2], dtype=tf.float32)),
            'b_mu': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'b_logvar': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        network_weights['decoder_weights'] = {
            'W1': tf.Variable(xavier_init(n_z, n_hidden_dec_1)),
            'W2': tf.Variable(xavier_init(n_hidden_dec_1, n_hidden_dec_2)),
            'W_out': tf.Variable(xavier_init(n_hidden_dec_2, n_input))}
        network_weights['decoder_bias'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_dec_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_dec_2], dtype=tf.float32)),
            'b_out': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return network_weights


    def encode(self, weights, bias):
        """
        Use the encoder model to map the input data to the latent space.
        """

        hidden_1 = tf.nn.relu(tf.add(tf.matmul(self.x, weights['W1']), bias['b1']))
        hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, weights['W2']), bias['b2']))
        mu = tf.add(tf.matmul(hidden_2, weights['W_mu']), bias['b_mu'])
        logvar = tf.add(tf.matmul(hidden_2, weights['W_logvar']), bias['b_logvar'])
        return (mu, logvar)

    def decode(self, weights, bias):
        """
        Use the decoder model to reconstruct the input data.
        """

        hidden_1 = tf.nn.leaky_relu(tf.add(tf.matmul(self.z, weights['W1']), bias['b1']))
        hidden_2 = tf.nn.leaky_relu(tf.add(tf.matmul(hidden_1, weights['W2']), bias['b2']))
        recon_x = tf.nn.sigmoid(tf.add(tf.matmul(hidden_2, weights['W_out']), bias['b_out']))
        return recon_x


    def forward(self):
        """
        Build the VAE network.
        """

        # Initialize weights and bias
        network_weights = self.initialize_weights(**self.network_architecture)

        # Use encoder model to obtain latent z
        self.mu, self.logvar = self.encode(network_weights["encoder_weights"],
                                      network_weights["encoder_bias"])

        # Draw sample z from Gaussian using reparametrization trick
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.mu, tf.multiply(tf.sqrt(tf.exp(self.logvar)), eps))

        # Use decoder model to obtain the reconstructed input
        self.recon_x = self.decode(network_weights["decoder_weights"],
                                    network_weights["decoder_bias"])


    def backward(self):
        """
        Calculate gradients using backpropagation and update weights using Adam optimizer.
        """

        rec_loss = - tf.reduce_sum(self.x * tf.log(1e-8 + self.recon_x)
                           + (1 - self.x) * tf.log(1e-8 + 1 - self.recon_x), 1)

        kl = -0.5 * tf.reduce_sum(1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar), 1)

        self.loss = tf.reduce_mean(rec_loss + kl)

        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.loss)

    def train(self, X):
        """
        Train model based on mini-batch of input data.
        Return loss of mini-batch.
        """

        opt, loss = self.sess.run((self.optimizer, self.loss),
                                                feed_dict={self.x: X})
        return loss

    def transform(self, X):
        """
        Transform data by mapping it into the latent space.
        """
        # Note: This maps to mean of distribution, we could alternatively sample from Gaussian distribution
        return self.sess.run((self.mu, self.logvar), feed_dict={self.x: X})

    def generate(self, mu = None):
        """
        Generate data by sampling from the latent space.
        """
        if mu is None:
            # Data is alternatively generated from the prior in the latent space
            mu = np.random.normal(size = self.network_architecture["n_z"])

        return self.sess.run(self.recon_x, feed_dict={self.z: mu})

    def reconstruct(self, X):
        """
        Reconstruct the given input data.
        """

        return self.sess.run(self.recon_x, feed_dict={self.x: X})


def evaluate(model, X, recon_X):
    """
    Evaluate the accuracy of the model with reconstruction error and marginal log-likelihood.
    """
    rec_loss = - np.sum(X * np.log(1e-8 + recon_X)
                           + (1 - X) * np.log(1e-8 + 1 - recon_X), 1)
    mu, logvar = model.transform(X)
    kl = -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar), 1)
    loss = np.mean(rec_loss + kl)
    return loss



def train_mnist(network_architecture, train_images, test_images,
                learning_rate=0.001, batch_size=100, n_epoch=10):
    """
    Train the VAE model on the MNIST data set.
    """

    vae = VAE(network_architecture, learning_rate=learning_rate, batch_size=batch_size)

    train_size = len(train_images)
    train_data = train_images.reshape((train_size, 784)) / 255 # normalize to [0,1]
    test_size = len(test_images)
    test_data = test_images.reshape((test_size, 784)) / 255 # normalize to [0,1]

    elbo_lst = []
    test_elbo_lst = []

    for epoch in range(n_epoch):

        avg_loss = 0.
        test_elbo = 0.
        n_batch = int(train_size / batch_size)
        n_test_batch = int(test_size / batch_size)

        for idx in range(n_batch):
            train_batch = train_data[idx * batch_size:idx * batch_size + batch_size]
            loss = vae.train(train_batch)
            avg_loss += loss / train_size * batch_size
    
        for idx in range(n_test_batch):
            test_batch = test_data[idx * batch_size:idx * batch_size + batch_size]
            recon_test = vae.reconstruct(test_batch)
            elbo = evaluate(vae, test_batch, recon_test)
            test_elbo += elbo / test_size * batch_size

        print("Epoch:", "%d/%d" % (epoch+1, n_epoch),
              "Loss =", "{:.4f}".format(avg_loss),
              "Test ELBO =", "{:.4f}".format(test_elbo))
        
	elbo_lst.append(avg_loss)
        test_elbo_lst.append(test_elbo)

    return vae, elbo_lst, test_elbo_lst



network_architecture = dict(n_hidden_enc_1 = 500,
                             n_hidden_enc_2 = 500,
                             n_hidden_dec_1 = 500,
                             n_hidden_dec_2 = 500,
                             n_input = 784,
                             n_z = 20)

vae, train_elbo, test_elbo = train_mnist(network_architecture, mnist.train_images(), 
        mnist.test_images(), learning_rate=0.001, n_epoch = 200)

np.save('train_elbo.npy', np.array(train_elbo))
np.save('test_elbo.npy', np.array(test_elbo))


