import os
import numpy as npy

try:
    import cupy as np
except ImportError:
    import numpy as np

from vae.utils.functionals import sigmoid, lrelu, tanh, relu, initialize_weight, initialize_bias


class VAE():
    def __init__(self, input_size, latent_size, hidden_size, 
    			batch_size, learning_rate, beta1, beta2, tolerance):
    """
    Initialize the VAE model.
    """
        
        self.n_in = input_size
        self.n_z = latent_size
        self.n_hidden = hidden_size

        self.n_batch = batch_size
        self.lr = learning_rate

        # Initialize to sample from the Gaussians
        self.z = 0
        self.sample = 0
                
        # Initialize the encoder
        self.e_W0 = initialize_weight(self.n_in, self.n_hidden)
        self.e_b0 = initialize_bias(self.n_hidden)

        self.e_W_mu = initialize_weight(self.n_hidden, self.n_z)
        self.e_b_mu = initialize_bias(self.n_z)
        
        self.e_W_logvar = initialize_weight(self.n_hidden, self.n_z)
        self.e_b_logvar = initialize_bias(self.n_z)

        # Initialize the decoder 
        self.d_W0 = initialize_weight(self.n_z, self.n_hidden)
        self.d_b0 = initialize_bias(self.n_hidden)
        
        self.d_W1 = initialize_weight(self.n_hidden, self.n_in)
        self.d_b1 = initialize_bias(self.n_in)          
        
        # Initialize the optimizer
        self.b1 = beta1
        self.b2 = beta2
        self.tol = tolerance
        self.m = [0] * 10
        self.v = [0] * 10
        self.t = 0
        
    def encode(self, x):
        """
        Construct the encoder model of the VAE.
        """

        self.e_in = np.reshape(x, (self.n_batch, -1))
    
        self.e_h0 = self.e_in.dot(self.e_W0) + self.e_b0
        self.e_h0_a = lrelu(self.e_h0)
    		
        self.e_logvar = self.e_h0_a.dot(self.e_W_logvar) + self.e_b_logvar
        self.e_mu = self.e_h0_a.dot(self.e_W_mu) + self.e_b_mu
    
        return self.e_mu, self.e_logvar
    
    def decode(self, z):
        """
        Construct the decoder model of the VAE.
        """

        self.z = np.reshape(z, (self.n_batch, self.n_z))
        
        self.d_h0 = self.z.dot(self.d_W0) + self.d_b0		
        self.d_h0_a = relu(self.d_h0)

        self.d_h1 = self.d_h0_a.dot(self.d_W1) + self.d_b1
        self.d_h1_a = sigmoid(self.d_h1)

        # FIX IT: DO NOT USE ARBITRARY SIZE !
        self.xhat = np.reshape(self.d_h1_a, (self.n_batch, 28, 28, 1))

        return self.xhat

    def reparametrize(self, mu, logvar):
        """
        Use reparameterization trick to sample latent z from gaussians.
        """

        self.sample = np.random.standard_normal(size=(self.n_batch, self.n_z))
        z = mu + np.exp(.5 * logvar) * self.sample
        return z
    
    def forward(self, x):
        """
        Forward pass of the VAE network.
        """

        mu, logvar = self.encode(x)     
        self.z = self.reparametrize(mu, logvar)       
        decode = self.decode(self.z)
        
        return decode, mu, logvar

    def optimize(self, grads):
        """
        Update weights using the Adam optimizer.
            - grads: gradients from backpropagation
        """ 

        self.t += 1
        for i, grad in enumerate(grads):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.power(grad, 2)
            momentum = self.m[i] / (1 - (self.b1 ** self.t))
            velocity = self.v[i] / (1 - (self.b2 ** self.t))
            grads[i] = momentum / (np.sqrt(velocity) + self.tol)
        
        grad_e_W0, grad_e_b0, grad_W_mu, grad_b_mu, grad_W_logvar, grad_b_logvar, grad_d_W0, grad_d_b0, grad_d_W1, grad_d_b1 = grads

        # Update all weights
        for i in range(self.n_batch):
            # Update encoder weights
            self.e_W0 = self.e_W0 - self.lr * grad_e_W0[i]
            self.e_b0 = self.e_b0 - self.lr * grad_e_b0[i]
    
            self.e_W_mu = self.e_W_mu - self.lr * grad_W_mu[i]
            self.e_b_mu = self.e_b_mu - self.lr * grad_b_mu[i]
            
            self.e_W_logvar = self.e_W_logvar - self.lr * grad_W_logvar[i]
            self.e_b_logvar = self.e_b_logvar - self.lr * grad_b_logvar[i]
    
            # Update decoder weights
            self.d_W0 = self.d_W0 - self.lr * grad_d_W0[i]
            self.d_b0 = self.d_b0 - self.lr * grad_d_b0[i]
            
            self.d_W1 = self.d_W1 - self.lr * grad_d_W1[i]
            self.d_b1 = self.d_b1 - self.lr * grad_d_b1[i]

        return
    
    def backward(self, x, xhat):
        """
        Use backpropagation to calculate the gradients from the reconstructed output.
            - x: input to reconstruct from
            - xhat: reconstructed input
        """
        
        x_in = np.reshape(x, (self.n_batch, -1))
        x_out = np.reshape(xhat, (self.n_batch, -1))
        
        ###### Calculate decoder gradients (left side terms) ######
        dL_l = - x_in * (1 / x_out)
        dSig = sigmoid(self.d_h1, requires_grad=True)        
        drelu = relu(self.d_h0, requires_grad=True)

        db1_d_l = dL_l * dSig
        dW1_d_l = np.matmul(np.expand_dims(self.d_h0_a, axis=-1), np.expand_dims(db1_d_l, axis=1))
         
        db0_d_l = db1_d_l.dot(self.d_W1.T) * drelu
        dW0_d_l = np.matmul(np.expand_dims(self.z, axis=-1), np.expand_dims(db0_d_l, axis=1))
        
        ###### Calculate decoder gradients (right side terms) ######
        dL_r = (1 - x_in) * (1 / (1 - x_out))

        db1_d_r = dL_r * dSig
        dW1_d_r = np.matmul(np.expand_dims(self.d_h0_a, axis=-1), np.expand_dims(db1_d_r, axis=1))
        
        db0_d_r = db1_d_r.dot(self.d_W1.T) * drelu
        dW0_d_r = np.matmul(np.expand_dims(self.z, axis=-1), np.expand_dims(db0_d_r, axis=1))
        
        ###### Combine gradients for decoder ######
        grad_d_W0 = dW0_d_l + dW0_d_r
        grad_d_b0 = db0_d_l + db0_d_r
        grad_d_W1 = dW1_d_l + dW1_d_r
        grad_d_b1 = db1_d_l + db1_d_r



         
        ###### Calculate encoder gradients from reconstruction (left side terms) ######
        d_b_mu_l  = db0_d_l.dot(self.d_W0.T)
        d_W_mu_l = np.matmul(np.expand_dims(self.e_h0_a, axis=-1), np.expand_dims(d_b_mu_l, axis=1))
        
        db0_e_l = d_b_mu_l.dot(self.e_W_mu.T) * lrelu(self.e_h0, requires_grad=True)
        dW0_e_l = np.matmul(np.expand_dims(x_in, axis=-1), np.expand_dims(db0_e_l, axis=1)) 
        
        d_b_logvar_l = d_b_mu_l * np.exp(self.e_logvar * .5) * .5 * self.sample
        d_W_logvar_l = np.matmul(np.expand_dims(self.e_h0_a, axis=-1), np.expand_dims(d_b_logvar_l, axis=1))
        
        db0_e_l_2 = d_b_logvar_l.dot(self.e_W_logvar.T) * lrelu(self.e_h0, requires_grad=True)
        dW0_e_l_2 = np.matmul(np.expand_dims(x_in, axis=-1), np.expand_dims(db0_e_l_2, axis=1)) 
        
        ###### Calculate encoder gradients from reconstruction (right side terms) ######
        d_b_mu_r  = db0_d_r.dot(self.d_W0.T)
        d_W_mu_r = np.matmul(np.expand_dims(self.e_h0_a, axis=-1), np.expand_dims(d_b_mu_r, axis=1))
        
        db0_e_r = d_b_mu_r.dot(self.e_W_mu.T) * lrelu(self.e_h0, requires_grad=True)
        dW0_e_r = np.matmul(np.expand_dims(x_in, axis=-1), np.expand_dims(db0_e_r, axis=1)) 
        
        d_b_logvar_r = d_b_mu_r * np.exp(self.e_logvar * .5) * .5 * self.sample
        d_W_logvar_r = np.matmul(np.expand_dims(self.e_h0_a, axis=-1), np.expand_dims(d_b_logvar_r, axis=1))
        
        db0_e_r_2 = d_b_logvar_r.dot(self.e_W_logvar.T) * lrelu(self.e_h0, requires_grad=True)
        dW0_e_r_2 = np.matmul(np.expand_dims(x_in, axis=-1), np.expand_dims(db0_e_r_2, axis=1))
           

        ###### Calculate encoder gradients from KL divergence (logvar) ###### 
        dKL_b_log = -.5 * (1 - np.exp(self.e_logvar))
        dKL_W_log = np.matmul(np.expand_dims(self.e_h0_a, axis= -1), np.expand_dims(dKL_b_log, axis= 1))
        
        dlrelu = lrelu(self.e_h0, requires_grad=True)  

        dKL_e_b0_1 = .5 * dlrelu * (np.exp(self.e_logvar) - 1).dot(self.e_W_logvar.T)
        dKL_e_W0_1 = np.matmul(np.expand_dims(x_in, axis= -1), np.expand_dims(dKL_e_b0_1, axis= 1))
        
        ###### Calculate encoder gradients from KL divergence (mu) ###### 
        dKL_W_m = .5 * (2 * np.matmul(np.expand_dims(self.e_h0_a, axis=-1), np.expand_dims(self.e_mu, axis=1)))
        dKL_b_m = .5 * (2 * self.e_mu)
        
        dKL_e_b0_2 = .5 * dlrelu * (2 * self.e_mu).dot(self.e_W_mu.T)
        dKL_e_W0_2 = np.matmul(np.expand_dims(x_in, axis= -1), np.expand_dims(dKL_e_b0_2, axis= 1))
        
        ###### Combine gradients for encoder ######
        grad_b_logvar = dKL_b_log + d_b_logvar_l + d_b_logvar_r
        grad_W_logvar = dKL_W_log + d_W_logvar_l + d_W_logvar_r
        grad_b_mu = dKL_b_m + d_b_mu_l + d_b_mu_r
        grad_W_mu = dKL_W_m + d_W_mu_l + d_W_mu_r
        grad_e_b0 = dKL_e_b0_1 + dKL_e_b0_2 + db0_e_l + db0_e_l_2 + db0_e_r + db0_e_r_2
        grad_e_W0 = dKL_e_W0_1 + dKL_e_W0_2 + dW0_e_l + dW0_e_l_2 + dW0_e_r + dW0_e_r_2
        
        
        grads = [grad_e_W0, grad_e_b0, grad_W_mu, grad_b_mu, grad_W_logvar, grad_b_logvar,
                     grad_d_W0, grad_d_b0, grad_d_W1, grad_d_b1]

        # Update weights
        self.optimize(grads)

        return

 
    

