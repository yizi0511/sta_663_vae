# Python Implementation of Variational Auto-Encoder

<img src="https://github.com/yizi0511/sta_663_vae/blob/master/demo/vae.png" width="1000" height="180">
A variational auto-encoder (VAE) is implemented in this repository for STA 663 final project. Our work includes:

- An implementation of VAE using NumPy
- Using Tensorflow and GPU to optimize the code
- Comparison to two other image reconstruction algorithms: Restricted Boltzman Machine (RBM) and Bayesian Convolutional Neural Networks with Bayes by Backprop (BNN)
- We tested our VAE model on the simulated networks, MNIST and other real-world data sets.

# Requirements
Python >= 3.6 <br/>
numpy <br/>
cupy <br/>
tensorflow <br/>
mnist <br/>
scipy <br/>
matplotlib <br/>

# Installation

Our package has several dependencies. We recommend creating a virtual environment and then installing the required packages:

1. Clone the Git repo
2. Create a virtual conda environment: ```cd sta_663_vae; python3 -m venv vae_env```
3. Activate that environment: ```source vae_env/bin/activate```
4. Install the packages into the environment: ```pip install -r requirements.txt```
5. Install our package: ```python setup.py install```

# Data Set:

Location: `./vae/data/` <br/>
1. MNIST: import package mnist or load file `train-images-idx3-ubyte.gz`
2. Frey Face: ```frey_rawface.mat```
3. CIFAR-10: see `./demo/vae_freyface_cifar10_demo.ipynb`
4. Simulated network data: see ```./demo/simulate_random_networks.ipynb```


# Testing Performed:

#### 1. Train VAE: <br/>

##### MNIST (NumPy)
```python3 vae_mnist_numpy.py``` <br/>
##### MNIST (Tensorflow)
```python3 vae_mnist_tensorflow.py``` <br/>

You can also specify a list of arguments as inputs to the auto-encoder: <br/>
```python3 vae_mnist_tensorflow.py --epoch=200 --learning_rate=0.01 --use_cpu=0``` <br/>

##### Frey Face & CIFAR-10 
See `./demo/vae_freyface_cifar10_demo.ipynb` <br/>
##### Simulation study
See ```./demo/simulate_random_networks.ipynb```


#### 2. Run comparative algorithms on MNIST: <br/>

Note the code for RBM and DBN are from the author "2015xli". Please see the resources section for a link to the Git repo.

##### RBM 
See `RBM_demo.ipynb` <br/>

##### DBN
See `DBN_demo.ipynb` <br/>

#### 3. For demonstration and visualization of our model on various data sets, please see the ```demo``` directory. 

#### 4. For a profile summary of our model on MNIST, please see the ```profile_output``` directory. 


# Resources

1. [The original VAE paper by Kingma and Welling](https://arxiv.org/abs/1312.6114)
2. [Implementation of RBM and DBN](https://github.com/2015xli/DBN)

