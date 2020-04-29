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
2. Create a virtual conda environment: <br/>
```cd sta_663_vae``` <br/>
```python3 -m venv vae_env```
3. Activate that environment: ```source vae_env/bin/activate```
4. Install the packages into the environment: ```pip install -r requirements.txt```
5. Install our package: ```python setup.py install```

# Data Set:

1. MNIST: located in `./vae/data/`
2. HCP:
   - ```HCP_subcortical_CMData_desikan.mat``` includes network data from 1065 human subjects
   - ```HCP_Covariates.mat``` includes human cognition traits 
3. Simulated data: see ```./demo/simulate_random_networks.ipynb```


# Instructions

#### 1. Train VAE on the following data sets: <br/>

##### MNIST (NumPy)
```python3 vae_mnist_numpy.py``` <br/>
##### MNIST (Tensorflow)
```python3 vae_mnist_tensorflow.py``` <br/>
##### HCP 
```python3 vae_hcp_pytorch.py``` <br/>

You can specify a list of arguments as inputs to the auto-encoder: <br/>
```python3 vae_mnist_tensorflow.py --epoch=200 --learning_rate=0.01 --use_cpu=0``` <br/>

#### 2. Run comparative algorithms on MNIST: <br/>

##### RBM 
``` python3 ./other_algorithms/rbm_mnist.py``` <br/>
##### BNN 
``` python3 ./other_algorithms/Bayesian-Neural-Networks/train_BayesByBackprop_MNIST.py``` <br/>

#### 3. For demonstration and visualization of our model on various data sets, please see the ```demo``` directory. 


# Resources

1. [The original VAE paper by Kingma and Welling](https://arxiv.org/abs/1312.6114)
2. [Implementation of Bayesian-Neural-Networks](https://github.com/JavierAntoran/Bayesian-Neural-Networks)
2. [A demo of using RBM](https://www.kaggle.com/nicw102168/restricted-boltzmann-machine-rbm-on-mnist)

