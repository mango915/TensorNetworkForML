import numpy as np
from torchvision.datasets import MNIST

def create_dataset(n_samples, linear_dim=5, sigma=0.5, prob_zero=0.5):
    """
    Create a dataset of greyscale images with 2 different patterns and their labels.
    
    Parameters
    ----------
    n_samples : int, 
        number of samples to be created
    linear_dim : int, 
        linear dimension of squared images
    sigma : float in [0,1], 
        level of noise (0 = no noise, 1 = only noise)
    prob_zero : float in [0,1], 
        probability that an image is created according to pattern 0
        
    Returns
    -------
    data : numpy array of floats in [0,1] of shape (n_samples, linear_dim, linear_dim)
        array of greyscale images
    labels : numpy array of int of length n_samples
        labels of the corresponding images (either 0 or 1)
    
    Notes
    -----
    Example of use case:
    
    import data_generator as gen
    n_samples = 10 
    data, labels = gen.create_dataset(n_samples)
    
    """
    # true images (names=labels)
    one = np.eye(linear_dim)
    zero = one[::-1,:]
    
    # sample labels according to the prob of having a label = 0 (prob_zero)
    labels = np.random.choice([0,1], size=n_samples, p=[prob_zero, 1-prob_zero])
    
    data = np.zeros((n_samples,linear_dim,linear_dim))
    zero_mask = (labels==0)
    data[zero_mask] = zero
    data[~zero_mask] = one
    # add noise
    noise = np.random.rand(n_samples,linear_dim,linear_dim)*sigma
    data = data*(1-sigma)+noise
    
    return data, labels

def get_MNIST_dataset(data_root_dir = './datasets', download=True):
    """
    Import MNIST dataset in numpy splitted in training and test sets.
    
    Parameters
    ----------
    data_root_dir : str,
        path of the directory where to download or import the dataset
    download: bool,
        downloads the dataset from http://yann.lecun.com/exdb/mnist/
        
    Return
    ------
    train_data : numpy array, float, shape (60000, 28, 28)
    train_labels : numpy array, int, shape (60000,)
    test_data : numpy array, float, shape (10000, 28, 28)
    test_labels : numpy array, int, shape (10000,)
    
    Notes
    -----
    Requires torchvision library

    """
    torch_train_dataset = MNIST(data_root_dir, train=True,  download=download)
    torch_test_dataset  = MNIST(data_root_dir, train=False, download=download)
    
    train_data = np.array([np.array(x[0]) for x in torch_train_dataset])
    train_labels = np.array([np.array(x[1]) for x in torch_train_dataset])

    test_data = np.array([np.array(x[0]) for x in torch_test_dataset])
    test_labels = np.array([np.array(x[1]) for x in torch_test_dataset])
    
    return train_data, train_labels, test_data, test_labels
