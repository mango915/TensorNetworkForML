import numpy as np

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