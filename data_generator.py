import numpy as np
from torchvision.datasets import MNIST
import skimage, skimage.io, skimage.transform
import skimage.measure
from os import walk
    
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

def get_computer_characters(factor):
    # some fonts used are completely misleading, thus I checked them and we are going to filter them out manually

    index = \
    [i for i in range(161,165)] +\
    [i for i in range(209,213)] +\
    [i for i in range(293,297)] +\
    [i for i in range(385,389)] +\
    [i for i in range(401,403)] +\
    [i for i in range(501,505)] +\
    [i for i in range(529,533)] +\
    [i for i in range(557,561)] +\
    [i for i in range(613,617)] +\
    [i for i in range(629,633)] +\
    [i for i in range(689,693)] +\
    [i for i in range(701,705)] +\
    [i for i in range(753,761)] +\
    [i for i in range(777,793)] +\
    [i for i in range(805,809)] +\
    [i for i in range(873,881)] +\
    [i for i in range(941,945)] +\
    [i for i in range(993,1001)]

    index = np.array(index) - 1 #offset in the indexing

    mask = np.ones(1016)
    for i in index:
        mask[i] = 0
    mask = mask.astype(bool)


    common_path = 'datasets/0-9_and_A-Z/Sample'
    images = []
    labels = []

    for n in range(1,3):
        f = []
        x = "%03.0f"%n
        mypath = common_path+x
        for (dirpath, dirnames, filenames) in walk(mypath):
            f.extend(filenames)
            break
        print("Number of files in "+mypath+" : ", len(f), '\n')

        for i in range(len(f)):
            f[i] = mypath+'/'+f[i]

        template_imgs = skimage.io.imread_collection(f) # loads all the images

        rescaled_template_imgs = []
        #skimage.io.imshow(template_imgs[0])

        for i in range(len(template_imgs)):
            image_rescaled = skimage.transform.rescale(template_imgs[i], 1.0 / 4.0, anti_aliasing=False) # preprocessing
            rescaled_template_imgs.append(image_rescaled)

        rescaled_template_imgs = np.array(rescaled_template_imgs)
        filtered_imgs = rescaled_template_imgs[mask]   
        hotEncodedLabel = np.array([i==n-1 for i in range(36)], dtype=int) # one hot encoding
        class_labels = np.full((len(filtered_imgs),len(hotEncodedLabel)), hotEncodedLabel)

        #skimage.io.imshow(rescaled_template_imgs[0])
        images.append(filtered_imgs)
        labels.append(class_labels)
        #print(template_imgs.shape)

    images = np.array(images).reshape((-1,1,32,32)) 
    labels = np.array(labels).reshape((-1,36))
    y = np.argmax(labels, axis=1)
    print(y.shape)
    
    mask0 = (y==0)
    mask1 = (y==1)
    mask01 = mask0 + mask1
    X = images[mask01]
    y = y[mask01].astype('int')
    n_samples = len(y)
    P = np.random.permutation(n_samples)
    X = X[P]
    y = y[P]
    
    def pooling(X, factor):
        X = skimage.measure.block_reduce(X, (1,1,factor,factor), np.mean)
        return X
    
    Xp = pooling(X, factor)
    Xp = 1. - Xp
    return Xp, y

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



