import numpy as np
import copy
from tqdm import tnrange
new = np.newaxis

class Tensor():
    """ 
    Class used to represent a Tensor.
    
    Attributes
    ----------
    elem: numpy array, optional
        Multi-dimensional matrix (ndarray) containing the elements of the tensor
    shape: tuple, optional
        Shape of the elem attribute
    axes_names: list of str, optional
        Names of each axis of the Tensor
    rank: int
        Rank of the tensor (i.e. length of the shape attribute)
    aggregations: dict of dict
        Each key is the name of an aggregated axis, whose values are the names of the 
        axes mapped in it and their original dimensions 
        
    Methods
    -------
    aggregate(axes_names=None, new_ax_name=None, debug=False)
        Maps multiple axes (axes_names) in a new one whose dimension is the product of the
        dimensions of the aggregated axis 
    disaggregate(ax)
        Unpack aggregated axis (ax) to original axes
 
    Notes
    -----
    If during initialization just the shape of the elements is provided, initializes the 
    elements randomly from the uniform distribution in [0,1] and then divides them by an
    heuristic factor sqrt(# elems/2) - KEEP IT UPDATED
    """
    
    def __init__(self, elem=None, shape=None, axes_names=None, scale=1.):
        """
        Parameters
        ----------
        elem: numpy array, optional
            Multi-dimensional matrix (ndarray) containing the elements of the tensor
        shape: tuple, optional
            Shape of the elem attribute
        axes_names: list of str, optional
            Names of each axis of the Tensor
        sigma: float, optional
            Not implemented
            
        Raises
        ------
        Exception
            If neither elem nor shape is provided
        ValueError
            If the number of names axes_names is different from the rank of the tensor
        TypeError
            If axes_names does not support the built-in len() function
        """
        # Numeric initialization
        if (elem is None) and (shape is not None):
            self.elem = np.random.random(size=shape) # uniform in [0,1]
            self.elem /= scale 
        elif elem is not None:
            self.elem = elem
        else:
            raise Exception('You have to provide either the elements of the tensor or its shape')
            
        # Relevant attributes initialization
        self.shape = self.elem.shape
        self.rank = len(self.shape)
        self.aggregations = {}

        if axes_names is not None:
            try:
                if len(axes_names) == self.rank:
                    # FIXME: history_axes_names is obsolete (check)
                    self.history_axes_names = [np.array(axes_names)] 
                    self.axes_names = np.array(axes_names)
                else:
                    raise ValueError("") # this error is handled with the except ValueError below
            except TypeError:
                print("=== Warning ===\nThe object that describes the indexes names have at least to support the built-in len function."\
                          +"\naxes_names attribute has not been inizialized.")
                self.axes_names = None
            except ValueError:
                print("=== Warning ===\nThe number of names should match the rank of the tensor."\
                          +"\naxes_names attribute has not been inizialized.")
                self.axes_names = None
        else:
            self.axes_names = None

        return

    def aggregate(self, axes_names=None, new_ax_name=None, debug=False):
        """ 
        Maps multiple axes (axes_names) in a new one whose dimension is the product of the
        dimensions of the aggregated axis 
        
        Parameters
        ----------
        axes_names: list of str, optional
            List containing the names of the axes to be aggregated
            (default is None and performs aggregation on all the axes)
        new_ax_name: str
            Name of the new axis (defautl is None, which raises a ValueError)
        debug: bool, optional
            If True, prints debugging information
            
        Raises
        ------
        ValueError
            If new_ax_name is not provided or if the Tensor has no names for the axes
        AssertionError
            If one of the names provided as input does not correspond to any axis' name 
        """
        
        dprint = print if debug else lambda *args, **kwargs : None
        
        # Sanity checks
        if (axes_names is None) and (new_ax_name is not None):
            axes_names = self.axes_names # if axes_names is None -> aggregate all axes
        elif new_ax_name is None:
            raise ValueError("You have to provide the name of the new axes")
            
        if self.axes_names is None:
            raise ValueError("This function can be called only if the axes names are defined")
            
        for name in axes_names:
            assert name in self.axes_names, "The " + name + " axes wasn't found in the tensor"
            
        dprint("Aggregating...")


        # Convert the axes names to their index positions
        indexes = self.ax_to_index(axes_names)
        
        # Store original shape of the aggregated indexes
        axes_sizes = np.array(self.shape)[indexes]
        self.aggregations[new_ax_name] = dict(zip(axes_names, axes_sizes))
        
        # Gather the non contracted indexes
        all_indexes = set(range(len(self.elem.shape)))
        other_indexes = list(all_indexes.difference(set(indexes)))
        other_indexes.sort()

        dprint("axes_numerical+other_axes: ", indexes+other_indexes)

        # Perform actual reshaping
        self.elem = np.transpose(self.elem, indexes+other_indexes)        
        other_sizes = np.array(self.shape)[other_indexes].tolist()
        self.elem = self.elem.reshape([-1]+other_sizes)
        
        # Update class members
        self.update_members(np.concatenate([[new_ax_name], self.axes_names[other_indexes]]))
        
        return
        

    def disaggregate(self, ax):
        """
        Unpack aggregated axis (ax) to original axes
        
        Parameters
        ----------
        ax: str
            Name of an aggregate axis to be disaggregated
           
        Raises
        ------
        AssertionError
            If the name provided does not correspond to an existing axis or to
            an axis that is not the result of an aggregation 
        """
        
        assert ax in self.axes_names, "The " + ax + " ax wasn't found in the tensor."
        assert ax in self.aggregations.keys(), "The " + ax + " does not represent an aggregated ax."
        
        original_dict = self.aggregations[ax]
        original_names = list(original_dict.keys())
        original_shape = list(original_dict.values())
        
        index = self.ax_to_index(ax)
        
        # transpose to have the aggregated index at the beginning
        permutation = [index] + np.arange(index).tolist() + np.arange(index+1, self.rank).tolist()
        self.elem = np.transpose(self.elem, permutation)
        self.update_members(self.axes_names[permutation])
        
        # Disaggregate axis by reshaping the tensor
        self.elem = self.elem.reshape(original_shape + list(self.shape[1:]))
        self.update_members(np.concatenate([original_names, self.axes_names[1:]]))
        
        # Remove aggregated index from the memory
        self.aggregations.pop(ax)
        
        return

    def transpose(self, permutation):
        """
        Changes the axes order according to the permutation of the names provided
        
        Parameters
        ----------
        permutation: list of str
            List containing the names of all the axes of the tensor in the order that
            must be obtained after the permutation
        """
        # permutation is axes_names in the new order
        indexes = self.ax_to_index(permutation)
        self.elem = np.transpose(self.elem, indexes)
        self.update_members(permutation)
        return

    def ax_to_index(self, axes):
        """
        Gets the indices associated to axes names given (axes)
        
        Parameters
        ----------
        axes: str or list of str
            Names of the axes of which we want to get the numerical
            positions (indices)
            
        Returns
        -------
        int or list of int
            Numerical indices of the axes
        """
        # handle single and multiple indices separately
        if type(axes) == str:
            return np.where(self.axes_names == axes)[0][0]
        else:
            return_axes = []
            for ax in axes:
                return_axes.append(np.where(self.axes_names == ax)[0][0])
            return return_axes

    def update_members(self, axes_names):
        """
        Updates axes names, shape and rank attributes after an aggregation or disaggregation
        
        Parameters
        ----------
        axes_names: list of str
            New names of the axes of the tensor
        """
        self.axes_names = np.array(axes_names)
        self.shape = self.elem.shape
        self.rank = len(self.shape)
        return
    
    def check_names(self):
        """
        Prints the type of axes_names attribute
        """
        print("="*10+"axes_names type"+"="*10)
        print(type(self.axes_names))
        
    def __str__(self):
        print("="*10+" Tensor description "+"="*10)
        print("Tensor shape: ", self.shape)
        print("Tensor rank: ", self.rank)
        print("Axes names: ", self.axes_names)
        return ""
    
    def __add__(self, o): 
        """
        Perform sum of two tensors permuting the axes of the second so that they are alligned.
        """

        # check all names match between two tensors
        assert np.all(np.isin(self.axes_names, o.axes_names)), "Error: axes don't match, cannot sum tensors."

        o.transpose(self.axes_names)
        t3 = self.elem + o.elem
        T3 = Tensor(elem = t3, axes_names = self.axes_names)
        return T3

    
def _contract_(T1, T2, contracted_axis1, contracted_axis2, common_axis1=[], common_axis2=[]):
    """
    Contracts two tensors along one axis.
    
    Parameters
    ----------
    T1, T2: Tensor
        Tensors to be contracted
    contracted_axis1, contracted_axis2: list of int
        Indices of the axes of T1 and T2 to be contracted
    common_axis1, common_axis2: list of int, optional
         Indices of the axes of T1 and T2 in common
    
    Returns
    -------
    Tensor
        Contracted tensors
        
    Raises
    ------
    AssertionError
        If the number of common axes is different or if the dimensions of contracted or common axes 
        do not match between tensors
    """

    # Sanity checks
    assert len(common_axis1) == len(common_axis2), "number of common axes is different"
    
    if type(contracted_axis1) != list:
        # assuming contracted_axis1/2 is numeric
        assert T1.shape[contracted_axis1] == T2.shape[contracted_axis2], "dimensions of contracted axes do not match"
        contracted_axis1 = [contracted_axis1]
        contracted_axis2 = [contracted_axis2]
    
    for i in range(len(common_axis1)):
        # assuming common_axis1/2 is numeric
        assert T1.shape[common_axis1[i]] == T2.shape[common_axis2[i]], "dimensions of common axes do not match"
        
    original_shape1 = np.array(T1.shape)
    original_shape2 = np.array(T2.shape)
        
    def perm(contracted_axis, original_shape, common_axis):
        # assuming contracted_axis and common_axis list of integers
        # astype is for handling the case in the first array is empty, 
        # in which the function cannot infer the type
        last_axis = np.concatenate((common_axis, contracted_axis)).astype("int64")         

        remaining_axis = np.delete(np.arange(len(original_shape)), last_axis)
        permutation = np.concatenate((remaining_axis, last_axis))
        return permutation

    permutation1 = perm(contracted_axis1, original_shape1, common_axis1)
    permutation2 = perm(contracted_axis2, original_shape2, common_axis2)

    shape1 = original_shape1[permutation1]
    shape2 = original_shape2[permutation2]

    # param for match the rank of the two shapes
    unique1 = len(shape1)-len(common_axis1)-len(contracted_axis1)
    unique2 = len(shape2)-len(common_axis1)-len(contracted_axis1)

    new_shape1 = np.concatenate((shape1[:unique1],[1 for i in range(unique2)],shape1[unique1:])).astype("int64")
    new_shape2 = np.concatenate(([1 for i in range(unique1)],shape2)).astype("int64")
    
    T1.transpose(T1.axes_names[permutation1])
    T2.transpose(T2.axes_names[permutation2])
    
    T3_axes_names = np.concatenate([T1.axes_names[:unique1], T2.axes_names[:T2.rank-len(contracted_axis2)]])
    #else: 
    #    T3_axes_names = None

    T3 = (T1.elem.reshape(new_shape1)*T2.elem.reshape(new_shape2))
    if len(contracted_axis1) > 0:
        # if len(contracted_axis1) == 0 just to tensor product
        T3 = T3.sum(axis=-1)
        
    T3 = Tensor(elem=T3, axes_names=T3_axes_names)
    return T3



def contract(T1, T2, contracted_axis1=[], contracted_axis2=[], common_axis1=[], common_axis2=[], contracted=None, common=None):
    """
    Contracts two tensors along one axis.
    
    Parameters
    ----------
    T1, T2: Tensor
        Tensors to be contracted
    contracted_axis1, contracted_axis2: str, optional
        Names of the axes of T1 and T2 to be contracted
    common_axis1, common_axis2: str or list of str, optional
        Names of the axes of T1 and T2 in common
    contracted: str, optional
        Shortcut for contracted_axis1/2 if the names are the same
    common: str or list of str, optional
        Shortcut for common_axis1/2 if the names are the same
    
    Returns
    -------
    Tensor
        Contracted tensors
        
    Notes
    -----
    This function is a wrapper for the _contract_ function, where the actual contracion is performed
    (see help(tn._contract_) for more info)
    
    TODO: check that contracted_axis1 and 2 are strings and not lists
    """
    
    # if contracted is specified, assign its value to both contracted axes
    if contracted is not None:
        contracted_axis1 = contracted
        contracted_axis2 = contracted
 
    # if common is specified, assign its value to both common axes
    if common is not None:
        common_axis1 = common
        common_axis2 = common
        
    # If common_axis1/2 are provided as int or str, redefine them as lists
    if type(common_axis1) != list:
        common_axis1 = [common_axis1]
    if type(common_axis2) != list:
        common_axis2 = [common_axis2]

    # If contracted_axis1/2 is a string, get the corresponding numerical axis
    if type(contracted_axis1) == str:
        contracted_axis1 = T1.ax_to_index(contracted_axis1)
    if type(contracted_axis2) == str:
        contracted_axis2 = T2.ax_to_index(contracted_axis2)
       
    # If common_axis1/2 is a list of strings, get the corresponding numerical axes
    temp = []
    for key in common_axis1:
        if type(key) == str:
            temp.append(np.where(T1.axes_names == key)[0][0])
        else:
            temp.append(key)
    common_axis1 = temp
    # should also work something like common_axis1 = T1.ax_to_index(common_axis1)

    temp = []
    for key in common_axis2:
        if type(key) == str:
            temp.append(np.where(T2.axes_names == key)[0][0])
        else:
            temp.append(key)
    common_axis2 = temp
    
    # call _contract_ function for actual contraction
    return _contract_(T1, T2, contracted_axis1, contracted_axis2, common_axis1, common_axis2)


def partial_trace(T, ax1, ax2):
    """
    Compute the partial trace of a tensor (contraction of two axis between themselves)
    
    Parameters
    ----------
    T: Tensor
        Tensor on which partial trace is performed
    ax1, ax2: str
        Names of the two axes to be traced out
        
    Returns
    -------
    Tensor
        Tensor obtained through partial trace
        
    TODO: write assertion to see if dimensions of ax1 and 2 match
    """
    traced_axes = np.array([ax1,ax2])
    traced_indexes = T.ax_to_index(traced_axes)
    remaining_axis = np.delete(T.axes_names, traced_indexes)
    permutation = np.concatenate((traced_axes, remaining_axis))
    T.transpose(permutation)
    t = T.elem.trace(axis1=0, axis2=1)
    T = Tensor(elem=t, axes_names=remaining_axis)
    return T



def tensor_svd(T, threshold=0.999):
    """
    Performs Singular Value Decomposition (SVD) on a 2D Tensor
    
    Parameters
    ----------
    T : Tensor
    threshold: float in (0,1)
        Minimum of variance percentage to be explained after truncation
        
    Returns
    -------
    TU : Tensor
        Tensor correspondig to matrix U of U S Vh decomposition
        If original shape was (a,b), it has shape (a,m')
    TSVh : Tensor
        Tensor correspondig to matrix S Vh of U S Vh decomposition
        If original shape was (a,b), it has shape (m',b)
        
    Raises
    ------
    TypeError
        If T is not a Tensor
    ValueError
        If T is not a 2D Tensor
        
    Notes
    -----
    Currently the minimum bond dimension is set to 10. Adaptive thresholding 
    technique is still to be tested in depth before removing this constraint.
    """
    
    if type(T) != Tensor:
        raise TypeError("This function only support object from the class Tensor")

    if len(T.shape) != 2:
        raise ValueError("This function only support a 2D tensors")

    debug = False
    if debug:
        print('\nSVD debug: ', debug)
        print('T.elem.shape: ', T.elem.shape)
        print('T.elem.sum(): ', T.elem.sum())
    # perform SVD of T elements
    U, S, Vh = np.linalg.svd(copy.deepcopy(T.elem))

    # compute adaptive bond dimension
    cumulative_variance_explained = np.cumsum(S)/S.sum()
    #print(cumulative_variance_explained)
    index = np.argmax(cumulative_variance_explained>threshold)
    m = T.aggregations['i']['left']
    if debug:
        print('m: ', m)
    #####################################
    #m_new = max(10,min(index,m))
    m_new = m # debug
    #####################################
    # truncate tensors according to the new bond dimension
    Vh = Vh[:m_new,:]
    U = U[:,:m_new]
    S = np.eye(m_new,m_new)*S[:m_new]
    SVh = np.dot(S, Vh) # new is np.newaxis
    
    T_new = np.dot(U,SVh)
    
    if debug:
        print('T_new.shape: ', T_new.shape)
        print('T_new.sum(): ', T_new.sum())
        print('U.shape', U.shape)
        print('U.sum', U.sum())
        print('S.shape', S.shape)
        print('S.sum', S.sum())
        print('Vh.shape', Vh.shape)
        print('Vh.sum', Vh.sum())
        print('SVh.shape', SVh.shape)
        print('SVh.sum', SVh.sum())
        
    # building new tensors
    TU = Tensor(elem=U, axes_names=['i','right'])
    TSVh = Tensor(elem=SVh, axes_names=['left','j'])
    TU.aggregations['i'] = T.aggregations['i']
    TSVh.aggregations['j'] = T.aggregations['j']
    
    # retrieving original dimensions
    TU.disaggregate('i')
    TSVh.disaggregate('j')
    
    return TU, TSVh


class Network():
    """ 
    Class used to represent a Matrix Product State (MPS) Tensor Network.
    
    Attributes
    ----------
    N: int
        First dimension of the input (e.g. number of pixels for an image)
    D: int
        Second dimension of the input (e.g. number of color channels or
        embedding dimension)
    L: int
        Number of possible labels
    M: int
        Initial bond dimension
    normalize: bool 
        If True, normalizes the weights so that the expected output is of order 1
        for inputs X with all entries in [0,1]   
    As: list of Tensors
        List of matrix-like tensors of the MPS network
    l_pos: int
        Index of As at which there is the Tensor with the label dimension 
    TX: list of Tensors
        Object to wrap the input as a list of Tensors in order to perform
        contractions with the As
    cum_contraction: list of Tensors
        List of cumulative contractions from the "right" to the "left" of the network
        Calling A_TX the list of the contracted tensors As[i] TX[i], the first element 
        of cum_contraction is A_TX[-1], then the second is the contraction of A_TX[-1]
        and A_TX[-2], the third the contraction of cum_contraction[1] with A_TX[-3] 
        and so on
    left_contraction: Tensor
        Cumulative contractions from the "right" to the "left" of the network
        
    Methods
    -------
    forward(X)
        Computes predictions for input X
    train(train_loader, val_loader, lr, n_epochs = 10, early_stopping = False, print_freq = 100)
        Trains the network. 
    accuracy(x, y, f=None)
        Computes the accuracy of the networks' predictions
        
    Notes
    -----
    This class implements a circular MPS Tensor Network, meaning that the network topology is a ring
    where all the tensors are contracted both to the left and to the right.
    """
    
    def __init__(self, N, M, D=2, L=10, normalize=False, calibration_X=None):
        """
        Parameters
        ----------
        N: int
            First dimension of the input (e.g. number of pixels for an image)
        D: int
            Second dimension of the input (e.g. number of color channels or
            embedding dimension)
        L: int
            Number of possible labels
        M: int
            Initial bond dimension
        normalize: bool 
            If True, normalizes the weights so that the expected output is of order 1
            for inputs X with all entries in [0,1]
        calibration_X: numpy array
            Shape (batch_size, N, D)
        """
        
        self.N = N
        self.D = D
        self.L = L
        self.M = M
        
        self.As = []
        
        if normalize:
            print('Normalizing weights...')
            K = (self.D*self.N)**(-1./self.N)
            print('K', K)
            scale = float(self.M)/(3*K)
            print('Scaling factor: %.2f'%scale)
            self.As.append(Tensor(shape=[L,M,M,D], axes_names=['l','left','right','d0'], scale = scale))
            for i in range(1,N):
                self.As.append(Tensor(shape=[M,M,D], axes_names=['left','right','d'+str(i)], scale = scale))
        else:
            self.As.append(Tensor(shape=[L,M,M,D], axes_names=['l','left','right','d0']))
            for i in range(1,N):
                self.As.append(Tensor(shape=[M,M,D], axes_names=['left','right','d'+str(i)]))
                
        self.l_pos = 0
        
        if normalize:
            if calibration_X is None:
                B = 16 # batch size of the samples used for calibration
                X = np.random.random((B, self.N))
                print(X.shape)
                def psi(x):
                    x = np.array((np.sin(np.pi*x/2),np.cos(np.pi*x/2)))
                    return np.transpose(x, [1,2,0])
                X = psi(X)
            else:
                X = calibration_X
                B = X.shape[2]
            
            print(X.shape)
            f = self.forward(X)
            f_max = np.abs(f.elem).max().astype('float')
            print('f_max for random input of %d samples : '%(B),f_max)
            F2 = f_max**(1./self.N)
            print("F2: ", F2)
            sum_of_A = 0 # debug
            for i in range(self.N):
                #self.As[i].elem = copy.deepcopy(self.As[i].elem)/F2
                self.As[i].elem = self.As[i].elem/F2
                sum_of_A += self.As[i].elem.sum() # debug
            print("Sum of all elements of all As (after init): ", sum_of_A) # debug
            f = self.forward(X)  
            f_max = np.abs(f.elem).max().astype('float')
            print('f_max for random input of %d samples (after): '%(B),f_max)
            
            
        return
        
    def forward(self, X):
        """
        Computes predictions for input X
        
        Parameters
        ----------
        X : numpy array
            Shape (batch_size, N, D) (see init parameters for better explanation)
            
        Returns
        -------
        out: Tensor
            Tensor of shape (batch_size, L), where L is the number of possible labels
            out.elem shows the scores assigned by the network to each label
            (the argmax along the first axis yields the predicted label for each sample
            in the batch)
            
        Raises
        ------
        AssertionError
            If X.shape[1] does not match with self.N
        """
        
        assert self.N == X.shape[1], "The 1 dimension of the input data must be the flattened number of pixels"

        # X should be batch_size x 784 x 2
        TX = []
        for i in range(self.N):
            TX.append(Tensor(elem=X[:,i,:], axes_names=['b','d'+str(i)]))
                      
        # This must be futher investigate, three ways:
        #     * numpy vectorize
        #     * list comprehension
        #     * multithread
                      
        #A_TX = np.vectorize(contract)(TX, A, contracted='d'+str(i))
        
        ##############################################################################
        #sum_of_A = 0 # debug
        #for i in range(self.N):
        #        sum_of_A += self.As[i].elem.sum() # debug
        #print("Sum of all elements of all As (before forward): ", sum_of_A) # debug
        ##############################################################################
        
        A_TX = [contract(self.As[i], TX[i], contracted='d'+str(i)) for i in range(self.N)]
        cum_contraction = []
        cum_contraction.append(A_TX[-1])
        for j in range(1,self.N): # instead of self.N - 1
            tmp_cum = copy.deepcopy(cum_contraction[-1])
            tmp_cum = contract(A_TX[-(j+1)], tmp_cum, 'right', 'left', common='b')
            cum_contraction.append(tmp_cum)

        self.cum_contraction = cum_contraction[::-1]
        self.left_contraction = None
        self.TX = TX

        out = partial_trace(self.cum_contraction[0], 'right', 'left') # close the circle
        return out

    def train(self, train_loader, val_loader, lr, n_epochs = 10, early_stopping = False, print_freq = 100):
        """
        Trains the network
        
        Parameters
        ----------
        train_loader, val_loader: DataLoader (from torch.utils.data)
            Data loaders for training and validation set
            Must yield numpy arrays of shape (batch_size, N, D)
        lr: float in (0,1]
            Learning rate that multiplies the gradient
        n_epochs: int, optional
            Number of epochs of training (default=10). One epoch consists in a full
            sweep for each batch in the train_loader.
        early_stopping: bool, optional
            Not implemented. Default is False.
        print_freq: int, optional
            Number of batches after which prints a training update
            
        Returns
        -------
        train_acc, val_acc: list of float
            List of accuracies on training and validation sets for each epoch
        """
        
        train_acc = []
        val_acc = []
        # if early_stopping = False
        for epoch in tnrange(n_epochs, desc="Epoch loop", leave = True):
            epoch_train_acc = np.zeros(len(train_loader))
           
            # train
            print_every = int(len(train_loader)/print_freq)
            for i, data in enumerate(train_loader, 0):
                x = np.array([data[i][0] for i in range(len(data))])
                #f_debug = self.forward(x)
                #print("f_debug.elem: ", f_debug.elem)  # incorrect of 10^-11 order
                y = np.array([data[i][1] for i in range(len(data))])
                f = self.sweep(x, y, lr)
                batch_acc = self.accuracy(x, y, f)
                epoch_train_acc[i] = batch_acc
                if (i+1) % (print_every) == 0:
                    print('\r'+"Epoch %d - train accuracy : %.4f - completed : %.2f "%(epoch, epoch_train_acc[:i].mean(), (i+1)*100/len(train_loader))+'%', end=' ')
            train_acc.append(epoch_train_acc.mean())
            
            # validation
            epoch_val_acc = np.zeros(len(train_loader))
            batch_len = np.zeros(len(train_loader))
            for i, data in enumerate(val_loader, 0):
                x = np.array([data[i][0] for i in range(len(data))])
                y = np.array([data[i][1] for i in range(len(data))])
                batch_acc = self.accuracy(x, y)
                epoch_val_acc[i] = batch_acc
                batch_len[i] = len(x)
                if (i+1) % (print_every) == 0:
                    tmp_val_acc = epoch_val_acc[:i].sum()/batch_len[:i].sum()
                    print('\r'+"Epoch %d - train accuracy : %.4f - val accuracy: %.4f"%(epoch, train_acc[-1], tmp_val_acc), end=' ')

            epoch_val_acc *= batch_len
            val_acc.append(epoch_val_acc.sum()/batch_len.sum())
            print('\r'+"Epoch %d - train accuracy : %.4f - val accuracy: %.4f"%(epoch, train_acc[-1], val_acc[-1]))
        
        return train_acc, val_acc
    
    def accuracy(self, X, y, f=None):
        """
        Computes the accuracy of the networks' predictions
        
        Parameters
        ----------
        X: numpy array
            Test set of shape (samples, N, D)
        y: numpy array
            Prediction targets of shape (samples,)
            This must not be provided as one hot encoded
        f: Tensor, optional
            If provided should be Network.forward(X)
            
        Returns
        -------
        accuracy: float
            Fraction of correctly labelled samples over the total
            number of samples
        """
        
        if f is None:
            f = self.forward(X)
        y_pred = np.argmax(f.elem, axis=0)
        errors = (y!=y_pred).sum()
        accuracy = (len(y_pred)-errors)/len(y_pred)
        return accuracy
    
    def sweep(self, X, y, lr):
        """
        Makes an optimization "sweep", consisting of optimizing each pair
        of adjacent Tensors As[i] As[i+1]
        
        Parameters
        ----------
        X : numpy array of float
            Shape (batch_size, N, D) (see init parameters for better explanation)
        y: numpy array of int
            Prediction targets of shape (batch_size,)
        lr: float in (0,1]
            Learning rate that multiplies the gradient
            
        Returns
        -------
        f: Tensor
            Equivalent to self.forward(X) after the optimization sweep
        """
        
        batch_size = len(y)
        f = self.forward(X)
        #print("Target: \n", y)
        one_hot_y = np.zeros((y.size, self.L))
        one_hot_y[np.arange(y.size),y] = 1
        y = one_hot_y.T
        
        for i in range(self.N-1):
            print("\nsweep step ",i)
            f = self.sweep_step(f, y, lr, batch_size)
            ####################
            #sum_of_A = 0 # debug
            #for i in range(self.N):
            #        sum_of_A += self.As[i].elem.sum() # debug
            #print("Sum of all elements of all As (before forward): ", sum_of_A) # debug
            ####################
            
        ### ERROR IN THIS PART ###
        # svd 
        B = contract(self.As[-1], self.As[0], "right", "left")
        # reconstruct optimized network tensors
        B.aggregate(axes_names=['d'+str(self.l_pos),'left'], new_ax_name='i')
        B.aggregate(axes_names=['d0','right','l'], new_ax_name='j')
        B.transpose(['i','j'])
        self.As[-1], self.As[0] = tensor_svd(B)
        ##########################
        
        self.l_pos = 0
        return f
    
    def sweep_step(self, f, y, lr, batch_size):
        """
        Makes a step of the optimization "sweep", consisting in the optimization of
        a pair of Tensors As[i] As[i+1]
        
        Parameters
        ----------
        f: Tensor
            Equivalent to self.forward(X)
        y: numpy array of int
            One hot encoded version of the prediction targets
            Shape is (batch_size,L)
        batch_size: int
            Number of samples in a batch
            
        Returns
        -------
        f: Tensor
            Equivalent to self.forward(X) after optimization step
        """
        
        # ID of the node A at which the output of the net is computed
        l = self.l_pos
        
        # (always true)
        B = contract(self.As[l], self.As[l+1], "right", "left")

        # (always true)
        # computing all elements for delta_B
        # Contributions:
        # - TX[l]    (always))
        # - TX[l+1]    (always))
        # - left_contribution    (for l > 0)
        # - cum_contraction[-(l+2)]    (for l < N-2)
        # - y-f    (always)
        
        phi = contract(self.TX[l], self.TX[l+1], common="b")
        
        if l==0:
            # tensor product with broadcasting on batch axis
            phi = contract(phi, self.cum_contraction[l+2], common = "b")
        
        elif (l > 0) and (l<(self.N-2)):
            # compute new term for the left contribute
            new_contribution = contract(self.As[l-1], self.TX[l-1], contracted='d'+str(l-1))
            if l==1:
                # define left_contraction (['right','b'])
                self.left_contraction = new_contribution
            else:
                # update left_contraction (['right','b'])
                self.left_contraction = contract(self.left_contraction, new_contribution, 'right', 'left', common='b')
            circle_contraction = contract(self.cum_contraction[l+2], self.left_contraction, 'right', 'left', common='b')
            # tensor product with broadcasting on batch axis
            phi = contract(phi, circle_contraction, common = "b")
            
        else:
            new_contribution = contract(self.As[l-1], self.TX[l-1], contracted='d'+str(l-1))
            
            # update left_contraction (['right','b'])
            self.left_contraction = contract(self.left_contraction, new_contribution, 'right', 'left', common='b')
            
            # tensor product with broadcasting on batch axis
            phi = contract(phi, self.left_contraction, common = "b")
            
        ######################################################
        y_pred = np.argmax(f.elem, axis=0) 
        y_target = np.argmax(y, axis=0) 
        #print("Target: ", y_target)
        #print("Prediction (before optim.): ", y_pred)
        
        errors = (y_target!=y_pred).sum()
        accuracy = (len(y_pred)-errors)/len(y_pred)
        MSE = ((y-f.elem)**2).mean()
        print("Accuracy (before optim.): ", accuracy)
        print("MSE (before optim.): ", MSE)
        ######################################################
        
        f.elem = y-f.elem
  
        deltaB = contract(f, phi, contracted="b")
        # gradient clipping
        if np.abs(deltaB.elem).sum() > 1000:
            deltaB.elem /= np.abs(deltaB.elem).sum()/1000
        deltaB.elem *= lr

        left_index = deltaB.ax_to_index('left')
        right_index = deltaB.ax_to_index('right')
        deltaB.axes_names[left_index] = 'right'
        deltaB.axes_names[right_index] = 'left'
        # update B
        debug = False
        if debug:
            print('\nB.elem.sum(): ', B.elem.sum())
        print('deltaB.elem.sum(): ', deltaB.elem.sum())
        # just trying to regularize
        B.elem *= (1-lr)
        B = B + deltaB
        if debug:
            print('B.elem.sum() (after update): ', B.elem.sum())
        # compute new output of the net
        out = contract(B, self.TX[l], contracted='d'+str(l))
        out = contract(out, self.TX[l+1], contracted='d'+str(l+1), common='b')
        if l == 0:
            out = contract(out, self.cum_contraction[l+2], 'right', 'left', common = "b")
        elif (l > 0) and (l<(self.N-2)):
            out = contract(self.left_contraction, out, 'right', 'left', common = "b")
            out = contract(out, self.cum_contraction[l+2], 'right', 'left', common = "b")
        else:
            out = contract(self.left_contraction, out, 'right', 'left', common = "b")
        
        out = partial_trace(out, 'right', 'left') # close the circle
        if debug:
            print("Shape of the outputs: ", out.elem.shape)
            print("Total sum of the outputs: ", np.abs(out.elem).sum())
           
        ######################################################
        y_pred = np.argmax(out.elem, axis=0) 
        #print("Prediction (after optim.): ", y_pred)
        errors = (y_target!=y_pred).sum()
        accuracy = (len(y_pred)-errors)/len(y_pred)
        MSE = ((y-out.elem)**2).mean()
        print("Accuracy (after optim.): ", accuracy)
        print("MSE (after optim.): ", MSE)
        ######################################################
        
        # reconstruct optimized network tensors
        B.aggregate(axes_names=['d'+str(l),'left'], new_ax_name='i')
        B.aggregate(axes_names=['d'+str(l+1),'right','l'], new_ax_name='j')
        B.transpose(['i','j'])
        self.As[l], self.As[l+1] = tensor_svd(B)
        
        # update position of l
        self.l_pos += 1
        
        return out
    
    
        
    
        
