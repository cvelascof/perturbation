"""Interface and utility methods for different generators."""

import numpy as np

# Use the SciPy fft by default. If SciPy is not installed, fall back to the 
# numpy implementation.
try:
    import scipy.fftpack as fft
except ImportError:
    from numpy import fft
    
def get_method(name):
    """Return two callable functions to initialize and generate 2d perturbations
    for precipitation fields. The available options are:\n\
    
    +-------------------+--------------------------------------------------------+
    |     Name          |              Description                               |
    +===================+========================================================+
    |  nonparametric    | this generator uses global Fourier filering.           |
    +-------------------+--------------------------------------------------------+
    |  nested           | this generator uses a local Fourier filtering          |
    +-------------------+--------------------------------------------------------+
    """
    if name == "nonparametric":
        return initialize_nonparam_2d_fft_filter, generate_noise_2d_fft_filter
    elif name == "nested":
        return initialize_nonparam_2d_nested_filter, generate_noise_2d_ssft_filter
    else:
        raise ValueError("unknown method %s" % name)
 
def initialize_nonparam_2d_fft_filter(X, tapering_function='flat-hanning', donorm=False):
    """Takes a 2d input field and produces a fourier filter by using the Fast 
    Fourier Transform (FFT).
    
    Parameters
    ----------
    X : array-like
      Two-dimensional array containing the input field. All values are required 
      to be finite.
    tapering_function : string
       Optional tapering function to be applied to X.
       (hanning, flat-hanning)
    
    Returns
    -------
    F : array-like
      A two-dimensional array containing the non-parametric filter.
      It can be passed to generate_noise_2d_fft_filter().
    """
    if len(X.shape) != 2:
        raise ValueError("the input is not two-dimensional array")
    if np.any(~np.isfinite(X)):
      raise ValueError("X contains non-finite values")
      
    X = X.copy()
    if tapering_function is not None:
        X -= X.min()
        tapering = build_2D_tapering_function(X.shape, tapering_function)
    else:
        tapering = np.ones_like(X)
    F = fft.fft2(X*tapering)
    
    # normalize the real and imaginary parts
    if donorm:
        F.imag = (F.imag - np.mean(F.imag))/np.std(F.imag)
        F.real = (F.real - np.mean(F.real))/np.std(F.real)
    
    return np.abs(F)
 
def generate_noise_2d_fft_filter(F, seed=None):
    """Produces a field of correlated noise using global Fourier filering.
    
    Parameters
    ----------
    F : array-like
        Two-dimensional array containing the input filter. 
        It can be computed by related methods.
        All values are required to be finite.
    seed : int
        Value to set a seed for the generator. None will not set the seed.
    
    Returns
    -------
    N : array-like
        A two-dimensional numpy array of stationary correlated noise.
    """
    
    if len(F.shape) != 2:
        raise ValueError("the input is not two-dimensional array")
    if np.any(~np.isfinite(F)):
      raise ValueError("F contains non-finite values")
      
    # set the seed
    np.random.seed(seed)
    
    # produce fields of white noise
    N = np.random.randn(F.shape[0], F.shape[1])
    
    # apply the global Fourier filter to impose a correlation structure
    fN = fft.fft2(N)
    fN *= F
    N = np.array(fft.ifft2(fN).real)
    N = (N - N.mean())/N.std()
            
    return N
   
def initialize_nonparam_2d_nested_filter(X, gridres=1.0, **kwargs):
    """Function to compute the local Fourier filters using a nested approach.

    Parameters
    ----------
    X : array-like
        Two-dimensional array containing the input field. All values are required 
        to be finite and the domain must be square.
    gridres : float
        Grid resolution in km.
    Optional kwargs:
    ----------
    max_level : int 
        Localization parameter. 0: global noise, >0: increasing degree of localization.
    win_type : string ['hanning', 'flat-hanning'] 
        Type of window used for localization.
    war_thr : float [0;1]
        Threshold for the minimum fraction of rain needed for computing the FFT.

    Returns
    -------
    F : array-like
        Four-dimensional array containing the 2d fourier filters distributed over
        a 2d spatial grid.


    """
    
    if len(X.shape) != 2:
        raise ValueError("X must be a two-dimensional array")
    if X.shape[0] != X.shape[1]:
        raise ValueError("X must have a square domain")
    if np.any(np.isnan(X)):
        raise ValueError("X must not contain NaNs")
        
    # Set default parameters
    max_level = kwargs.get('max_level', 3)
    win_type = kwargs.get('win_type', 'flat-hanning')
    war_thr = kwargs.get('war_thr', 0.1)
    
    # make sure non-rainy pixels are set to zero
    min_value = np.min(X)
    X = X.copy()
    X -= min_value
    
    # 
    dim = X.shape
    dim_x = dim[1]
    dim_y = dim[0]
       
    # Nested algorithm 
    
    # prepare indices
    Idxi = np.array([[0, dim_y]])
    Idxj = np.array([[0, dim_x]])
    Idxipsd = np.array([[0, 2**max_level]])
    Idxjpsd = np.array([[0, 2**max_level]])
    
    # generate the FFT sample frequencies
    freq = fft.fftfreq(dim_y, gridres)
    fx,fy = np.meshgrid(freq, freq)
    freq_grid = np.sqrt(fx**2 + fy**2)
    
    # domain fourier filter
    F0 = initialize_nonparam_2d_fft_filter(X, win_type, True)
    # and allocate it to the final grid
    F = np.zeros((2**max_level, 2**max_level, F0.shape[0], F0.shape[1]))
    F += F0[np.newaxis, np.newaxis, :, :]
    
    # now loop levels and build composite spectra
    level=0 
    while level < max_level:

        for m in xrange(len(Idxi)):
        
            # the indices of rainfall field
            Idxinext, Idxjnext = _split_field(Idxi[m, :], Idxj[m, :], 2)
            # the indices of the field of fourier filters
            Idxipsdnext, Idxjpsdnext = _split_field(Idxipsd[m, :], Idxjpsd[m, :], 2)
            
            for n in xrange(len(Idxinext)):
            
                mask = _get_mask(dim[0], Idxinext[n, :], Idxjnext[n, :], win_type)
                war = np.sum((X*mask) > 0.01)/float((Idxinext[n, 1] - Idxinext[n, 0])**2)
                
                if war > war_thr:
                    # the new filter 
                    newfilter = initialize_nonparam_2d_fft_filter(X*mask, None, True)
                    
                    # compute logistic function to define weights as function of frequency
                    # k controls the shape of the weighting function
                    k = 0.05
                    x0 = (Idxinext[n, 1] - Idxinext[n, 0])/2.
                    merge_weights = 1/(1 + np.exp(-k*(1/freq_grid - x0)))
                    newfilter *= (1 - merge_weights)
                    
                    # perform the weighted average of previous and new fourier filters
                    F[Idxipsdnext[n, 0]:Idxipsdnext[n, 1], Idxjpsdnext[n,0]:Idxjpsdnext[n, 1], :, :] *= merge_weights[np.newaxis, np.newaxis, :, :]
                    F[Idxipsdnext[n, 0]:Idxipsdnext[n, 1],Idxjpsdnext[n, 0]:Idxjpsdnext[n, 1], :, :] += newfilter[np.newaxis, np.newaxis, :, :] 
            
        # update indices
        level += 1
        Idxi, Idxj = _split_field((0, dim[0]), (0, dim[1]), 2**level)
        Idxipsd, Idxjpsd = _split_field((0, 2**max_level), (0, 2**max_level), 2**level)
        
    return F
   
def generate_noise_2d_ssft_filter(F, seed=None, **kwargs):
    """Function to compute the locally correlated noise using a nested approach.

    Parameters
    ----------
    F : array-like
        Four-dimensional array containing the 2d fourier filters distributed over
        a 2d spatial grid.
    seed : int
        Value to set a seed for the generator. None will not set the seed.
        
    Optional kwargs:
    ----------
    overlap : float 
        Percentage overlap [0-1] between successive windows.
    win_type : string ['hanning', 'flat-hanning'] 
        Type of window used for localization.

    Returns
    -------
    N : array-like
        A two-dimensional numpy array of non-stationary correlated noise.

    """
    
    if len(F.shape) != 4:
        raise ValueError("the input is not four-dimensional array")
    if np.any(~np.isfinite(F)):
      raise ValueError("F contains non-finite values")
      
    # Set default parameters
    overlap = kwargs.get('overlap', 0.2)
    win_type = kwargs.get('win_type', 'flat-hanning')
    
    # set the seed
    np.random.seed(seed)
    
    dim_y = F.shape[2]
    dim_x = F.shape[3]
    
    # produce fields of white noise
    N = np.random.randn(dim_y, dim_x)
    fN = fft.fft2(N)
    
    # initialize variables
    cN = np.zeros((dim_y, dim_x))
    sM = np.zeros((dim_y, dim_x))
    
    idxi = np.zeros((2, 1), dtype=int)
    idxj = np.zeros((2, 1), dtype=int)
    
    # get the window size
    winsize = np.round( dim_y/float(F.shape[0]) )
    
    # loop the windows and build composite image of correlated noise

    # loop rows
    for i in xrange(F.shape[0]):
        # loop columns
        for j in xrange(F.shape[1]):
        
            # apply fourier filtering with local filter
            lF = F[i,j,:,:]
            flN = fN * lF
            flN = np.array(np.fft.ifft2(flN).real)
            
            # compute indices of local window
            idxi[0] = np.max( (int(i*winsize - overlap*winsize), 0) )
            idxi[1] = np.min( (int(idxi[0] + winsize  + overlap*winsize), dim_y) )
            idxj[0] = np.max( (int(j*winsize - overlap*winsize), 0) )
            idxj[1] = np.min( (int(idxj[0] + winsize  + overlap*winsize), dim_x) )
            
            # build mask and add local noise field to the composite image
            M = _get_mask(dim_y, idxi, idxj, win_type)
            cN += flN*M
            sM += M 

    # normalize the field
    cN[sM > 0] /= sM[sM > 0]         
    cN = (cN - cN.mean())/cN.std()
            
    return cN
        
def build_2D_tapering_function(winsize, wintype='flat-hanning'):
    """Produces two-dimensional tapering function for rectangular fields.

    Parameters
    ----------
    winsize : tuple of int
        Size of the tapering window as two-element tuple of integers.
    wintype : str
        Name of the tapering window type (hanning, flat-hanning)
    Returns
    -------
    w2d : array-like
        A two-dimensional numpy array containing the 2D tapering function.
    """
    
    if wintype == 'hanning':
        w1dr = np.hanning(winsize[0])
        w1dc = np.hanning(winsize[1])
        
    elif wintype == 'flat-hanning':
    
        T = winsize[0]/4.0
        W = winsize[0]/2.0
        B = np.linspace(-W,W,2*W)
        R = np.abs(B)-T
        R[R < 0] = 0.
        A = 0.5*(1.0 + np.cos(np.pi*R/T))
        A[np.abs(B) > (2*T)] = 0.0
        w1dr = A
        
        T = winsize[1]/4.0
        W = winsize[1]/2.0
        B = np.linspace(-W, W, 2*W)
        R = np.abs(B) - T
        R[R < 0] = 0.
        A = 0.5*(1.0 + np.cos(np.pi*R/T))
        A[np.abs(B) > (2*T)] = 0.0
        w1dc = A   
        
    else:
        print("Unknown window type, returning a rectangular window.")
        w1dr = np.ones(winsize[0])
        w1dc = np.ones(winsize[1])
    
    # Expand to 2-D
    w2d = np.sqrt(np.outer(w1dr,w1dc))
    
    # Set nans to zero
    if np.sum(np.isnan(w2d)) > 0:
        w2d[np.isnan(w2d)] = np.min(w2d[w2d > 0])

    return w2d
    
def _split_field(idxi, idxj, Segments):
    ''' Split domain field into a number of equally sapced segments.
    '''

    sizei = (idxi[1] - idxi[0]) 
    sizej = (idxj[1] - idxj[0]) 
    
    winsizei = int(sizei/Segments)
    winsizej = int(sizej/Segments)
    
    Idxi = np.zeros((Segments**2,2))
    Idxj = np.zeros((Segments**2,2))
    
    count=-1
    for i in xrange(Segments):
        for j in xrange(Segments):
            count+=1
            Idxi[count,0] = idxi[0] + i*winsizei
            Idxi[count,1] = np.min( (Idxi[count, 0] + winsizei, idxi[1]) )
            Idxj[count,0] = idxj[0] + j*winsizej
            Idxj[count,1] = np.min( (Idxj[count, 0] + winsizej, idxj[1]) )

    Idxi = np.array(Idxi).astype(int)
    Idxj = np.array(Idxj).astype(int)  
    
    return Idxi, Idxj
    
def _get_mask(Size, idxi, idxj, wintype):
    '''Compute a mask of zeros with a window at a given position. 
    '''

    idxi = np.array(idxi).astype(int) 
    idxj =  np.array(idxj).astype(int)
    
    winsize = (idxi[1] - idxi[0] , idxj[1] - idxj[0])
    wind = build_2D_tapering_function(winsize, wintype)
    
    mask = np.zeros((Size, Size)) 
    mask[idxi.item(0):idxi.item(1), idxj.item(0):idxj.item(1)] = wind
    
    return mask
    