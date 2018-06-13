"""Interface and utility methods for different generators."""

import numpy as np

# Use the SciPy fft by default. If SciPy is not installed, fall back to the 
# numpy implementation.
try:
    import scipy.fftpack as fft
except ImportError:
    from numpy import fft
    
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
        A two-dimensional numpy array of correlated noise.
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
    
def initialize_nonparam_2d_fft_filter(X, tapering_function='flat-hanning'):
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
      
    Xref = X.copy()
    if tapering_function is not None:
        Xref -= Xref.min()
        tapering = build_2D_tapering_function(X.shape, tapering_function)
    F = fft.fft2(Xref*tapering)
    
    return np.abs(F)
    
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