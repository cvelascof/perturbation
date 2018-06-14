"""Methods for generating perturbations of two-dimensional motion fields."""

import numpy as np
from scipy import linalg

def initialize_motion_perturbations_bps(V, p_pert_par, p_pert_perp, pixelsperkm, 
                                        timestep, seed=None):
    """Initialize the motion field perturbator described in Bowler et al. 
    2006: STEPS: A probabilistic precipitation forecasting scheme which merges 
    an extrapolation nowcast with downscaled NWP. For simplicity, the bias 
    adjustment procedure described in the above reference has not been 
    implemented. The perturbator generates a constant field whose magnitude 
    depends on lead time, see generate_motion_perturbations_bps.
    
    Parameters
    ----------
    V : array_like
      Array of shape (2,m,n) containing the x- and y-components of the m*n 
      motion field to perturb.
    p_pert_par : tuple
      Tuple containing the parameters a,b and c for the standard deviation of 
      the perturbations in the direction parallel to the motion vectors. The 
      standard deviations are modeled by the function f_par(t) = a*t^b+c, where 
      t is lead time.
    p_pert_perp : tuple
      Tuple containing the parameters a,b and c for the standard deviation of 
      the perturbations in the direction perpendicular to the motion vectors. 
      The standard deviations are modeled by the function f_par(t) = a*t^b+c, 
      where t is lead time.
    pixelsperkm : float
      Spatial resolution of the motion field (pixels/kilometer).
    timestep : float
      Time step for the motion vectors (minutes).
    seed : int
      Optional seed number for the random generator.
    
    Returns
    -------
    out : dict
      A dictionary containing the perturbator that can be supplied to 
      generate_motion_perturbations_bps.
    """
    if len(V.shape) != 3:
        raise ValueError("V is not a three-dimensional array")
    if V.shape[0] != 2:
        raise ValueError("the first dimension of V is not 2")
    if len(p_pert_par) != 3:
        raise ValueError("the length of p_pert_par is not 3")
    if len(p_pert_perp) != 3:
        raise ValueError("the length of p_pert_perp is not 3")
    
    perturbator = {}
    
    np.random.seed(seed)
    
    v_pert_x = np.random.laplace()
    v_pert_y = np.random.laplace()
    V_pert = np.stack([v_pert_x*np.ones(V.shape[1:3]), 
                       v_pert_y*np.ones(V.shape[1:3])])
    
    # scale factor for converting the unit of the advection velocities into km/h
    vsf = 60.0 / (timestep * pixelsperkm)
    V = V * vsf
    
    N = linalg.norm(V, axis=0)
    V_n = V / np.stack([N, N])
    DP = np.sum(V_pert*V_n, axis=0)
    
    perturbator["vsf"]    = vsf
    perturbator["p_par"]  = p_pert_par
    perturbator["p_perp"] = p_pert_perp
    V_pert_par = V_n * np.stack([DP, DP])
    perturbator["V_pert_par"]  = V_pert_par
    perturbator["V_pert_perp"] = V_pert - V_pert_par
    
    return perturbator

def generate_motion_perturbations_bps(perturbator, t):
    """Generate a motion perturbation field by using the method described in 
    Bowler et al. 2006: STEPS: A probabilistic precipitation forecasting scheme 
    which merges an extrapolation nowcast with downscaled NWP.
    
    Parameters
    ----------
    perturbator : dict
      A dictionary returned by initialize_motion_perturbations_bps.
    t : float
      Lead time for the perturbation field (minutes).
    
    Returns
    -------
    out : ndarray
      Array of shape (2,m,n) containing the x- and y-components of the motion 
      vector perturbations, where m and n are determined from the perturbator.
    """
    vsf         = perturbator["vsf"]
    p_par       = perturbator["p_par"]
    p_perp      = perturbator["p_perp"]
    V_pert_par  = perturbator["V_pert_par"]
    V_pert_perp = perturbator["V_pert_perp"]
    
    g_par  = p_par[0]  * pow(t, p_par[1])  + p_par[2]
    g_perp = p_perp[0] * pow(t, p_perp[1]) + p_perp[2]
    
    return (g_par*V_pert_par + g_perp*V_pert_perp) / vsf
