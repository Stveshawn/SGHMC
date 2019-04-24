import numpy as np


#### HMC

def HMC_updater(U, gradU, p, eps, L, MH=True):
    """
    updater of HMC Sampler (leapfrog discretization)
    
    Args:
        U: potential energy function
        gradU: gradient of negative log target distribution given the whole dataset
        p: initial parameter to be sampled
        eps: stepsize
        L: # of steps
        M_i: inversion of preconditioned mass matrix
        mass ommited since assumed to be 1
    """

    def log_r(U, p0, r0, p, r):
        """log of acceptance ratio"""
        return (U(p0) + 1/2*r0.dot(r0)) - (U(p0) + 1/2*r.dot(r))
    
    if type(p) != "np.array":
        d = 1
        p0 = p
    else:
        d = len(p)
        p0 = p
        
    r = np.random.randn(d)
    r0 = r
    
    r = r - eps/2 * gradU(p)
    for i in range(L-1):
        p = p + eps * r
        r = r - eps * gradU(p)
    p = p + eps * r
    r = r - eps/2 * gradU(p)
    if MH:
        # M-H
        a = np.exp(log_r(U, p0, r0, p, r))
        u = np.random.rand()

        if u < a:
            return p
        else:
            return p0
    else:
        return p
    
    
def HMC(nburnin, nsample, U, gradU, p0, data, eps=0.01, L=100, MH=True):
    """
    Function to get posterior samples given parameters with HMC
    """
    
    p = p0
    n = nburnin + nsample
    if type(p) != "np.array":
        d = 1
    else:
        d = len(p)
    samples = np.zeros((n,d))
    
    for i in range(n):
        p = HMC_updater(U, gradU, p, eps, L, MH)
        samples[i] = p.copy()
        
    return samples[nburnin:]


#### SGHMC

def SGHMC_updater(gradU, p, batch, alpha, eta, L, V=0):
    """
    updater of Stochastic Gradient HMC Sampler
    
    Args:
        gradU: gradient of target distribution given a minibatch
        p: initial parameter to be sampled
        alpha: momentum decay
        eta: learning rate, = eps ** 2, details in paper eqn (15)
        V: estimated fisher info
        batch: minibatch
        mass ommited since assumed to be 1
    """
    
    if type(p) != "np.array":
        d = 1
    else:
        d = len(p)
    beta = V * eta / 2
    v = np.random.randn(d) * np.sqrt(eta);  # auxiliary variable, eps * r = v
    try:
        sigma = np.sqrt( 2 * eta * (alpha-beta) )
    except ValueError:
        print("Set a smaller eta.")
    
    for i in range(L):
        v = v - gradU(p, batch) * eta - alpha * v + np.random.rand(d) * sigma
        p = p + v
        
    return p

def SGHMC(nburnin, nsample, gradU, p0, batch, alpha=0.01, eta=0.0001, L=100, V=0):
    """
    Function to get posterior samples given parameters with SGHMC
    """
    
    p = p0
    n = nburnin + nsample
    if type(p) != "np.array":
        d = 1
    else:
        d = len(p)
    samples = np.zeros((n,d))
    
    for i in range(n):
        p = SGHMC_updater(gradU, p0, batch, alpha, eta, L, V)
        samples[i] = p.copy()
        
    return samples[nburnin:]


#### SGLD

def SGLD_updater(gradU, p, batch, eta, L, V):
    """
    updater of Stochastic Gradient Langevin Dynamics Sampler
    
    Args:
        gradU: gradient of target distribution given a minibatch
        p: initial parameter to be sampled
        eta: learning rate
        V: estimated fisher info
        batch: minibatch
        mass ommited since assumed to be 1
    """
    
    if type(p) != "np.array":
        d = 1
    else:
        d = len(p)
    beta = V * eta / 2
    try:
        sigma = np.sqrt( 2 * eta * (1-beta) )
    except ValueError:
        print("Set a smaller eta.")
    
    for i in range(L):
        p = p - gradU(p, batch) * eta + np.random.randn(d) * sigma
#         samples[i,:] = p
    
#     return samples
    return p

def SGLD(nburnin, nsample, gradU, p, batch, eta=0.0001, L=100, V=3):
    """
    Function to get posterior samples given parameters with SGLD
    """
    
    n = nburnin + nsample
    if type(p) != "np.array":
        d = 1
    else:
        d = len(p)
    samples = np.zeros((n,d))
    
    for i in range(n):
        p = SGLD_updater(gradU, p, batch, eta, L, V)
        samples[i] = p.copy()
        
    return samples[nburnin:]