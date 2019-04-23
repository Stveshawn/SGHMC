
import numpy as np
import scipy.linalg as la
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from plot_utils import *

#############################
### 1. Univariate normal  ###
#############################

np.random.seed(2019)
data = np.random.normal(1, size = 10000)

sig2_pos = 1/(1/1 + len(data) / np.cov(data))
mean_pos = (0 + xx.mean()*len(data)/np.cov(data))/(1/1 +len(data) / np.cov(data))
dist = multivariate_normal(mean_pos, (sig2_pos))
sim = dist.rvs(1000)


# Global parameters
nburnin = 100
nsample = 1000
niter = nburnin + nsample


# U = lambda mu: mu**2/2 + sum([(x-mu)**2/2 for x in xx])
U = lambda mu: mu**2/2 + np.sum((data-mu)**2/2)
gradU = lambda mu, batch: mu + np.sum(mu-batch) * len(data) / len(batch)
Vhat = lambda mu, batch: np.cov(mu-batch)



def SGHMC_update_1d(Vhat, gradU, p, r, eps = 0.01, L = 100, M_i = 1):
    """
    Using leapfrog to discretalize
    
    Args:
        Vhat: empirical fisher info matrix
        gradU: gradient of potential energy (posterior)
        p: position (parameters)
        r: momentum (auxiliary)
        eps: stepsize
        L: # of steps
        M_i: inversion of preconditioned mass matrix
    """


    for i in range(L):
        p = p + eps*M_i * r
        idx = np.random.choice(len(xx), nbatch)
        batch = xx[idx]
        V = Vhat(p, batch)
        B = 1/2 * eps * V
        C = 3
        r = r - eps*gradU(p, batch) - eps*C*M_i*r + np.random.normal(0, np.sqrt(2*(C-B)*eps))
    return p, r



niter = 1100
eps = 0.0001
L = 100
nbatch= 500
np.random.seed(2019)
samples = np.zeros(niter+1)
p = np.array([0.0])
samples[0] = p
for k in range(niter):
    r0 = np.random.normal(0,1)
    p, r = SGHMC_update2(Vhat, gradU, p, r0, eps, L)

    samples[k+1] = p
    print("%.2f %%" % np.round((k+1)/niter*100,2), end = "\r")
    
    
plt.figure(figsize=(10,6))
sns.kdeplot(samples[100:], label = 'Samples with SGHMC')
sns.kdeplot(sim, label = 'Samples with HMC')
plt.title("SGHMC (univariate normal)");
# plt.savefig('SGHMC_1d.png');



#############################
### 2. Bivariate normal #####
#############################

# True parameters for generating data X (sampling distribution)
mean_or = np.array([1,-1])
sig_or = np.array([[1,0.75],[0.75,1]])
sig_or_i = la.inv(sig_or)

# Sampling distribution for generating X
np.random.seed(2019)
dist = multivariate_normal(mean_or, sig_or)
data = dist.rvs(10000)


# Theoretical posterior parameters and distribution
sig_pos = la.inv(len(data)*la.inv(np.cov(data.T)) + np.eye(2)) 
mean_pos = (la.inv(len(data)*la.inv(np.cov(data.T)) + np.eye(2)) @
            (len(data)*la.inv(np.cov(data.T))@np.mean(data,0) + np.eye(2)@np.zeros(2)))
post = multivariate_normal(mean_pos, sig_pos) # true posterior distribution
np.random.seed(2019)
sim = post.rvs(1000) # simulate 1000 points from posterior

print("True posterior mean:\n%s\n True posterior covariance:\n%s" % (mean_pos,Sig_pos))


#############################
### 2.1 sampling scheme 1 ###
#############################

gradU = lambda mu, batch: mu - sig_or_i.dot((batch-mu).T).sum(1) / len(batch) * len(data)
Vhat = lambda mu, batch: np.cov(sig_or_i.dot((batch-mu).T))

def SGHMC_2d1(gradU, p, r, eps = 0.01, L = 100, M_i = np.eye(2), C = np.eye(2)):
    """
    Stochastic Gradient HMC for n-d setting
    
    Args:
        gradU: gradient of potential energy (posterior)
        p: position (parameters)
        r: momentum (auxiliary)
        eps: stepsize
        L: # of steps
        M_i: inversion of preconditioned mass matrix
    """


    for i in range(L):
        p = p + eps*M_i @ r
        idx = np.random.choice(len(data), nbatch)
        batch = data[idx]
        V = Vhat(p, batch)
        B = 1/2 * eps * V
#         rr = multivariate_normal(np.zeros(len(p)), (2*(C-B)*eps))
        rr = np.random.normal(0, np.sqrt((2*(C-B)*eps)[0,0]), 2)
        r = r - eps*gradU(p,batch) - eps*C@M_i@r + rr
    return p, r


niter = 1100
eps = 0.00005 # stepsize
L = 100 # steps
nbatch = 500
# M_i = np.array([[1,0],[0,1]]) # M is identity matrix by default
np.random.seed(1)
samples1 = np.zeros((niter, 2))
p = np.array([0,0.0])
for k in range(niter):
    r0 = np.random.normal(0,1,2)
    p, r = SGHMC_2d1(gradU, p, r0, eps, L)

    samples1[k] = p
    print("%.2f %%" % np.round((k+1)/niter*100,2), end = "\r")
    
    
print("mean of posterir samples:\n%s\n cov of posterior samples:\n%s" %
      (np.mean(samples1[nburnin:],0),np.cov(samples1[nburnin:].T)))


kde_stack(samples1[nburnin:, 0], samples1[nburnin:, 1],
          sim[:, 0], sim[:, 1],
          h=10, w=12, title= 'True posterior and samples with SGHMC',
          label=['Samples with SGHMC', 'Simulation from true posterior'])
# plt.savefig('SGHMC_2d1.png');


#############################
### 2.2 sampling scheme 2 ###
#############################


def SGHMC_2d2(gradU, p, r, alpha, eta, beta = 0, eps = 0.01, L = 100, nbatch=500):
    """
    Stochastic Gradient HMC for n-d setting
    
    Args:
        gradU: gradient of potential energy (posterior)
        p: position (parameters)
        r: momentum (auxiliary)
        eps: stepsize
        L: # of steps
        M_i: inversion of preconditioned mass matrix
    """

    v = eps * r
    for i in range(L):
        p += v
        idx = np.random.choice(len(data), nbatch)
        batch = data[idx]
#         V = Vhat(p, batch)
        v = v - eta * gradU(p, batch) - alpha * v + np.random.normal(0, np.sqrt(2*(alpha-beta)*eta), len(p))
    return p, v


eps = 0.0001
alpha = 0.000
beta = 0.00
eta = eps**2
L = 100
nbatch= 500

samples2 = np.zeros((niter, 2))
p = np.zeros(2)
for k in range(niter):
    r0 = np.random.normal(0,1,2)
    p, r = SGHMC_2d2(gradU, p, r0, alpha, eta, beta, eps, L)

    samples2[k] = p.copy()
    print("%.2f %%" % np.round((k+1)/niter*100,2), end = "\r")
    
    
print("mean of posterir samples:\n%s\n cov of posterior samples:\n%s" %
      (np.mean(samples2[nburnin+1:],0),np.cov(samples2[nburnin+1:].T)))


kde_stack(samples2[nburnin+1:, 0], samples2[nburnin+1:, 1],
          sim[:, 0], sim[:, 1],
          h=10, w=12, title= 'True posterior and samples with SGHMC',
          label=['Samples with SGHMC', 'Simulation from true posterior'])
plt.savefig('SGHMC_2d2.png');
