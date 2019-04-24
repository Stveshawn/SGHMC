
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pybind11
import numpy.linalg as la
from scipy.stats import multivariate_normal

###################
## vectorization ##
###################

n = int(1e5)
nbatch= 500
data = np.random.normal(1, size = n)
idx = np.random.choice(len(data), nbatch)
batch = data[idx]


# 1. compare different ways of iterating over data.

# list comprehension
gradU_list = lambda mu, batch: mu + sum([mu-x for x in batch]) * len(data) / len(batch)

# for loop
gradUi = lambda mu, x: mu-x
def gradU_for(mu, batch):
    """
    Using forloop to calculate gradient.
    """
    
    gradU = 0
    for x in batch:
        gradU += gradUi(mu, x)
        
    gradU *= len(data) / len(batch)
    gradU += mu
    return gradU

# np.array vectorization
gradU_array = lambda mu, batch: mu + np.sum(mu-batch) * len(data) / len(batch)


# time comparison
ls = (10 ** np.linspace(2, 5, 50)).astype(int)
T = np.zeros((len(ls), 3, 100))
f_list = [gradU_for, gradU_list, gradU_array]

for i, nbatch in enumerate(ls) :
    idx = np.random.choice(len(data), nbatch)
    batch = data[idx]
    for j, f in enumerate(f_list):
        for k in range(100):
            start = time.time()
            f(1, batch)
            elapsed = time.time() - start
            T[i, j, k] = elapsed
    print((i+1)/len(ls), end='\r')
    
    
T_mean = T.mean(2)
T_sd = np.sqrt(((T-T.mean(2)[:,:,np.newaxis]) ** 2).mean(2))
T_log_mean = np.log(T).mean(2)

plt.figure(figsize=(16,4.5))
plt.subplot(121)
plt.plot(ls, T_mean[:,0], label = 'list comprehension')
plt.plot(ls, T_mean[:,1], label = 'for loop')
plt.plot(ls, ls*1e-6, label = 'linear')
plt.plot(ls, T_mean[:,2], label = 'numpy array vectorization')
plt.legend()
plt.title('Runtime by mini-batch size')
plt.subplot(122)
plt.plot(np.log10(ls), T_log_mean[:,0], label = 'list comprehension')
plt.plot(np.log10(ls), T_log_mean[:,1], label = 'for loop')
plt.plot(np.log10(ls), np.log(ls*1e-6), label = 'linear')
plt.plot(np.log10(ls), T_log_mean[:,2], label = 'numpy array vectorization')
plt.title('Runtime by mini-batch size (log-log scale)')
plt.legend();
# plt.savefig('runtime1.png');

#################
### pybind 11 ###
#################


import cppimport
cppimport.force_rebuild()
SGHMC_utils=cppimport.imp("SGHMC_utils")


U_array = lambda mu, batch: mu**2/2 + np.sum((mu-batch)**2/2)
gradU_array = lambda mu, batch, ndata: mu + np.sum(mu-batch) * ndata / len(batch)
Vhat_array = lambda batch: np.cov(batch)

print(np.isclose(gradU_array(1, batch, len(data)), SGHMC_utils.gradU(1, batch, len(data))))
print(np.isclose(U_array(1, batch), SGHMC_utils.U(1, batch)))
print(np.isclose(Vhat_array(batch), SGHMC_utils.Vhat(batch)))


def SGHMC_update(Vhat, gradU, p, r, nbatch = 50, eps = 0.01, L = 100, M_i = 1):
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
        idx = np.random.choice(len(data), nbatch)
        batch = data[idx]
        V = Vhat(batch)
        B = 1/2 * eps * V
        C = 3
        r = r - eps*gradU(p, batch, len(data)) - eps*C*M_i*r + np.random.normal(0, np.sqrt(2*(C-B)*eps))
    return p, r


p, r0 = 0, 0


data = np.random.normal(1, size = int(1e5))
ls = (10 ** np.linspace(2, 5, 10)).astype(int)
T2 = np.zeros((len(ls), 100, 2))

for i, nbatch in enumerate(ls):
    for j in range(100):
        t1 = time.time()
        SGHMC_update(Vhat, gradU_array, p, r0, nbatch, eps = 0.01, L = 100, M_i = 1)
        t2 = time.time()
        SGHMC_update(SGHMC_utils.Vhat, SGHMC_utils.gradU, p, r0, eps = 0.01, L = 100, M_i = 1)
        t3 = time.time()
        T2[i, j, 0] = t2 - t1
        T2[i, j, 1] = t3 - t2
    print((i+1)/len(ls), end='\r')
    
    
Tpy = T2.mean(1)[:,0]
Tc = T2.mean(1)[:,1]
print(Tpy)
print(Tc)

import pandas as pd
T2l = np.log10(T2)

df1 = pd.melt(pd.DataFrame(T2l[:,:,0].T, columns=ls), col_level=0)
df2 = pd.melt(pd.DataFrame(T2l[:,:,1].T, columns=ls), col_level=0)

plt.figure(figsize=(16,9))
sns.boxplot(y="value", x= "variable", data = df1, palette = sns.color_palette("Blues", n_colors = 10))
sns.boxplot(y="value", x= "variable", data = df2, palette = sns.color_palette("Greens", n_colors = 10))
plt.xlabel('batch size')
plt.ylabel('log-avg. runtime')
plt.title("Runtime by batch size (naive Python vs pybind11)");
# plt.savefig("py_vs_cpp.png");
