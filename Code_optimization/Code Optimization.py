
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pybind11
import numpy.linalg as la
from scipy.stats import multivariate_normal

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
