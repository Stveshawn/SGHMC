
from sghmc import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cppimport
from functools import partial
import time

cppimport.force_rebuild()
SGHMC_utils=cppimport.imp("SGHMC_utils")

gradU_cpp = partial(lambda x1, x2, x3: SGHMC_utils.gradU(x2, x3, x1), len(data))

np.random.seed(2019)
data = np.random.normal(1, size = 10000)
nbatch= 500
idx = np.random.choice(len(data), nbatch)
batch = data[idx] # we used the same batch since we are only testing time complexity

gradU2 = lambda mu: mu + np.sum(mu-data)
U = lambda mu: mu**2/2 + np.sum((mu-data)**2/2)
gradU = lambda mu, batch: mu + np.sum(mu-batch) * len(data) / len(batch)

samples_hmc = HMC(100, 1000, U, gradU2, 1, data, eps=0.01, L=100, MH=False)
samples_sghmc = SGHMC(100, 1000, gradU, 1, batch)
samples_sgld = SGLD(100, 1000, gradU, 1, batch)


# plt.plot(samples_sghmc,alpha=0.3)
# plt.plot(samples_hmc,alpha=0.3)
# plt.plot(samples_sgld,alpha=0.3)
