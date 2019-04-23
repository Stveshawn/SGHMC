# Examples of BLR and BLASSO using HMC and SGHMC
# based on boston housing price data (multivariate regression)


from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import numpy as np
import numpy.linalg as la
from sklearn.model_selection import KFold
import seaborn as sns

# utility function
rmse = lambda y, yhat: np.sqrt(np.mean((y-yhat)**2))

# load data and pre-process
data = datasets.load_boston()
X=data['data']
y=data['target']
nburnin = 500
nsample = 1000
niter = nburnin + nsample
X = scale(X)
X = np.c_[np.ones(len(X)), X]

##################
#### BLR_HMC #####
##################

def leapfrog(gradU, p, r, eps = 0.01, L = 100):
    """
    Using leapfrog to discretalize
    
    Args:
        gradU: gradient of potential energy (posterior)
        p: position (parameters)
        r: momentum (auxiliary)
        eps: stepsize
        L: # of steps
        M_i: inversion of preconditioned mass matrix
    """
    
    r = r - eps/2 * gradU(p)
    for i in range(L-1):
        p = p + eps * r
        r = r - eps * gradU(p)
    
    p = p + eps * r
    r = r - eps/2 * gradU(p)
    
    return p, r


def log_r(U, p0, r0, p, r):
    """
    log of acceptance ratio
    Mass matrix/ Mass matrix inversion assumed to be identity, so omitted.
    """
    return (U(p0) + 1/2*r0.dot(r0)) - (U(p0) + 1/2*r.dot(r))


def BLR_HMC(X, Y, nburnin=500, nsample=1000, eps=1e-4, L=100, prior='g-prior', MH=False, gradU=None):
    """
    Run Bayesian Linear Regression with a given prior using HMC samplar.
    
    Args:
        X, Y: training data
        niter: # of iterations = nburnin + nsample
        eps, L : stepsize and steps for differential discretization
        MH: whether do MH correction or not.
    """
    
    niter = nburnin + nsample
    n, d = X.shape
    if prior == 'g-prior':
        g, phi = n, 1 # suppose we know the precision for simplicity
        beta_ols = la.pinv(X) @ Y
        pos_mean = g/(1+g) * beta_ols
        pos_cov = g/(1+g) / phi * la.inv(X.T@X)
        gradU = lambda beta: la.inv(pos_cov) @ (beta - pos_mean) # gradient of U
        U = lambda beta: (beta-pos_mean).dot(la.inv(pos_cov)).dot(beta-pos_mean)/2
    else:
        print("Please specify the gradient for U based on your prior")
    
    pos_sample = np.zeros((niter+1,d))
    p = np.zeros(d) # initialize
    pos_sample[0,:] = p
    for k in range(niter):
        r0 = np.random.normal(0,1,d)
        p, r = leapfrog(gradU, p, r0, eps)
#         orbit[k+1,:] = p
        if MH:
            p0 = pos_sample[k]
            a = np.exp(log_r(U, p0, r0, p, r))
            u = np.random.rand()
            if u < a:
                pos_sample[k+1] = p
                ac+= 1
            else:
                pos_sample[k+1] = p0
        else:
            pos_sample[k+1] = p
        print("%.2f %%" % np.round((k+1)/niter*100,2), end = "\r")
    return pos_sample[nburnin+1:]


# run 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=2019)
kf.get_n_splits(X)
RMSE_blr = np.zeros(5)
i = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    samples1 = BLR_HMC(X_train,y_train)
    RMSE_blr[i] = rmse(X_test @ np.mean(samples1, 0), y_test)
    i += 1

# use the last fold to diagnostic plot
plt.figure(figsize=(15,8))
plt.subplot(221)
plt.plot(samples1[:,1])
plt.title(r"Trace plot of $\beta_1$")
plt.subplot(222)
plt.plot(samples1[:,2])
plt.title(r"Trace plot of $\beta_2$")
plt.subplots_adjust(top = 1)
plt.subplot(223)
sns.kdeplot(samples1[:,1])
plt.title(r"Density of $\beta_1$")
plt.subplot(224)
sns.kdeplot(samples1[:,2])
plt.title(r"Density of $\beta_2$")
plt.savefig('boston1.png');

#####################
#### BLASSO_HMC #####
#####################

def leapfrog_blasso(gradU, p, r, eps = 0.01, L = 100):
    """
    Using leapfrog to discretalize
    
    Args:
        gradU: gradient of potential energy (posterior)
        p: position (parameters)
        r: momentum (auxiliary)
        eps: stepsize
        L: # of steps
    """
    
    r = r - eps/2 * gradU(p, X, y)
    for i in range(L-1):
        p = p + eps * r
        r = r - eps * gradU(p, X, y)
    
    p = p + eps * r
    r = r - eps/2 * gradU(p, X, y)
    
    return p, r

def BLASOO_HMC(X, Y, nburnin=500, nsample=1000, eps=1e-4, L=100, MH=False, phi = 1, lam = 50):
    """
    Run Bayesian LASSO with a given prior using HMC samplar (Laplace prior).
    
    Args:
        X, Y: training data
        niter: # of iterations = nburnin + nsample
        eps, L : stepsize and steps for differential discretization
        MH: whether do MH correction or not.
        phi: prespecified precision. lam: prespecified penalty param
    """
    
    niter = nburnin + nsample
    n, d = X.shape
    gradU = lambda beta, X, y: - X.T@(y-X@beta)*phi + lam * np.sign(beta)
    
    pos_sample = np.zeros((niter+1,d))
    p = np.zeros(d) # initialize
    pos_sample[0,:] = p
    for k in range(niter):
        r0 = np.random.normal(0,1,d)
        p, r = leapfrog_blasso(gradU, p, r0, eps)
        
        if MH:
            p0 = pos_sample[k]
            a = np.exp(log_r(U, p0, r0, p, r))
            u = np.random.rand()
            if u < a:
                pos_sample[k+1] = p
                ac+= 1
            else:
                pos_sample[k+1] = p0
        else:
            pos_sample[k+1] = p
        print("%.2f %%" % np.round((k+1)/niter*100,2), end = "\r")
    return pos_sample[nburnin+1:]

kf = KFold(n_splits=5, shuffle=True, random_state=2019)
kf.get_n_splits(X)
RMSE_blasso = np.zeros(5)
i = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    samples2 = BLASOO_HMC(X_train, y_train)
    RMSE_blasso[i] = rmse(X_test @ np.mean(samples2, 0), y_test)
    i += 1
    
plt.figure(figsize=(15,8))
plt.subplot(221)
plt.plot(samples2[:,1])
plt.title(r"Trace plot of $\beta_1$")
plt.subplot(222)
plt.plot(samples2[:,2])
plt.title(r"Trace plot of $\beta_2$")
plt.subplots_adjust(top = 1)
plt.subplot(223)
sns.kdeplot(samples2[:,1])
plt.title(r"Density of $\beta_1$")
plt.subplot(224)
sns.kdeplot(samples2[:,2])
plt.title(r"Density of $\beta_2$")
plt.savefig('boston2.png');


####################
#### BLR_SGHMC #####
####################

def SGHMC(X, y, gradU, p, r, alpha, eta, beta = 0, eps = 0.01, L = 100, nbatch=50):
    """
    SGHMC updater by equation (15) in SGHMC paper.
    
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
        idx = np.random.choice(len(X), nbatch)
        batch = X[idx]
        batch_y = y[idx]
        v = v - eta * gradU(p, batch, batch_y) - alpha * v + np.random.normal(0, np.sqrt(2*(alpha-beta)*eta), X.shape[1])
    return p, v

def BLR_SGHMC(X, Y, nburnin=500, nsample=1000,
              eps=1e-4, L=100, prior='g-prior',
              gradU=None, alpha = 0.001,
              beta = 0, eta = 1e-5):
    """
    Run Bayesian Linear Regression with a given prior using SGHMC samplar.
    
    Args:
        X, Y: training data
        niter: # of iterations = nburnin + nsample
        eps, L : stepsize and steps for differential discretization
        MH: whether do MH correction or not.
        updates follows equation (15) as in SGHMC paper
    """
    
    niter = nburnin + nsample
    n, d = X.shape
    if prior == 'g-prior':
        g, phi = n, 1 # suppose we know the precision for simplicity
        beta_ols = la.pinv(X) @ Y
        pos_mean = g/(1+g) * beta_ols
        pos_cov = g/(1+g) / phi * la.inv(X.T@X)
        gradU = lambda beta, batch, batchy: la.inv(pos_cov) @ (beta - pos_mean) # gradient of U
    else:
        print("Please specify the gradient for U based on your prior")
    
    pos_sample = np.zeros((niter+1,d))
    p = np.zeros(d) # initialize
    pos_sample[0,:] = p
    for k in range(niter):
        r0 = np.random.normal(0,1,d)
        p, r = SGHMC(X, Y, gradU, p, r0, alpha, eta, beta = 0, eps = eps, L = L)
        pos_sample[k+1] = p
        print("%.2f %%" % np.round((k+1)/niter*100,2), end = "\r")
    return pos_sample[nburnin+1:]


kf = KFold(n_splits=5, shuffle=True, random_state=2019)
kf.get_n_splits(X)
RMSE_blr_sghmc = np.zeros(5)
i = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    samples3 = BLR_SGHMC(X_train, y_train)
    RMSE_blr_sghmc[i] = rmse(X_test @ np.mean(samples3, 0), y_test)
    i += 1
    
    
plt.figure(figsize=(15,8))
plt.subplot(221)
plt.plot(samples3[:,1])
plt.title(r"Trace plot of $\beta_1$")
plt.subplot(222)
plt.plot(samples3[:,2])
plt.title(r"Trace plot of $\beta_2$")
plt.subplots_adjust(top = 1)
plt.subplot(223)
sns.kdeplot(samples3[:,1])
plt.title(r"Density of $\beta_1$")
plt.subplot(224)
sns.kdeplot(samples3[:,2])
plt.title(r"Density of $\beta_2$")
plt.savefig('boston3.png');


#######################
#### BLASSO_SGHMC #####
#######################

def BLASSO_SGHMC(X, Y, nburnin=500, nsample=1000,
              eps=1e-4, L=100, prior='g-prior',
              gradU=None, alpha = 0.001,
              beta = 0, eta = 1e-5, phi=1, lam=50):
    """
    Run Bayesian LASSO using SGHMC samplar.
    
    Args:
        X, Y: training data
        niter: # of iterations = nburnin + nsample
        eps, L : stepsize and steps for differential discretization
        MH: whether do MH correction or not.
        updates follows equation (15) as in SGHMC paper
    """
    
    niter = nburnin + nsample
    n, d = X.shape
    gradU = lambda beta, batch, batchy: - batch.T@(batchy-batch@beta)*phi *len(X)/len(batch) + lam * np.sign(beta)
    
    pos_sample = np.zeros((niter+1,d))
    p = np.zeros(d) # initialize
    pos_sample[0,:] = p
    for k in range(niter):
        r0 = np.random.normal(0,1,d)
        p, r = SGHMC(X, Y, gradU, p, r0, alpha, eta, beta = 0, eps = eps, L = L)
        pos_sample[k+1] = p
        print("%.2f %%" % np.round((k+1)/niter*100,2), end = "\r")
    return pos_sample[nburnin+1:]


kf = KFold(n_splits=5, shuffle=True, random_state=2019)
kf.get_n_splits(X)
RMSE_blasso_sghmc = np.zeros(5)
i = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    samples4 = BLASSO_SGHMC(X_train, y_train)
    RMSE_blasso_sghmc[i] = rmse(X_test @ np.mean(samples4, 0), y_test)
    i += 1
    
    
plt.figure(figsize=(15,8))
plt.subplot(221)
plt.plot(samples4[:,1])
plt.title(r"Trace plot of $\beta_1$")
plt.subplot(222)
plt.plot(samples4[:,2])
plt.title(r"Trace plot of $\beta_2$")
plt.subplots_adjust(top = 1)
plt.subplot(223)
sns.kdeplot(samples4[:,1])
plt.title(r"Density of $\beta_1$")
plt.subplot(224)
sns.kdeplot(samples4[:,2])
plt.title(r"Density of $\beta_2$")
plt.savefig('boston4.png');


##### comparing RMSE
print(np.mean(RMSE_ols))
print(np.mean(RMSE_blr))
print(np.mean(RMSE_blasso))
print(np.mean(RMSE_blr_sghmc))
print(np.mean(RMSE_blasso_sghmc))
