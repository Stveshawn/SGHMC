# SGHMC
Python implementation and optimization of [Stochastic Gradient Hamilton Monte Carlo](https://arxiv.org/pdf/1402.4102.pdf). 

_This is the repository for the final project of STA 663 at Duke University, Durham, NC._ By Steve Shao and Freya Fu.

# Content

+ Implementation: implement the algorithm described by the paper along with relevant algorithms like HMC and SGLD.

+ Code optimization: optimize codes' performance by using vectorization and `C++`-coded critical functions (wrapped for Python with `pybind11`), etc.

+ Simulation: simulate data in univariate and bivariate normal case, and check the performance of SGHMC. Plus: comparison simulation as in the original paper.

+ Real data analysis: Bayesian inference (BLR, BLASSO) example based on `Boston housing price` data using HMC and SGHMC samplers.

# Note:

The `C++` code depends on __Eigen__ library. To get everything to work out, please make sure you have eigen in the correct directory or you can download it by running the following line in terminal:

`! git clone https://github.com/RLovelett/eigen.git`
