<%
cfg["compiler_args"] = ["-std=c++11"]
cfg["include_dirs"] = ["../eigen"]
setup_pybind11(cfg)
%>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

double U(double mu, Eigen::VectorXd batch) {
    return mu*mu/2 + (mu-batch.array()).square().sum()/2;
}

double gradU(double mu, Eigen::VectorXd batch, int ndata) {
    return mu + (mu-batch.array()).sum() * ndata/ batch.size();
}

double Vhat(Eigen::VectorXd batch) {
    return (batch.array() - batch.mean()).square().sum()/(batch.size()-1);
}


PYBIND11_MODULE(SGHMC_utils, m) {
    m.doc() = "module to do calculate basic quantities for updating based on pybind11";
    m.def("U", &U, "Potential energy evaluated based one the whole dataset");
    m.def("gradU", &gradU, "estimated gradient of U based on minibatch");
    m.def("Vhat", &Vhat, "empirical Fishier Informatetion");
}
