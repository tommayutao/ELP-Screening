#include <vector>
#include <map>
#include <eigen3/Eigen/Dense>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace py = pybind11;
std::tuple<double,double,double> evaluate_kernel(const std::string& x, const std::string& y,py::EigenDRef<Eigen::MatrixXd> E_matrix,size_t L, double sigma_p, double sigma_c);
std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> compute_gram_matrix(const std::vector<std::string>& X, const std::vector<std::string>& Y, py::EigenDRef<Eigen::MatrixXd> E_matrix,size_t L, double sigma_p, double sigma_c, bool symmetric);
Eigen::VectorXd compute_diagonal(const std::vector<std::string>& X,py::EigenDRef<Eigen::MatrixXd> E_matrix,size_t L, double sigma_p, double sigma_c);



