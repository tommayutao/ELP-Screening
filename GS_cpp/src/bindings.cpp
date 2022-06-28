#include "compute_kernel.h"
namespace py = pybind11;

PYBIND11_MODULE(GS_cpp, m){
	m.doc() = "C++ code to evaluate GS gram matrix and its gradients";
	m.def("compute_gram_matrix",&compute_gram_matrix);
	m.def("compute_diagonal",&compute_diagonal);
}