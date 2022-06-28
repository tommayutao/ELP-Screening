#include "compute_kernel.h"
#include <iostream>
#include <algorithm>
#include <math.h>

namespace py = pybind11;

std::tuple<double,double,double> evaluate_kernel(const std::string& x, const std::string& y,
												py::EigenDRef<Eigen::MatrixXd> E_matrix ,size_t L, 
												double sigma_p, double sigma_c){
	size_t max_L = L;
	double kernel_val = 0.0;
	double dsigma_p = 0.0;
	double dsigma_c = 0.0;
	double tl,ml,B_ij,C_ij;
	int int_i,int_j;

	for (size_t i=0; i<x.length(); i++){
		for (size_t j=0; j<y.length();j++){
			max_L = std::min({L,x.length()-i,y.length()-j});
			tl = 1.0;
			ml = 0.0;
			B_ij = 0.0;
			C_ij = 0.0;
			for (size_t l=0; l<max_L;l++){
				//std::cout << x[i+l] <<" " << y[j+l] << E_matrix(int(x[i+l]),int(y[j+l])) <<std::endl;
				tl *= exp(-E_matrix(int(x[i+l]),int(y[j+l]))/(2*pow(sigma_c,2)));
				ml += E_matrix(int(x[i+l]),int(y[j+l]));
				B_ij += tl;
				C_ij += (tl*ml);
			}
			int_i = static_cast<int>(i);
			int_j = static_cast<int>(j);
			kernel_val += exp(-pow(int_i-int_j,2)/(2*pow(sigma_p,2)))*B_ij;
			dsigma_p += pow(sigma_p,-3)*pow(int_i-int_j,2)*exp(-pow(int_i-int_j,2)/(2*pow(sigma_p,2)))*B_ij;
			dsigma_c += exp(-pow(int_i-int_j,2)/(2*pow(sigma_p,2)))*pow(sigma_c,-3)*C_ij;
		}
	}
	return std::make_tuple(kernel_val,dsigma_p,dsigma_c);
}

std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> compute_gram_matrix(const std::vector<std::string>& X, const std::vector<std::string>& Y, 
																				py::EigenDRef<Eigen::MatrixXd> E_matrix,size_t L, 
																				double sigma_p, double sigma_c, bool symmetric){
	Eigen::MatrixXd K(X.size(),Y.size());
	Eigen::MatrixXd dsigma_p(X.size(),Y.size());
	Eigen::MatrixXd dsigma_c(X.size(),Y.size());

	if (symmetric){
		omp_set_num_threads(8);
		#pragma omp parallel for shared(K,dsigma_p,dsigma_c)
		for (size_t i=0; i < X.size();i++){
			for (size_t j=0; j <= i;j++){
				//s1 = py::cast<std::string>(X[i]);
				//s2 = py::cast<std::string>(Y[j]);
				std::string s1 = X.at(i);
				std::string s2 = Y.at(j);
				double temp_k,temp_dp,temp_dc;
				std::tie(temp_k,temp_dp,temp_dc) = evaluate_kernel(s1,s2,E_matrix,L,sigma_p,sigma_c);
				K(i,j) = temp_k;
				dsigma_p(i,j) = temp_dp;
				dsigma_c(i,j) = temp_dc;
				K(j,i) = K(i,j);
				dsigma_p(j,i) = dsigma_p(i,j);
				dsigma_c(j,i) = dsigma_c(i,j);
			}
		}
	}
	else{
		omp_set_num_threads(8);
		#pragma omp parallel for shared(K,dsigma_p,dsigma_c)
		for (size_t i=0; i < X.size();i++){
			//std::cout << "Thread ID: "<< omp_get_thread_num() << std::endl;
			for (size_t j=0; j < Y.size();j++){
				//s1 = py::cast<std::string>(X[i]);
				//s2 = py::cast<std::string>(Y[j]);
				std::string s1 = X.at(i);
				std::string s2 = Y.at(j);
				double temp_k,temp_dp,temp_dc;
				std::tie(temp_k,temp_dp,temp_dc) = evaluate_kernel(s1,s2,E_matrix,L,sigma_p,sigma_c);
				K(i,j) = temp_k;
				dsigma_p(i,j) = temp_dp;
				dsigma_c(i,j) = temp_dc;
			}
		}		
	}
	return std::make_tuple(K,dsigma_p,dsigma_c);
}

Eigen::VectorXd compute_diagonal(const std::vector<std::string>& X,py::EigenDRef<Eigen::MatrixXd> E_matrix,
								size_t L, double sigma_p, double sigma_c){
	Eigen::VectorXd K_diag(X.size());
	omp_set_num_threads(8);
	#pragma omp parallel for shared(K_diag)
	for (size_t i=0;i<X.size();i++){
		double temp_k,temp_dp,temp_dc;
		std::string s = X.at(i);
		std::tie(temp_k,temp_dp,temp_dc) = evaluate_kernel(s,s,E_matrix,L,sigma_p,sigma_c);
		K_diag(i) = temp_k;
	}
	return K_diag;
}