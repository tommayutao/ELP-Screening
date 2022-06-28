import numpy as np
import GS_cpp
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
from sklearn.gaussian_process.kernels import GenericKernelMixin,NormalizedKernelMixin
from sklearn.base import clone
import os

class GS_kernel(GenericKernelMixin,NormalizedKernelMixin,Kernel):
	'''
	Generic String Kernel
	'''
	def __init__(self,amino_acid_property,L = 2,sigma_p=1.0,sigma_c=1.0,length_scale_bounds=(1e-5, 1e5)):
		self.sigma_p = sigma_p
		self.sigma_c = sigma_c
		self.L = L
		self.amino_acid_property = amino_acid_property
		self.E_matrix_ = self._compute_E_matrix()
		self.length_scale_bounds = length_scale_bounds

	def _compute_E_matrix(self):
		# Read the file
		f = open(os.path.expandvars(self.amino_acid_property), 'r')
		lines = f.readlines()
		f.close()

		amino_acids = []
		nb_descriptor = len(lines[0].split()) - 1
		aa_descriptors = np.zeros((len(lines), nb_descriptor))
		
		# Read descriptors
		for i in range(len(lines)):
			s = lines[i].split()
			aa_descriptors[i] = np.array([float(x) for x in s[1:]])
			amino_acids.append(s[0])

		# If nb_descriptor == 1, then all normalized aa_descriptors will be 1
		if nb_descriptor > 1:
			# Normalize each amino acid feature vector
			for i in range(len(aa_descriptors)):
				aa_descriptors[i] /= np.linalg.norm(aa_descriptors[i])

		E_mat = np.zeros((128,128))
		for i in range(len(amino_acids)):
			for j in range(i):
				aa_i,aa_j = amino_acids[i],amino_acids[j]
				E_mat[ord(aa_i),ord(aa_j)] = np.sum((aa_descriptors[i]-aa_descriptors[j])**2)
				E_mat[ord(aa_j),ord(aa_i)] = E_mat[ord(aa_i),ord(aa_j)]
		return E_mat	

	def is_stationary(self):
		return False

	@property
	def hyperparameter_L(self):
		return Hyperparameter("L","numeric","fixed")

	@property
	def hyperparameter_sigma_p(self):
		return Hyperparameter("sigma_p","numeric",self.length_scale_bounds)

	@property
	def hyperparameter_sigma_c(self):
		return Hyperparameter("sigma_c","numeric",self.length_scale_bounds)

	def _normalize_gradient(self,K,dK):
		'''
		Evaluate the gradient of normalized gram matrix
		'''
		v,w = np.sqrt(np.diag(K)), np.diag(dK)
		part1 = np.outer(v,v)*dK
		part2 = K*(np.outer(v,1./v)*w + np.outer(1./v,v)*(w[:,np.newaxis]))
		return (part1 - 0.5*part2)/(np.outer(v,v)**2)	

	def __call__(self, X, Y=None, eval_gradient=False):
		''' Evaluate the kernel and optionally its gradient

		Parameters
		-----------
		X : List of strings.Left argument of the returned kernel k(X, Y).

		Y : List of string. Right argument of the returned kernel k(X, Y). Default = None.
			If None, compute K(X,X).

		eval_gradient : bool, default=False
						Determines whether the gradient with respect to the kernel
						hyperparameter is determined. Only supported when Y is None
		-----------

		Returns
		-----------
		K : ndarray of shape (len(X), len(Y))

		K_gradient : ndarray of shape (len(X), len(X), n_dims)
			The gradient of the kernel k(X, X) with respect to the
			hyperparameter of the kernel. Only returned when eval_gradient
			is True.
		-----------
		'''	
		#X = list(X)	
		if Y is not None:
			#Y = list(Y)
			if eval_gradient:
				raise ValueError("Gradient can only be evaluated when Y is None.")
			else:
				K,_,_ = GS_cpp.compute_gram_matrix(X,Y,self.E_matrix_,self.L,self.sigma_p,self.sigma_c,False)
				diag_X = GS_cpp.compute_diagonal(X,self.E_matrix_,self.L,self.sigma_p,self.sigma_c)
				diag_Y = GS_cpp.compute_diagonal(Y,self.E_matrix_,self.L,self.sigma_p,self.sigma_c)
				return K/np.outer(np.sqrt(diag_X),np.sqrt(diag_Y))
		else:
			if eval_gradient:
				K,dK_dsp,dK_dsc = GS_cpp.compute_gram_matrix(X,X,self.E_matrix_,self.L,self.sigma_p,self.sigma_c,True)
				dK_dsp = self._normalize_gradient(K,dK_dsp)
				dK_dsc = self._normalize_gradient(K,dK_dsc) 
				diag_X = np.sqrt(np.diagonal(K))
				K = K/(np.outer(diag_X,diag_X))
				dK_dL = np.empty((K.shape[0],K.shape[1],0))
				return K,np.dstack((dK_dL,self.sigma_c*dK_dsc[:,:,np.newaxis],self.sigma_p*dK_dsp[:,:,np.newaxis]))
			else:
				K,_,_ = GS_cpp.compute_gram_matrix(X,X,self.E_matrix_,self.L,self.sigma_p,self.sigma_c,True)
				diag_X = np.sqrt(np.diagonal(K))
				K = K/(np.outer(diag_X,diag_X))
				return K

	def clone_with_theta(self, theta):
		cloned = clone(self)
		cloned.theta = theta
		return cloned

	def __repr__(self):
		return "{0}(L={1}, sigma_c={2:.3g},sigma_p={3:.3g})".format(
			self.__class__.__name__, self.L, self.sigma_c, self.sigma_p)



	

