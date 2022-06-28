import numpy as np

def bhattacharyya_distance(mean1,mean2,cov1,cov2):

	cov = 0.5*(cov1 + cov2)
	T1 = (1 / 8) * (
		np.dot(mean1-mean2,np.linalg.solve(cov,mean1-mean2))
	)

	sign_cov, logdet_cov = np.linalg.slogdet(cov)
	assert sign_cov == 1
	sign_cov1, logdet_cov1 = np.linalg.slogdet(cov1)
	assert sign_cov1 == 1
	sign_cov2, logdet_cov2 = np.linalg.slogdet(cov2)
	assert sign_cov2 == 1


	T2 = (1 / 2) * (logdet_cov - 0.5*(logdet_cov1+logdet_cov2))
	return T1 + T2

def KL_divergence(mean1,cov1,mean2,cov2):
	dmu = mean1-mean2
	sign_cov1, logdet_cov1 = np.linalg.slogdet(cov1)
	assert sign_cov1 == 1
	sign_cov2, logdet_cov2 = np.linalg.slogdet(cov2)
	assert sign_cov2 == 1

	T1 = logdet_cov2 - logdet_cov1 - len(mean1)
	T2 = np.dot(dmu,np.linalg.solve(cov2,dmu))
	T3 = np.trace(np.linalg.solve(cov2,cov1))

	return 0.5*(T1+T2+T3) 


