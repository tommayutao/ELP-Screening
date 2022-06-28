from compute_dG import *
import pickle,os,copy,shutil
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import LeaveOneOut
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from GS_kernel import GS_kernel
from statistical_distances import *

def decode_string(s):
	# The shorthand for ELP is XmYn... = (VPGXG)m(VPGYG)n... #
	decoded = ''
	i = 0
	while i < len(s):
		if s[i].isalpha():
			j = i+1
			while j < len(s) and s[j].isnumeric():
				j += 1
			decoded += 'VPG%sG'%(s[i])*int(s[i+1:j])
			i = j
	return decoded

def encode_string(peptide):
	# Convert full name for ELP (VPGXG)m(VPGYG)n... into XmYn... #
	temp = []
	i = 0
	while i < len(peptide):
		cur = peptide[i:i+5] # VPGXG chunk
		temp.append(cur[-2])
		i += 5
	s = ''
	i = 0
	while i < len(temp):
		start_char = temp[i]
		count = 0
		while i < len(temp) and temp[i] == start_char:
			count += 1
			i += 1
		s += start_char + str(count)
	return s

def fit_gpr(X_train,Y_train,alpha_train,AA_matrix,L_grid,bounds):
	max_likelihood = -np.inf
	for l in L_grid:
		kernel = GS_kernel(AA_matrix, L = l,length_scale_bounds = bounds)
		gp = GaussianProcessRegressor(kernel = kernel, normalize_y = True, alpha = alpha_train/np.var(Y_train))
		gp.fit(X_train,Y_train)
		likelihood = gp.log_marginal_likelihood_value_
		if likelihood > max_likelihood:
			max_likelihood = likelihood
			cur_best_model = gp
	return cur_best_model

def expected_improvement(sequence,mu_sample_opt,gpr,xi=0.01):
	mu, sigma = gpr.predict([sequence], return_std=True)
	if np.isclose(sigma,0):
		return 0
	imp = mu_sample_opt-xi-mu
	z = imp/sigma
	ei = (mu-mu_sample_opt+xi)*norm.cdf(z)-sigma*norm.pdf(z)
	return ei

AA_matrix = 'amino_acids_matrix/AA.blosum62.dat'
with open('library.pickle','rb') as f:
	temp = pickle.load(f)
	candidates = temp['candidates']

## Backup files ##
shutil.copyfile('BO_samples.pickle','BO_samples_backup.pickle')
shutil.copyfile('optimization_history.pickle','optimization_history_backup.pickle')

## Load sampled sequences ##
cur_X,cur_Y,cur_error = [],[],[]
visited = set()
min_len = np.inf
max_len = -np.inf

with open("BO_samples.pickle",'rb') as f:
	history = pickle.load(f)
	shorthand,dG,dG_error = history[-1][-1]
	if dG is None:
		# This means it is a new sample. We need to evaluate its dG per residue #
		peptide = decode_string(shorthand)
		dG,dG_error = block_analysis_proj(shorthand)
		dG += translational_entropy_correction(shorthand)
	history[-1][-1] = (shorthand,dG,dG_error)
	for (shorthand,dG,dG_error) in history[-1]:
		peptide = decode_string(shorthand)
		cur_X.append(peptide)
		visited.add(peptide)
		min_len = min(min_len,len(peptide))
		max_len = max(max_len,len(peptide))
		cur_Y.append(dG)
		cur_error.append(dG_error)
X_array,Y_array,alpha_array = np.array(cur_X),np.array(cur_Y),np.array(cur_error)**2
shuffle_idx = np.random.permutation(len(X_array))
X_array,Y_array,alpha_array = X_array[shuffle_idx],Y_array[shuffle_idx],alpha_array[shuffle_idx]

## Fit gp to current data ##
L_grid = np.arange(1,min_len+1)
bounds = (1e-3,max_len)
loo = LeaveOneOut()
y_true,y_pred = [],[]
for train,test in loo.split(X_array):
	X_train,X_test = X_array[train],X_array[test]
	Y_train,Y_test = Y_array[train],Y_array[test][0]
	alpha_train = alpha_array[train]
	model = fit_gpr(X_train,Y_train,alpha_train,AA_matrix,L_grid,bounds)
	y_true.append(Y_test)
	y_pred.append(model.predict(X_test)[0])
mean_score = r2_score(y_true,y_pred)
best_gp = fit_gpr(X_array,Y_array,alpha_array,AA_matrix,L_grid,bounds)
print ('Finish fitting GPR model with score ', mean_score)
mu_samples = best_gp.predict(X_array)
mu_sample_opt = np.amin(mu_samples)

## Seach for next sequence ##
min_f = np.inf
next_point = None
for sequence in candidates:
	if sequence in visited:
		continue
	ei = expected_improvement(sequence,mu_sample_opt,best_gp,xi=0.01)
	if ei < min_f:
		min_f = ei
		next_point = sequence

## Save the updated traning examples and GPR model ##
cur_X.append(next_point)
cur_Y.append(None)
cur_error.append(None)
updated = []
for i in range(len(cur_X)):
	updated.append((encode_string(cur_X[i]),cur_Y[i],cur_error[i]))
history.append(updated)
with open('BO_samples.pickle','wb') as f:
	pickle.dump(history,f)

if not os.path.exists('optimization_history.pickle'):
	with open('optimization_history.pickle','wb') as f:
		pickle.dump([{'gpr_model':best_gp,'R2':mean_score,'iteration':1,'stat_distance':[]}],f)
else:
	with open('optimization_history.pickle','rb') as f:
		temp = pickle.load(f)
	stat_dist = copy.deepcopy(temp[-1]['stat_distance'])
	cur_dict = dict()
	cur_dict['gpr_model'] = best_gp
	cur_dict['R2'] = mean_score
	cur_dict['iteration'] = temp[-1]['iteration'] + 1
	temp.append(cur_dict)

	## Evaluate statistical distances between gaussian posteriors
	gpr1 = temp[-1]['gpr_model']
	gpr2 = temp[-2]['gpr_model']
	mu1,cov1 = gpr1.predict(np.array(candidates),return_cov = True)
	cov1 += np.eye(len(mu1))*1e-5
	mu2,cov2 = gpr2.predict(np.array(candidates),return_cov = True)
	cov2 += np.eye(len(mu2))*1e-5
	stat_dist.append(bhattacharyya_distance(mu1,mu2,cov1,cov2))
	temp[-1]['stat_distance'] = stat_dist
	with open('optimization_history.pickle','wb') as f:
		pickle.dump(temp,f)