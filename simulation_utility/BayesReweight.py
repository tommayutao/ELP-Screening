"""

% Copyright:	Andrew L. Ferguson, UIUC 
% Last updated:	2 Jan 2016

% SYNOPSIS
%
% code to perform reweighting of dim_UMB-dimensional free energy surface inferred from Bayesian inference of biased umbrella sampling simulations in dim_UMB-dimensional variables, psi, into arbitrary dim_TRAJ-dimensional auxiliary variables, xi, recorded during umbrella sampling runs    
%
% 1. assumes 	(i)   harmonic restraining potentials in dim_UMB-dimensional umbrella variables psi 
%            	(ii)  all simulations conducted at same temperature, T
%            	(iii) rectilinear binning of histograms collected over each umbrella simulation  
% 2. requires 	(i)   trajectory of dim_UMB-dimensional umbrella variables, psi, recorded over each biased simulation, and biased histograms compiled over each biased run 
%				(ii)  maximum a posteriori (MAP) estimates of f_i = Z/Z_i = ratio of unbiased partition function to that of biased simulation i, for i=1..S biased simulations 
%				(iii) Metropolis-Hastings (MH) samples from the Bayes posterior of f_i = Z/Z_i = ratio of unbiased partition function to that of biased simulation i, for i=1..S biased simulations 
%				(iii) trajectory of dim_TRAJ-dimensional projection variables, xi, collected over each biased simulation at same time steps as the recorded umbrella variables 

% INPUTS
%
% T							- [float] temperature in Kelvin at which all umbrella sampling simulations were conducted  
% dim_UMB					- [int] dimensionality of umbrella sampling data in psi(1:dim_UMB) = number of coordinates in which umbrella sampling was conducted 
% periodicity_UMB			- [1 x dim_UMB bool] periodicity in each dimension across range of histogram bins 
%                             -> periodicity(i) == 0 => no periodicity in dimension i; periodicity(i) == 1 => periodicity in dimension i with period defined by range of histogram bins 
% harmonicBiasesFile_UMB	- [str] path to text file containing (1 + 2*dim_UMB) columns and S rows listing location and strength of biasing potentials in each of the S umbrella simulations 
%                             -> col 1 = umbrella simulation index 1..S, col 2:(2+dim_UMB-1) = biased simulation umbrella centers / (arbitrary units), col (2+dim_UMB):(2+2*dim_UMB-1) = harmonic restraining potential force constant / kJ/mol.(arbitrary units)^2 
% trajDir_UMB				- [str] path to directory holding i=1..S dim_UMB-dimensional trajectories in files traj_i.txt recording trajectory of dim_UMB-dimensional umbrella variables psi over each biased simulation 
%							  -> each file contains N_i rows constituting the number of samples recorded over the run, each containing dim_UMB columns recording the value of the umbrella variables psi(1:dim_UMB) 
%							  -> the file trajDir_UMB/traj_i.txt constitutes the raw data from which the histogram in histDir_UMB/hist_i.txt was constructed 
% histBinEdgesFile_UMB		- [str] path to text file containing dim_UMB rows specifying edges of the rectilinear histogram bins in each dimension used to construct dim_UMB-dimensional histograms held in histDir/hist_i.txt 
%							  -> histBinEdgesFile_UMB contains k=1..dim_UMB lines each holding a row vector specifying the rectilinear histogram bin edges in dimension k 
%                             -> for M_k bins in dimension k, there are (M_k+1) edges 
%                             -> bins need not be regularly spaced 
% histDir_UMB				- [str] path to directory holding i=1..S dim_UMB-dimensional histograms in files hist_i.txt compiled from S biased trajectories over dim_UMB-dimensional rectilinear histogram grid specified in histBinEdgesFile  
%                             -> hist_i.txt comprises a row vector containing product_k=1..dim_UMB M_k = (M_1*M_2*...*M_dim_UMB) values recording histogram counts in each bin of the rectilinear histogram 
%                             -> values recorded in row major order (last index changes fastest) 
%							  -> the file trajDir_UMB/traj_i.txt constitutes the raw data from which the histogram in histDir_UMB/hist_i.txt was constructed 
% fMAPFile_UMB				- [str] path to file containing MAP estimates of f_i = Z/Z_i = ratio of unbiased partition function to that of biased simulation i, for i=1..S biased simulations 
%							  -> S values stored as a row vector 
% fMHFile_UMB				- [str] path to file containing nSamples_MH Metropolis-Hastings samples from the Bayes posterior of f_i = Z/Z_i = ratio of unbiased partition function to that of biased simulation i, for i=1..S biased simulations 
%							  -> nSamples_MH rows each containing a S-element row vector 
% trajDir_PROJ				- [str] path to directory holding i=1..S dim_TRAJ-dimensional trajectories in files traj_i.txt recording trajectory of dim_TRAJ-dimensional projection variables xi recorded simultaneously with umbrella variables over each biased simulation 
%							  -> each file trajDir_PROJ/traj_i.txt contains N_i rows constituting the number of samples recorded over the run -- which must be collected simultaneoulsy with umbrella variables in trajDir_UMB/traj_i.txt -- each containing dim_TRAJ columns recording the value of the projection variables xi(1:dim_TRAJ) 
% histBinEdgesFile_PROJ		- [str] path to text file containing dim_PROJ rows specifying edges of the rectilinear histogram bins in each dimension that will be used within this code to construct dim_PROJ-dimensional histograms and infer the reweighted unbiased free energy projection into the projection variables 
%							  -> histBinEdgesFile_PROJ contains k=1..dim_PROJ lines each holding a row vector specifying the rectilinear histogram bin edges in dimension k 
%                             -> for M_k_PROJ bins in dimension k, there are (M_k_PROJ+1) edges 
%                             -> bins need not be regularly spaced 

% OUTPUTS
%
% hist_binCenters_PROJ.txt	- [dim x M_k_PROJ float] text file containing dim_PROJ rows specifying M_k_PROJ k=1..dim_PROJ centers of the rectilinear histogram bins in each dimension used to construct dim_PROJ-dimensional histograms constituting pdf_PROJ_MAP.txt/pdf_PROJ_MH.txt and betaF_PROJ_MAP.txt/betaF_PROJ_MH.txt 
% hist_binWidths_PROJ.txt	- [dim x M_k_PROJ float] text file containing dim_PROJ rows specifying M_k_PROJ k=1..dim_PROJ widths of the rectilinear histogram bins in each dimension used to construct dim_PROJ-dimensional histograms constituting pdf_PROJ_MAP.txt/pdf_PROJ_MH.txt and betaF_PROJ_MAP.txt/betaF_PROJ_MH.txt 
% p_PROJ_MAP.txt			- [1 x M_PROJ float] text file containing MAP estimate of unbiased probability distribution p_l_PROJ_MAP over l=1..M_PROJ bins of dim_PROJ-dimensional rectilinear histogram 
%							  -> values recorded in row major order (last index changes fastest) 
% pdf_PROJ_MAP.txt			- [1 x M_PROJ float] text file containing MAP estimate of unbiased probability density function pdf_l_PROJ_MAP over l=1..M_PROJ bins of dim_PROJ-dimensional rectilinear histogram 
%							  -> values recorded in row major order (last index changes fastest) 
% betaF_PROJ_MAP.txt 		- [1 x M_PROJ float] text file containing MAP estimate of unbiased free energy surface betaF_l_PROJ_MAP = -ln(p(psi)/binVolume) + const. over l=1..M_PROJ bins of dim_PROJ-dimensional rectilinear histogram  
%							  -> values recorded in row major order (last index changes fastest) 
% p_PROJ_MH.txt				- [nSamples_MH x M_PROJ float] text file containing nSamples_MH Metropolis-Hastings samples from the Bayes posterior of unbiased probability distribution p_l_PROJ_MAP over l=1..M_PROJ bins of dim_PROJ-dimensional rectilinear histogram 
%							  -> values recorded in row major order (last index changes fastest) 
% pdf_PROJ_MH.txt			- [nSamples_MH x M_PROJ float] text file containing nSamples_MH Metropolis-Hastings samples from the Bayes posterior of unbiased probability density function pdf_l_PROJ_MAP over l=1..M_PROJ bins of dim_PROJ-dimensional rectilinear histogram 
%							  -> values recorded in row major order (last index changes fastest) 
% betaF_PROJ_MH.txt			- [nSamples_MH x M_PROJ float] text file containing nSamples_MH Metropolis-Hastings samples from the Bayes posterior of unbiased free energy surface betaF_l_PROJ_MAP = -ln(p(psi)/binVolume) + const. over l=1..M_PROJ bins of dim_PROJ-dimensional rectilinear histogram  
%							  -> values recorded in row major order (last index changes fastest) 

"""


## imports
import os, re, sys, time, json
import random, math

import numpy as np
import numpy.matlib

## classes

## methods

# usage
def _usage():
	print ("USAGE: %s T dim_UMB periodicity_UMB harmonicBiasesFile_UMB trajDir_UMB histBinEdgesFile_UMB histDir_UMB fMAPFile_UMB fMHFile_UMB trajDir_PROJ histBinEdgesFile_PROJ" % sys.argv[0])
	print ("       T                         - [float] temperature in Kelvin at which all umbrella sampling simulations were conducted ")
	print ("       dim_UMB                   - [int] dimensionality of umbrella sampling data in psi(1:dim_UMB) = number of coordinates in which umbrella sampling was conducted ")
	print ("       periodicity_UMB           - [1 x dim_UMB bool] periodicity in each dimension across range of histogram bins ")
	print ("                                   -> periodicity(i) == 0 => no periodicity in dimension i; periodicity(i) == 1 => periodicity in dimension i with period defined by range of histogram bins ")
	print ("       harmonicBiasesFile_UMB    - [str] path to text file containing (1 + 2*dim_UMB) columns and S rows listing location and strength of biasing potentials in each of the S umbrella simulations ")
	print ("                                   -> col 1 = umbrella simulation index 1..S, col 2:(2+dim_UMB-1) = biased simulation umbrella centers / (arbitrary units), col (2+dim_UMB):(2+2*dim_UMB-1) = harmonic restraining potential force constant / kJ/mol.(arbitrary units)^2 ")
	print ("       trajDir_UMB               - [str] path to directory holding i=1..S dim_UMB-dimensional trajectories in files traj_i.txt recording trajectory of dim_UMB-dimensional umbrella variables psi over each biased simulation ")
	print ("                                   -> each file contains N_i rows constituting the number of samples recorded over the run, each containing dim_UMB columns recording the value of the umbrella variables psi(1:dim_UMB) ")
	print ("                                   -> the file trajDir_UMB/traj_i.txt constitutes the raw data from which the histogram in histDir_UMB/hist_i.txt was constructed ")
	print ("       histBinEdgesFile_UMB      - [str] path to text file containing dim_UMB rows specifying edges of the rectilinear histogram bins in each dimension used to construct dim_UMB-dimensional histograms held in histDir/hist_i.txt ")
	print ("                                   -> histBinEdgesFile_UMB contains k=1..dim_UMB lines each holding a row vector specifying the rectilinear histogram bin edges in dimension k ")
	print ("                                   -> for M_k bins in dimension k, there are (M_k+1) edges ")
	print ("                                   -> bins need not be regularly spaced ")
	print ("       histDir_UMB               - [str] path to directory holding i=1..S dim_UMB-dimensional histograms in files hist_i.txt compiled from S biased trajectories over dim_UMB-dimensional rectilinear histogram grid specified in histBinEdgesFile ")
	print ("                                   -> hist_i.txt comprises a row vector containing product_k=1..dim_UMB M_k = (M_1*M_2*...*M_dim_UMB) values recording histogram counts in each bin of the rectilinear histogram ")
	print ("                                   -> values recorded in row major order (last index changes fastest) ")
	print ("                                   -> the file trajDir_UMB/traj_i.txt constitutes the raw data from which the histogram in histDir_UMB/hist_i.txt was constructed ")
	print ("       fMAPFile_UMB              - [str] path to file containing MAP estimates of f_i = Z/Z_i = ratio of unbiased partition function to that of biased simulation i, for i=1..S biased simulations ")	
	print ("                                   -> S values stored as a row vector ")
	print ("       fMHFile_UMB               - [str] path to file containing nSamples_MH Metropolis-Hastings samples from the Bayes posterior of f_i = Z/Z_i = ratio of unbiased partition function to that of biased simulation i, for i=1..S biased simulations ")
	print ("                                   -> nSamples_MH rows each containing a S-element row vector ")
	print ("       trajDir_PROJ              - [str] path to directory holding i=1..S dim_TRAJ-dimensional trajectories in files traj_i.txt recording trajectory of dim_TRAJ-dimensional projection variables xi recorded simultaneously with umbrella variables over each biased simulation ")
	print ("                                   -> each file trajDir_PROJ/traj_i.txt contains N_i rows constituting the number of samples recorded over the run -- which must be collected simultaneoulsy with umbrella variables in trajDir_UMB/traj_i.txt -- each containing dim_TRAJ columns recording the value of the projection variables xi(1:dim_TRAJ) ")
	print ("       histBinEdgesFile_PROJ     - [str] path to text file containing dim_PROJ rows specifying edges of the rectilinear histogram bins in each dimension that will be used within this code to construct dim_PROJ-dimensional histograms and infer the reweighted unbiased free energy projection into the projection variables ")
	print ("                                   -> histBinEdgesFile_PROJ contains k=1..dim_PROJ lines each holding a row vector specifying the rectilinear histogram bin edges in dimension k ")
	print ("                                   -> for M_k_PROJ bins in dimension k, there are (M_k_PROJ+1) edges ")
	print ("                                   -> bins need not be regularly spaced ")

def load_json(file):
	f = open(file,'r')
	parsed_json = json.load(f)
	print (json.dumps(parsed_json,indent=3, separators=(',',':')))
	f.close()
	return parsed_json

def unique(a):
	""" return the list with duplicate elements removed """
	return list(set(a))

def intersect(a, b):
	""" return the intersection of two lists """
	return list(set(a) & set(b))

def union(a, b):
	""" return the union of two lists """
	return list(set(a) | set(b))

def ind2sub_RMO(sz,ind):
	
	if (ind < 0) | (ind > (np.prod(sz)-1)):
		print("\nERROR - Variable ind = %d < 0 or ind = %d > # elements in tensor of size sz = %d in ind2sub_RMO" % (ind,ind,np.prod(sz)))
		sys.exit(-1)
	
	sub = np.zeros(sz.shape[0], dtype=np.uint64)
	for ii in range(0,len(sz)-1):
		P = np.prod(sz[ii+1:])
		sub_ii = math.floor(float(ind)/float(P))
		sub[ii] = sub_ii
		ind = ind - sub_ii*P
	sub[-1] = ind
	
	return sub

def sub2ind_RMO(sz,sub):
	
	if sz.shape != sub.shape:
		print("\nERROR - Variables sub and sz in sub2ind_RMO are not commensurate")
		sys.exit(-1)
	
	for ii in range(0,len(sz)-1):
		if ( sub[ii] < 0 ) | ( sub[ii] >= sz[ii] ):
			print("\nERROR - sub[%d] = %d < 0 or sub[%d] = %d >= sz[%d] = %d in sub2ind_RMO"  % (ii,sub[ii],ii,sub[ii],ii,sz[ii]))
			sys.exit(-1)
	
	ind=0
	for ii in range(0,len(sz)-1):
		ind = ind + sub[ii]*np.prod(sz[ii+1:])
	ind = ind + sub[-1]

	return int(ind)

def binner(valVec,binE):
	
	# function to place a dim-dimensional vector valVec into a dim-dimensional histogram with bin edges defined by binE 
	
	# error checking
	dim = valVec.shape[0]
	if binE.shape[0] != dim:
		print("\nERROR - Dimensionality %d of row vector valVec != dimensionality %d of binE" % (valVec.shape[0],binE.shape[0]))
		sys.exit()
	
	# binning values in valVec
	sub = -np.ones(dim,dtype=np.uint64)
	for d in range(0,dim):
		if ( ( valVec[d] < binE[d][0] ) | ( valVec[d] >= binE[d][-1] ) ):
			sub = -np.ones(dim)		# if at least one value of valVec falls outside histogram bins defined by binE returning -1's vector
			return sub
		for i in range(0,len(binE[d])):
			if ( ( valVec[d] >= binE[d][i] ) & ( valVec[d] < binE[d][i+1] ) ):
				sub[d] = i
				break
	
	return sub


## main

# parameters
kB = 0.0083144621		# Boltzmann's constant / kJ/mol.K

# loading inputs
param_file = sys.argv[1]
parameters = load_json(param_file)
# - reading args
T = parameters['T']
dim_UMB = parameters['dim_UMB']
periodicity_UMB = parameters['periodicity_UMB']
harmonicBiasesFile_UMB = parameters['harmonicBiasesFile']
trajDir_UMB = parameters['trajDir_UMB']
histBinEdgesFile_UMB = parameters['histBinEdgesFile_UMB']
histDir_UMB = parameters['histDir_UMB']
fMAPFile_UMB = parameters['fMAPFile_UMB']
fMHFile_UMB = parameters['fMHFile_UMB']
trajDir_PROJ = parameters['trajDir_PROJ']
histBinEdgesFile_PROJ = parameters['histBinEdgesFile_PROJ']
savedir = parameters['savedir']

# - post-processing and error checking
beta = 1/(kB*T)			# beta = 1/kBT = (kJ/mol)^(-1)

# - printing args to screen
print("")
print("T = %e" % (T))
print("dim_UMB = %d" % (dim_UMB))
print("periodicity_UMB = %s" % (periodicity_UMB))
print("harmonicBiasesFile_UMB = %s" % (harmonicBiasesFile_UMB))
print("trajDir_UMB = %s" % (trajDir_UMB))
print("histBinEdgesFile_UMB = %s" % (histBinEdgesFile_UMB))
print("histDir_UMB = %s" % (histDir_UMB))
print("fMAPFile_UMB = %s" % (fMAPFile_UMB))
print("fMHFile_UMB = %s" % (fMHFile_UMB))
print("trajDir_PROJ = %s" % (trajDir_PROJ))
print("histBinEdgesFile_PROJ = %s" % (histBinEdgesFile_PROJ))
print("savedir = %s"%(savedir))
print("")


# loading data
print("Loading umbrella simulation data...")

# - loading location and strength of harmonic biasing potentials in each of S umbrella simulations 
harmonicBiasesFile_UMB_DATA = []
with open(harmonicBiasesFile_UMB,'r') as fin:
	for line in fin:
		harmonicBiasesFile_UMB_DATA.append(line.strip().split()[:-1])

S = len(harmonicBiasesFile_UMB_DATA)

if len(harmonicBiasesFile_UMB_DATA[0]) != (1+2*dim_UMB):
	print("\nERROR - Number of columns in %s != 1+2*dim_UMB = %d as expected" % (harmonicBiasesFile_UMB,1+2*dim_UMB))
	sys.exit(-1)
	
umbC = [item[1:1+dim_UMB] for item in harmonicBiasesFile_UMB_DATA]
umbC = [[float(y) for y in x] for x in umbC]
umbC = np.array(umbC)

umbF = [item[1+dim_UMB:1+2*dim_UMB] for item in harmonicBiasesFile_UMB_DATA]
umbF = [[float(y) for y in x] for x in umbF]
umbF = np.array(umbF)

# - loading histogram bin edges 
binE = []
with open(histBinEdgesFile_UMB,'r') as fin:
	for line in fin:
		binE.append(line.strip().split())

if len(binE) != dim_UMB:
	print("\nERROR - %s does not contain expected number of dim_UMB = %d lines specifying histogram bin edges" % (histBinEdgesFile_UMB,dim_UMB))
	sys.exit(-1)

binE = [[float(y) for y in x] for x in binE]
binE = np.array(binE)

# - counting number of bins in each dimension M_k k=1..dim_UMB, and total number of bins M = product_k=1..dim_UMB M_k = M_1*M_2*...*M_dim_UMB 
M_k = np.zeros(dim_UMB, dtype=np.uint64)
for d in range(0,dim_UMB):
	M_k[d] = len(binE[d])-1
M = np.prod(M_k)

# - converting binEdges into binCenters and binWidths
binC = []
binW = []
for d in range(0,dim_UMB):
	binC_d = np.zeros(M_k[d])
	binW_d = np.zeros(M_k[d])
	for ii in range(0,M_k[d]):
		binC_d[ii] = 0.5 * ( binE[d][ii] + binE[d][ii+1] )
		binW_d[ii] = binE[d][ii+1] - binE[d][ii]
	binC.append(binC_d)
	binW.append(binW_d)

# - computing period in each periodic dimension as histogram bin range 
period = np.zeros(dim_UMB)
for d in range(0,dim_UMB):
	if periodicity_UMB[d] == 0:
		period[d] = float('nan')
	else:
		period[d] = binE[d][-1] - binE[d][0]


# loading histograms compiled over the S umbrella simulations recorded as row vectors in row major order (last index changes fastest) 
n_il = []
for i in range(0,S):
	hist_UMB_filename = histDir_UMB + '/hist_' + str(i+1) + '.txt'
	hist_UMB_DATA = []
	with open(hist_UMB_filename,'r') as fin:
		for line in fin:
			hist_UMB_DATA.append(line.strip().split())
	if len(hist_UMB_DATA) != 1:
		print("\nERROR - Did not find expected row vector in reading %s" % (hist_UMB_filename))
		sys.exit(-1)
	if len(hist_UMB_DATA[0]) != M:
		print("\nERROR - Row vector in %s did not contain expected number of elements M = M_1*M_2*...*M_dim_UMB = %d given histogram bins specified in %s" % (hist_UMB_filename,M,histBinEdgesFile_UMB))
		sys.exit(-1)
	n_il.append(hist_UMB_DATA[0])
n_il = [[float(y) for y in x] for x in n_il]
n_il = np.array(n_il,dtype = int)


# precomputing aggregated statistics
N_i = np.sum(n_il,axis=1)         # total counts in simulation i
M_l = np.sum(n_il,axis=0)         # total counts in bin l

print("DONE!\n\n")


print("Loading MAP and MH f values simulation data...")

f_i_MAP = np.exp(np.loadtxt(fMAPFile_UMB)[:,1]*beta)

if len(f_i_MAP) != S:
	print("\nERROR - Row vector in %s did not contain expected number of elements %d corresponding to number of simulations in directory %s" % (fMAPFile_UMB,S,histDir_UMB))
	sys.exit(-1)

f_i_MH = np.exp(np.loadtxt(fMHFile_UMB,ndmin = 2)*beta)

nSamples_MH = f_i_MH.shape[0]
if f_i_MH.shape[1] != S:
	print("\nERROR - Number of columns in %s did not contain expected number of elements %d corresponding to number of simulations in directory %s" % (fMHFile_UMB,S,histDir_UMB))
	sys.exit(-1)

print("DONE!\n\n")


print("Loading umbrella trajectory...")

traj = []
for i in range(0,S):
	traj_i_filename = trajDir_UMB + '/traj_' + str(i+1) + '.txt'
	traj_i_DATA = np.loadtxt(traj_i_filename,ndmin=2)[:,1:]
	if traj_i_DATA.shape[1] != dim_UMB:
		print("\nERROR - Dimensionality of %s is not dim_UMB = %d as expected" % (traj_i_filename,dim_UMB))
		sys.exit(-1)
	if traj_i_DATA.shape[0] != N_i[i]:
		print("\nERROR - Number of samples in %s does not match number of counts in histogram loaded from %s" % (traj_i_filename,histDir_UMB + '/hist_' + str(i+1) + '.txt'))
		sys.exit(-1)
	traj.append(traj_i_DATA)

print("DONE!\n\n")


print("Loading projection trajectory...")

traj_PROJ = []
for i in range(0,S):
	traj_PROJ_i_filename = trajDir_PROJ + '/traj_' + str(i+1) + '.txt'
	traj_PROJ_i_DATA = np.loadtxt(traj_PROJ_i_filename,ndmin=2)[:,1:]
	if i == 0:
		dim_PROJ = traj_PROJ_i_DATA.shape[1]
	else:
		if traj_PROJ_i_DATA.shape[1] != dim_PROJ:
			print("\nERROR - Dimensionality of %s does not match that of %s" % (traj_PROJ_i_filename,trajDir_PROJ + '/traj_1.txt'))
			sys.exit(-1)
	if traj_PROJ_i_DATA.shape[0] != N_i[i]:
		print("\nERROR - Number of samples in %s does not match number of counts in histogram loaded from %s" % (traj_PROJ_i_filename,histDir_UMB + '/hist_' + str(i+1) + '.txt'))
		sys.exit(-1)
	traj_PROJ.append(traj_PROJ_i_DATA)

print("DONE!\n\n")


print("Loading projection histogram bin edges...")

binE_PROJ = []
with open(histBinEdgesFile_PROJ,'r') as fin:
	for line in fin:
		binE_PROJ.append(line.strip().split())

if len(binE_PROJ) != dim_PROJ:
	print("\nERROR - %s does not contain expected number of dim_PROJ = %d lines specifying histogram bin edges" % (histBinEdgesFile_PROJ,dim_PROJ))
	sys.exit(-1)

binE_PROJ = [[float(y) for y in x] for x in binE_PROJ]
binE_PROJ = np.array(binE_PROJ)

# - counting number of bins in each dimension M_k_PROJ k=1..dim_PROJ, and total number of bins M = product_k=1..dim_PROJ M_k_PROJ = M_1*M_2*...*M_dim_UMB 
M_k_PROJ = np.zeros(dim_PROJ, dtype=np.uint64)
for d in range(0,dim_PROJ):
	M_k_PROJ[d] = len(binE_PROJ[d])-1
M_PROJ = np.prod(M_k_PROJ)

# - converting binEdges into binCenters and binWidths
binC_PROJ = []
binW_PROJ = []
for d in range(0,dim_PROJ):
	binC_PROJ_d = np.zeros(M_k_PROJ[d])
	binW_PROJ_d = np.zeros(M_k_PROJ[d])
	for ii in range(0,M_k_PROJ[d]):
		binC_PROJ_d[ii] = 0.5 * ( binE_PROJ[d][ii] + binE_PROJ[d][ii+1] )
		binW_PROJ_d[ii] = binE_PROJ[d][ii+1] - binE_PROJ[d][ii]
	binC_PROJ.append(binC_PROJ_d)
	binW_PROJ.append(binW_PROJ_d)

print("DONE!\n\n")


# precomputing biasing potentials for each simulation i in each bin l 
print("Computing biasing potentials for each simulation in each bin...")
c_il = np.zeros((S,M))		# S-by-M matrix containing biases due to artificial harmonic potential for simulation i in bin l
for i in range(0,S):
	for l in range(0,M):
		sub = ind2sub_RMO(M_k,l)
		expArg = 0
		for d in range(0,dim_UMB):
			if periodicity_UMB[d] != 0:
				delta = min( abs(binC[d][sub[d]]-umbC[i,d]), abs(binC[d][sub[d]]-umbC[i,d]+period[d]), abs(binC[d][sub[d]]-umbC[i,d]-period[d]) )
			else:
				delta = abs(binC[d][sub[d]]-umbC[i,d])
			expArg = expArg + 0.5*umbF[i,d]*math.pow(delta,2)
		c_il[i,l] = math.exp(-beta*expArg)
	print("\tProcessing of simulation %d of %d complete" % (i+1,S))
print("DONE!\n\n")


# performing MAP and MH projections into projection trajectory variables
print("Performing MAP and MH projection...")

p_l_PROJ_MAP = np.zeros(M_PROJ)
p_l_PROJ_MH = np.zeros((M_PROJ,nSamples_MH))

for i in range(0,S):
	for k in range(0,N_i[i]):
	
		# traj_PROJ
		
		# - extracting
		traj_PROJ_ik = traj_PROJ[i][k,:]
		
		# - bound checking
		for d in range (0,dim_PROJ):
			if ( ( traj_PROJ_ik[d] < binE_PROJ[d][0] ) | ( traj_PROJ_ik[d] >= binE_PROJ[d][-1] ) ):
				print("\tWARNING -- dimension %d of entry %d in projection trajectory file %s = %e is out of projection histogram bounds [%e,%e); this observation will be skipped" % (d+1,k+1,trajDir_PROJ + '/traj_' + str(i+1) + '.txt',traj_PROJ_ik[d],binE_PROJ[d][0],binE_PROJ[d][-1]))
				sys.exit(-1)
		
		# - indexing
		sub_PROJ = binner(traj_PROJ_ik,binE_PROJ)
		if sub_PROJ[0] < 0:
			continue
		idx_PROJ = sub2ind_RMO(M_k_PROJ,sub_PROJ)
		
		
		# traj
		
		# - extracting
		traj_ik = traj[i][k,:]
		
		# - bound checking
		for d in range (0,dim_UMB):
			if ( ( traj_ik[d] < binE[d][0] ) | ( traj_ik[d] >= binE[d][-1] ) ):
				print("\tWARNING -- dimension %d of entry %d in umbrella trajectory file %s = %e is out of umbrella histogram bounds [%e,%e); this observation will be skipped" % (d+1,k+1,trajDir_UMB + '/traj_' + str(i+1) + '.txt',traj_ik[d],binE[d][0],binE[d][-1]))
				sys.exit(-1)
		
		# - indexing
		sub_UMB = binner(traj_ik,binE)
		if sub_UMB[0] < 0:
			continue
		idx_UMB = sub2ind_RMO(M_k,sub_UMB)
		
		
		# accumulating reweighting
		p_l_PROJ_MAP[idx_PROJ] = p_l_PROJ_MAP[idx_PROJ] + 1/np.sum(N_i*c_il[:,idx_UMB]*f_i_MAP)
		for q in range(0,nSamples_MH):
			p_l_PROJ_MH[idx_PROJ,q] = p_l_PROJ_MH[idx_PROJ,q] + 1/np.sum(N_i*c_il[:,idx_UMB]*f_i_MH[q,:])
	
	print("\tProjection of simulation %d of %d complete" % (i+1,S))	
		
print("DONE!\n\n")

# - normalizing p_l_PROJ_MAP and p_l_PROJ_MH
p_l_PROJ_MAP = np.divide( p_l_PROJ_MAP, np.sum(p_l_PROJ_MAP) )
for q in range(0,nSamples_MH):
	p_l_PROJ_MH[:,q] = np.divide( p_l_PROJ_MH[:,q], np.sum(p_l_PROJ_MH[:,q]) )

# - converting p_l_PROJ_MAP into probability density function pdf_l_PROJ_MAP and free energy estimate betaF_l_PROJ_MAP 
#   -> mean zeroing betaF; when plotting multiple free energy surfaces this is equivalent to minimizing the mean squared error between the landscapes 
pdf_l_PROJ_MAP = np.zeros(p_l_PROJ_MAP.size)
betaF_l_PROJ_MAP = np.ones(p_l_PROJ_MAP.size)*float('nan')
for l in range(0,M_PROJ):
	sub = ind2sub_RMO(M_k_PROJ,l)
	binVol_PROJ = 1
	for d in range(0,dim_PROJ):
		binVol_PROJ = binVol_PROJ*binW_PROJ[d][sub[d]]
	if p_l_PROJ_MAP[l] > 0:
		pdf_l_PROJ_MAP[l] = p_l_PROJ_MAP[l]/binVol_PROJ
		betaF_l_PROJ_MAP[l] = -math.log(p_l_PROJ_MAP[l]/binVol_PROJ)
betaF_l_PROJ_MAP = betaF_l_PROJ_MAP - np.amin(betaF_l_PROJ_MAP[np.isfinite(betaF_l_PROJ_MAP)])

# - writing to file bin centers and bin widths of l=1..M_PROJ bins of dim_PROJ-dimensional rectilinear histogram grid as row vectors in row major order (last index changes fastest) 
with open("%s/hist_binCenters_PROJ.txt"%(savedir),"w") as fout:
	for d in range(0,dim_PROJ):
		for ii in range(0,M_k_PROJ[d]):
			fout.write("%15.5e" % (binC_PROJ[d][ii]))
		fout.write("\n")

with open("%s/hist_binWidths_PROJ.txt"%(savedir),"w") as fout:
	for d in range(0,dim_PROJ):
		for ii in range(0,M_k_PROJ[d]):
			fout.write("%15.5e" % (binW_PROJ[d][ii]))
		fout.write("\n")

# - writing to file MAP estimates for p_l_PROJ_MAP, pdf_l_PROJ_MAP, and betaF_l_PROJ_MAP over l=1..M_PROJ bins of dim_PROJ-dimensional rectilinear histogram grid as row vectors in row major order (last index changes fastest) 
with open("%s/p_PROJ_MAP.txt"%(savedir),"w") as fout:
	for l in range(0,M_PROJ):
		fout.write("%15.5e" % (p_l_PROJ_MAP[l]))
	fout.write("\n")

with open("%s/pdf_PROJ_MAP.txt"%(savedir),"w") as fout:
	for l in range(0,M_PROJ):
		fout.write("%15.5e" % (pdf_l_PROJ_MAP[l]))
	fout.write("\n")

with open("%s/betaF_PROJ_MAP.txt"%(savedir),"w") as fout:
	for l in range(0,M_PROJ):
		fout.write("%15.5e" % (betaF_l_PROJ_MAP[l]))
	fout.write("\n")


# - converting p_l_PROJ_MH into pdf_l_PROJ_MH and betaF_l_PROJ_MH
#   -> mean zeroing betaF; when plotting multiple free energy surfaces this is equivalent to minimizing the mean squared error between the landscapes 
# pdf_l_PROJ_MH = np.zeros(p_l_PROJ_MH.shape)
# betaF_l_PROJ_MH = np.ones(p_l_PROJ_MH.shape)*float('nan')
# for l in range(0,M_PROJ):
# 	sub_PROJ = ind2sub_RMO(M_k_PROJ,l)
# 	binVol_PROJ = 1
# 	for d in range(0,dim_PROJ):
# 		binVol_PROJ = binVol_PROJ*binW_PROJ[d][sub_PROJ[d]]
# 	for jj in range(0,nSamples_MH):
# 		if p_l_PROJ_MH[l,jj] > 0:
# 			pdf_l_PROJ_MH[l,jj] = p_l_PROJ_MH[l,jj]/binVol_PROJ
# 			betaF_l_PROJ_MH[l,jj] = -math.log(p_l_PROJ_MH[l,jj]/binVol_PROJ)
# betaF_l_PROJ_MH_colMin = np.ones(nSamples_MH)*float('nan')
# for jj in range(0,nSamples_MH):
# 	betaF_l_PROJ_MH_col = betaF_l_PROJ_MH[:,jj]
# 	betaF_l_PROJ_MH_colMin[jj] = np.amin(betaF_l_PROJ_MH_col[np.isfinite(betaF_l_PROJ_MH_col)])
# betaF_l_PROJ_MH = betaF_l_PROJ_MH - np.matlib.repmat( betaF_l_PROJ_MH_colMin, betaF_l_PROJ_MH.shape[0], 1 )

# # - writing to file nSamples_MH Metropolis-Hastings samples from Bayes posterior of p_l_PROJ_MH, pdf_l_PROJ_MH, and betaF_l_PROJ_MH over l=1..M_PROJ bins of dim_PROJ-dimensional rectilinear histogram grid as row vectors in row major order (last index changes fastest) 
# with open("%s/p_PROJ_MH.txt"%(savedir),"w") as fout:
# 	for jj in range(0,nSamples_MH):
# 		for l in range(0,M_PROJ):
# 			fout.write("%15.5e" % (p_l_PROJ_MH[l,jj]))
# 		fout.write("\n")

# with open("%s/pdf_PROJ_MH.txt"%(savedir),"w") as fout:
# 	for jj in range(0,nSamples_MH):
# 		for l in range(0,M_PROJ):
# 			fout.write("%15.5e" % (pdf_l_PROJ_MH[l,jj]))
# 		fout.write("\n")

# with open("%s/betaF_PROJ_MH.txt"%(savedir),"w") as fout:
# 	for jj in range(0,nSamples_MH):
# 		for l in range(0,M_PROJ):
# 			fout.write("%15.5e" % (betaF_l_PROJ_MH[l,jj]))
# 		fout.write("\n")




