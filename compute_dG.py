import numpy as np
import glob,re,os
from scipy.stats import norm,kstest,sem
from scipy.optimize import curve_fit
import glob,re,os

base_dir = 'simulation_data'

def gaussian_pdf(x,mean,std):
	return norm.pdf(x,mean,std)

def r_square(popt,xdata,ydata):
    residuals = ydata - norm.pdf(xdata,*popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    return 1 - (ss_res / ss_tot)	

def fit_gaussian(peptide_dir):
	ub_info = np.loadtxt('%s/ub_stage3/ub_info.txt'%(peptide_dir),skiprows=1)	
	gaussian_params = np.zeros((len(ub_info),2))
	r2 = np.zeros(len(ub_info))
	for i in range(len(ub_info)):
		data = np.loadtxt('%s/ub_stage3/ub%d/ub%d_md_pullx.xvg'%(peptide_dir,i,i),skiprows = 38)
		com = data[:,-2]
		density,edges = np.histogram(com[-5000:],bins=100,density = True)
		centers = 0.5*(edges[1:]+edges[:-1])
		popt,pcov = curve_fit(gaussian_pdf,centers,density,p0=[np.mean(com),np.std(com,ddof=1)])
		gaussian_params[i,:] = popt
		r2[i] = r_square(popt,centers,density)
	return gaussian_params,r2

def translational_entropy_correction(peptide_name):
	peptide_dir = base_dir+'/'+peptide_name
	gaussian_params,_ = fit_gaussian(peptide_dir)
	sigma = np.mean(gaussian_params[:,1]) ## sqrt(kT/K_r)
	L = (1660)**(1/3.)*0.1 ## V_f^(1/3) in nm
	return np.log(L/(np.sqrt(2*np.pi)*sigma))

def read_xvg(fname):
	with open(fname, 'r') as f:
		for i, line in enumerate(f):
			if not line.startswith(('#', '@')):
				skip = i
				break
	return np.genfromtxt(fname, skip_header=skip)


def block_analysis_proj(peptide_name):
	peptide_dir = base_dir+'/'+peptide_name
	## find zcom cutoff for bilayer/bulk solvent ##
	zcom = read_xvg('%s/zcom_pull2.xvg'%(peptide_dir))
	cutoff = zcom[-1,-1]
	
	## Compute block averages of delta G ##
	folders = glob.glob('%s/wham_whole_ub/size_*'%(peptide_dir))
	folders.sort(key = lambda x: int(re.findall(r'\d+', x)[-1]))
	dG_means,dG_error,block_sizes = [],[],[]
	for f in folders:
		block_sizes.append(int(re.findall(r'\d+', f)[-1]))
		dG = []
		sub_f = glob.glob('%s/b*'%(f))
		sub_f.sort(key = lambda x : int(re.findall(r'\d+', x)[-1]))
		for rua in sub_f:
			if not os.path.exists('%s/betaF_PROJ_MAP.txt'%(rua)):
				continue
			betaG = np.loadtxt('%s/betaF_PROJ_MAP.txt'%(rua))
			bincenters = np.loadtxt('%s/hist_binCenters_PROJ.txt'%(rua))
			idx = np.isfinite(betaG)
			betaG = betaG[idx]
			bincenters = bincenters[idx]
			dG.append(np.amin(betaG[bincenters < cutoff]) - np.amin(betaG[bincenters>=cutoff]))
		dG_means.append(np.mean(dG))
		dG_error.append(sem(dG))
	print ('dG_means: ',dG_means)
	print ('dG_error: ',dG_error)
	return dG_means[np.nanargmin(dG_error)],dG_error[np.nanargmin(dG_error)]