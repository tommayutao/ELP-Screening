import numpy as np
import glob,os,pickle,sys,re,json

def read_xvg(fname):
	with open(fname, 'r') as f:
		for i, line in enumerate(f):
			if not line.startswith(('#', '@')):
				skip = i
				break
	return np.genfromtxt(fname, skip_header=skip)

def read_info(f):
	with open(f,'r') as temp:
		lines = temp.readlines()
		for l in lines:
			if l.startswith('pull_coord1_k'):
				k1 = float(l.strip('\n').split(' ')[-1])
			elif l.startswith('pull_coord2_k'):
				k2 = float(l.strip('\n').split(' ')[-1])
			elif l.startswith('dt'):
				dt = float(l.strip('\n').split(' ')[-1])
			elif l.startswith('nsteps'):
				nsteps = float(l.strip('\n').split(' ')[-1])
	return k1,k2,dt,nsteps

def find_factors(x):
	ans = []
	for i in range(1, x + 1):
		if x % i == 0:
			ans.append(i)
	return ans

def write_submit(parent_outdir,b_size,b_idx,min_zh,max_zh,min_zt,max_zt,metafile,reweight_params,nbins=100):
	bias_outfile = '%s/size_%d/b%d/bias.txt'%(parent_outdir,b_size,b_idx)
	G_outfile = '%s/size_%d/b%d/G.txt'%(parent_outdir,b_size,b_idx)
	with open('sub_wham2d_%d_%d.sbatch'%(b_size,b_idx),'w') as f:
		f.write('#!/bin/bash\n')
		f.write('#SBATCH --job-name=wham2d_%d_%d\n'%(b_size,b_idx))
		f.write('#SBATCH --output=%s/wham2d_%d_%d.out\n'%(parent_outdir,b_size,b_idx))
		f.write('#SBATCH --partition=fela-cpu\n')
		f.write('##SBATCH --qos=gm4-cpu\n')
		f.write('#SBATCH --nodes=1\n')
		f.write('#SBATCH --ntasks-per-node=1\n')
		f.write('#SBATCH --cpus-per-task=1\n')
		f.write('##SBATCH --mem=32G\n')
		f.write('##SBATCH --nodelist=midway2-0740\n')
		f.write('source /home/yma3/.bashrc\n')
		f.write('wham-2d Px=0 %.8f %.8f %d Py=0 %.8f %.8f %d 1e-5 300.0 0 %s %s %s 1\n'%(min_zh-0.01,max_zh+0.01,nbins,min_zt-0.01,max_zt+0.01,nbins,metafile,bias_outfile,G_outfile))
		f.write('python3 BayesReweight.py %s\n'%(reweight_params))
		f.write('mv sub_wham2d_%d_%d.sbatch %s\n'%(b_size,b_idx,parent_outdir))
		f.write('mv %s %s\n'%(metafile,parent_outdir))
		f.write('mv %s %s'%(reweight_params,parent_outdir))

def extract_ub_folders(parent):
	temp = glob.glob('%s/ub*'%(parent))
	folders = []
	for f in temp:
		if os.path.isdir(f) and len(glob.glob('%s/*.sbatch'%(f)))>0:
			folders.append(f)
	folders.sort(key = lambda x : int(re.findall(r'\d+', x)[-1]))
	return folders


def load_data(size,step=1):
	stage1 = extract_ub_folders('ub_stage1')
	stage2 = extract_ub_folders('ub_stage2')
	stage3 = extract_ub_folders('ub_stage3')
	ub_folders = stage1+stage2+stage3
	print (ub_folders)

	min_zh,max_zh = np.inf,-np.inf
	min_zt,max_zt = np.inf,-np.inf	
	min_zcom,max_zcom = np.inf,-np.inf
	coords_dict = dict()
	harmonic_info_dict = dict()
	for i in range(len(ub_folders)):
		f = ub_folders[i]
		print ('loading data from %s'%(f))
		pullx_file = glob.glob('%s/*_md_pullx.xvg'%(f))[0]
		pullx = read_xvg(pullx_file)
		k1,k2,dt,nsteps = read_info('%s/mdout_md.mdp'%(f))
		if not np.isclose(pullx[-1,0],dt*nsteps):
			print ("Some error in folder %s"%(f))
			continue
		harmonic_info_dict[i+1] = [pullx[0,2],pullx[0,4],k1,k2]
		time = pullx[(len(pullx)-size)::step,0]
		zh = pullx[(len(pullx)-size)::step,1]
		zt = pullx[(len(pullx)-size)::step,3]
		zcom = pullx[(len(pullx)-size)::step,-2]
		coords_dict[i+1] = [np.stack((time,zh,zt),axis=1),np.stack((time,zcom),axis=1)]
		min_zh, max_zh = min(min_zh,np.amin(zh)), max(max_zh,np.amax(zh))
		min_zt, max_zt = min(min_zt,np.amin(zt)), max(max_zt,np.amax(zt))
		min_zcom,max_zcom = min(min_zcom,np.amin(zcom)),max(max_zcom,np.amax(zcom))
	return (len(time),min_zh,max_zh,min_zt,max_zt,min_zcom,max_zcom,harmonic_info_dict,coords_dict)
		
parent_outdir = sys.argv[1]
os.mkdir(parent_outdir)
length = 5000
N,min_zh,max_zh,min_zt,max_zt,min_zcom,max_zcom,harmonic_biases,coords_dict = load_data(length,step=1)
block_sizes = find_factors(N)
block_sizes.sort()
print (block_sizes)
with open('%s/coords_dict.pickle'%(parent_outdir),'wb') as handle:
	pickle.dump(coords_dict,handle)
with open('%s/min_max.txt'%(parent_outdir),'w') as f:
	f.write('%.8f  %.8f\n'%(min_zh,max_zh))
	f.write('%.8f  %.8f\n'%(min_zt,max_zt))
	f.write('%.8f  %.8f\n'%(min_zcom,max_zcom))
print (sorted(coords_dict.keys()))
print (sorted(harmonic_biases.keys()))

nbins_umb = 100
nbins_proj = 250

for b_size in block_sizes:
	if b_size < int(N/10):
		continue
	print ('Block size: ', b_size)
	os.mkdir('%s/size_%d'%(parent_outdir,b_size))
	Nb = int(N/b_size)
	for j in range(Nb):
		metafile = 'wham_metafile_%d_%d.txt'%(b_size,j)
		meta_file = open(metafile,'a')
		savedir = '%s/size_%d/b%d'%(parent_outdir,b_size,j)
		os.mkdir(savedir)

		umb_dir = '%s/umb_trajs'%(savedir)
		proj_dir = '%s/proj_trajs'%(savedir)
		umb_hist_dir = '%s/umb_hists'%(savedir)
		os.mkdir(umb_dir)
		os.mkdir(proj_dir)
		os.mkdir(umb_hist_dir)

		for key in sorted(coords_dict.keys()):
			umb_coords = coords_dict[key][0]
			umb_file = '%s/traj_%d.txt'%(umb_dir,key)
			np.savetxt(umb_file,umb_coords[j*b_size:(j+1)*b_size,:],fmt = '%d %.8f %.8f')
			hist,xedges,yedges = np.histogram2d(umb_coords[j*b_size:(j+1)*b_size,1],umb_coords[j*b_size:(j+1)*b_size,2],
												bins=nbins_umb,range=[[min_zh-0.01,max_zh+0.01],[min_zt-0.01,max_zt+0.01]])
			np.savetxt('%s/hist_%d.txt'%(umb_hist_dir,key),hist.flatten(),fmt='%d',newline = " ")
			cur_bias = harmonic_biases[key]
			meta_file.write('%s %.8f %.8f %.3f %.3f 0.0\n'%(umb_file,cur_bias[0],cur_bias[1],cur_bias[2],cur_bias[3]))

			proj_coords = coords_dict[key][1]
			proj_file = '%s/traj_%d.txt'%(proj_dir,key)
			np.savetxt(proj_file,proj_coords[j*b_size:(j+1)*b_size,:],fmt = '%d %.8f')	
		meta_file.close()
		proj_edges = np.linspace(min_zcom-0.01,max_zcom+0.01,num=nbins_proj+1)
		np.savetxt('%s/proj_edges.txt'%(savedir),proj_edges,newline=" ")
		np.savetxt('%s/umb_edges.txt'%(savedir),np.stack((xedges,yedges),axis=0))
		np.savetxt('%s/f_MH.txt'%(savedir),np.empty((1,len(coords_dict.keys())))*np.nan)

		parameters = dict()
		parameters['T'] = 300.0
		parameters['dim_UMB'] = 2
		parameters['periodicity_UMB'] = [0,0]
		parameters['harmonicBiasesFile'] = metafile
		parameters['trajDir_UMB'] = umb_dir
		parameters['histBinEdgesFile_UMB'] = '%s/umb_edges.txt'%(savedir)
		parameters['histDir_UMB'] = umb_hist_dir
		parameters['fMAPFile_UMB'] = '%s/bias.txt'%(savedir)
		parameters['fMHFile_UMB'] = '%s/f_MH.txt'%(savedir)
		parameters['trajDir_PROJ'] = proj_dir
		parameters['histBinEdgesFile_PROJ'] = '%s/proj_edges.txt'%(savedir)
		parameters['savedir'] = savedir
		reweight_params = 'reweight_params_%d_%d.json'%(b_size,j)
		with open(reweight_params, 'w') as json_file:
			json.dump(parameters, json_file, indent=4)
		write_submit(parent_outdir,b_size,j,min_zh,max_zh,min_zt,max_zt,metafile,reweight_params,nbins=nbins_umb)			



	




