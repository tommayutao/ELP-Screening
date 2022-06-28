import numpy as np
import mdtraj as md
import os,re

def read_xvg(fname):
	with open(fname, 'r') as f:
		for i, line in enumerate(f):
			if not line.startswith(('#', '@')):
				skip = i
				break
	return np.genfromtxt(fname, skip_header=skip)

def extract_frames(name,save_dir,start_idx = 0):
	traj_file,top_file,tpr_file,pullx_file = '%s.xtc'%(name),'%s.gro'%(name),'%s.tpr'%(name),'%s_pullx.xvg'%(name)
	os.makedirs('%s/init_config'%(save_dir))
	traj = md.load_xtc(traj_file,top=top_file)
	times = traj.time
	pullx = read_xvg(pullx_file)
	assert (len(traj) == len(pullx))
	assert (np.allclose(times,pullx[:,0]))
	pull_centers = pullx[:,[2,7]]
	os.system('echo bilayer pulled | gmx_mpi mindist -f %s -n index.ndx -group -od mindist_%s.xvg -on numcount_%s.xvg'%(traj_file,name,name))
	os.system('gmx_mpi distance -f %s -s %s -n index.ndx -select "com of group "bilayer" plus com of group "pulled"" -oxyz zcom_%s.xvg'%(traj_file,tpr_file,name))
	numcount = read_xvg('numcount_%s.xvg'%(name))
	assert (np.allclose(times,numcount[:,0]))
	simulation_idx = 0
	frame_idx = start_idx
	with open('%s/ub_info.txt'%(save_dir),'a') as f:
		f.write('Simulation_index Frame_index  center_zh  center_zt  k_npt  k_md\n')
		while True:
			cur = pull_centers[frame_idx,:]
			if numcount[frame_idx,1] < 25:
				spacing = 0.15
				f.write('%d %d %.8f  %8f  1000 1000\n'%(simulation_idx,frame_idx,cur[0],cur[1]))
			else:
				if name == 'pull1':
					spacing = 0.03
					f.write('%d %d %.8f  %8f  12000 20000\n'%(simulation_idx,frame_idx,cur[0],cur[1]))
				else:
					spacing = 0.05
					f.write('%d %d %.8f  %8f  8000 10000\n'%(simulation_idx,frame_idx,cur[0],cur[1]))
			os.system('echo 0 | gmx_mpi trjconv -f %s -s %s -pbc atom -dump %.4f -o %s/init_config/conf%d.gro'%(traj_file,tpr_file,pullx[frame_idx,0],save_dir,frame_idx))
			if frame_idx == len(pull_centers) - 1:
				break
			dist = np.linalg.norm(pull_centers[frame_idx+1:,:] - cur,axis = 1)
			frame_idx = frame_idx + 1 + np.argmin(abs(dist - spacing))
			simulation_idx += 1
	return

def write_submits(save_dir):
	ub_info = np.loadtxt('%s/ub_info.txt'%(save_dir),skiprows=1)
	for idx in range(len(ub_info)):
		k_npt,k_md = ub_info[idx,-2:]
		os.makedirs('%s/ub%d'%(save_dir,idx),exist_ok = True)
		with open('mdp_files/template_ub_npt.mdp','r') as f:
			contents = f.read()
		contents = re.sub('k1','%d'%(k_npt),contents)
		contents = re.sub('XXX','%8f'%(ub_info[idx,2]),contents)
		contents = re.sub('YYY','%8f'%(ub_info[idx,3]),contents)
		with open('%s/ub%d_npt.mdp'%(save_dir,idx),'w') as f:
			f.write(contents)

		with open('mdp_files/template_ub_md.mdp','r') as f:
			contents = f.read()
		contents = re.sub('k2','%d'%(k_md),contents)
		contents = re.sub('XXX','%8f'%(ub_info[idx,2]),contents)
		contents = re.sub('YYY','%8f'%(ub_info[idx,3]),contents)
		with open('%s/ub%d_md.mdp'%(save_dir,idx),'w') as f:
			f.write(contents)

		with open('template_sub_ub.sbatch','r') as f:
			contents = f.read()
		contents = re.sub('XXX','ub%d'%(idx),contents)
		contents = re.sub('YYY','init_config/conf%d.gro'%(ub_info[idx,1]),contents)
		with open('%s/sub_ub%d.sbatch'%(save_dir,idx),'w') as f:
			f.write(contents)
	return

extract_frames('pull1','ub_stage1',start_idx = 0)
write_submits('ub_stage1')

extract_frames('pull2','ub_stage2',start_idx = 1)
write_submits('ub_stage2')

extract_frames('pull3','ub_stage3',start_idx = 1)
write_submits('ub_stage3')


