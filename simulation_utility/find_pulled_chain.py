import numpy as np
import mdtraj as md
import os,re

def compute_com(coords,boxdim):
	pivot = coords[0,:]
	displacement = (coords-pivot) - boxdim*np.rint((coords-pivot)/boxdim)
	com = np.mean(pivot+displacement,axis=0)
	return (com-boxdim*np.rint(com/boxdim))  

def rewrite_contents(fname,pbc_atom_index):
	with open(fname,'r') as f:
		contents = f.read()
	contents = re.sub('PBC_ATOM','%d'%(pbc_atom_index),contents)
	with open(fname,'w') as f:
		f.write(contents)
	return

traj = md.load_xtc('md.xtc',top = 'md.gro')
coords = traj.xyz[-1]
top = traj.topology
df = top.to_dataframe()[0]
atoms_per_chain = md.load_pdb('elp_cg.pdb').n_atoms

peptide_index = df[(df['resName'] != 'W') & (df['resName'] != 'ION')]['serial'].to_numpy()
upper_idx = peptide_index[:len(peptide_index)//2]
upper_coords = coords[upper_idx-1,:]
com_upper = np.mean(upper_coords,axis=0)

# find pulled chain #
min_diff = np.inf
i = 0
while i < len(upper_coords):
	cur_com = compute_com(upper_coords[i:i+atoms_per_chain,:],traj.unitcell_lengths[-1,:])
	if np.linalg.norm(cur_com - com_upper) < min_diff:
		min_diff = np.linalg.norm(cur_com - com_upper)
		ans = upper_idx[i:i+atoms_per_chain]
	i += atoms_per_chain
print (len(ans))
print ("Selected pulled chain atom indices: ", ans)

# find reference atom for COM computation of bilayer #
bilayer_idx = np.setdiff1d(peptide_index,ans)
com_bilayer = np.mean(coords[bilayer_idx-1,:],axis=0)
min_idx = np.argmin(np.linalg.norm(coords[bilayer_idx-1,:]-com_bilayer,axis=1))
print ("Bilayer COM reference atom index: ", bilayer_idx[min_idx])
rewrite_contents('mdp_files/pull1.mdp',bilayer_idx[min_idx])
rewrite_contents('mdp_files/pull2.mdp',bilayer_idx[min_idx])
rewrite_contents('mdp_files/pull3.mdp',bilayer_idx[min_idx])
rewrite_contents('mdp_files/template_ub_npt.mdp',bilayer_idx[min_idx])
rewrite_contents('mdp_files/template_ub_md.mdp',bilayer_idx[min_idx])




