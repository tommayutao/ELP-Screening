import numpy as np
import mdtraj as md
import os,glob

def check_file(fname,expected):
	pdb_file = md.load_pdb(fname)
	return pdb_file.n_atoms == expected

pdb_file = md.load_pdb('elp_cg.pdb')
n_atoms = pdb_file.n_atoms
coord = pdb_file.xyz[0]
L = max(coord[:,-1]) - min(coord[:,-1])
N = 10 # number of chains per side

while (not os.path.exists('upper_layer.pdb')) or (not check_file('upper_layer.pdb',N**2*n_atoms)): 
	if not os.path.exists('upper_layer.pdb'):
		spacing = 0.9
	else:
		spacing += 0.1
	grid = np.arange(N)*spacing
	COM_pos1 = []
	box = [N*spacing,N*spacing,2*L+2]
	for x in grid:
		for y in grid:
			COM_pos1.append([x,y,box[-1]*0.5+L*0.5])
	np.savetxt('pos_upper.dat',COM_pos1)
	os.system('gmx_mpi insert-molecules -ci elp_cg.pdb -ip pos_upper.dat -box %.2f %.2f %.2f -nmol %d -rot none -o upper_layer.pdb'%(box[0],box[1],box[2],N**2))

while (not os.path.exists('bilayer.pdb')) or (not check_file('bilayer.pdb',N**2*n_atoms*2)):
	if not os.path.exists('bilayer.pdb'):
		dz = 0.0
	else:
		dz += 0.1
	COM_pos2 = []
	for x in grid:
		for y in grid:
			COM_pos2.append([x,y,box[-1]*0.5-L*0.5-dz])	
	np.savetxt('pos_lower.dat',COM_pos2)
	os.system('gmx_mpi insert-molecules -f upper_layer.pdb -ci elp_reverted.pdb -ip pos_lower.dat -nmol %d -rot none -o bilayer.pdb'%(N**2))

for backup in glob.glob('#*'):
	os.remove(backup)

print ([spacing,dz])





