import numpy as np
import mdtraj as md
import re

def compute_num_water(zmin,zmax,xmin,xmax,ymin,ymax,density=1.0):
	V = (zmax-zmin)*(xmax-xmin)*(ymax-ymin)*1e-24 # cm^(-3)
	M = density*V ## gram
	return int(np.ceil(M/(4*2.9915e-23)))

def multiple_replace(dict, text):
	# Create a regular expression  from the dictionary keys
	regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
	# For each match, look-up corresponding value in dictionary
	return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

traj = md.load_pdb('bilayer.pdb')
x_max = traj.unitcell_lengths[0][0]*10 # convert to Angstrom
y_max = traj.unitcell_lengths[0][1]*10 # convert to Angstrom
coords = traj.xyz[0]*10 # convert to Angstrom

with open('solvate_template.inp','r') as f:
	contents = f.read()

z_up_min = np.amax(coords[:,-1])
z_up_max = z_up_min + 345 ## 34.5 nm of water above works find empirically
num_water_up = compute_num_water(z_up_min,z_up_max,0.0,x_max,0.0,y_max,density=1.0)
sub_dict = {'NUM_UP':str(num_water_up),'X_MAX':'%.3f'%(x_max),'Y_MAX':'%.3f'%(y_max),'Z_UP_MIN':'%.3f'%(z_up_min),'Z_UP_MAX':'%.3f'%(z_up_max)}
contents = multiple_replace(sub_dict, contents)

z_low_max = np.amin(coords[:,-1])
z_low_min = z_low_max - 100 ## 10 nm of water below works find empirically
num_water_low = compute_num_water(z_low_min,z_low_max,0.0,x_max,0.0,y_max,density=1.0)
sub_dict = {'NUM_LOW':str(num_water_low),'X_MAX':'%.3f'%(x_max),'Y_MAX':'%.3f'%(y_max),'Z_LOW_MIN':'%.3f'%(z_low_min),'Z_LOW_MAX':'%.3f'%(z_low_max)}
contents = multiple_replace(sub_dict, contents)

with open('solvate.inp','w') as f:
	f.write(contents)

