echo 0 | gmx_mpi editconf -f elp_cg.pdb -o elp_cg.pdb -rotate 0 90 0 -princ -c
echo 0 | gmx_mpi editconf -f elp_cg.pdb -o elp_reverted.pdb -rotate 0 90 0 -princ
python3 build_layer.py
echo 0 0 | gmx_mpi trjconv -f bilayer.pdb -s bilayer.pdb -center -pbc atom -o bilayer.pdb
python3 setup_solvation.py
packmol < solvate.inp
echo 0 0 | gmx_mpi trjconv -f bilayer_solvated.pdb -s bilayer_solvated.pdb -center -pbc atom -o bilayer_solvated.pdb
echo "Finish setting up solvated bilayer. Please modify system.top to ensure correct number of peptide chains and water molecule."
