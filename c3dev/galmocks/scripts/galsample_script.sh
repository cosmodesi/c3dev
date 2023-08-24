#!/bin/bash

#SBATCH --job-name=galsample
#SBATCH --account=halotools
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

# Load software
source ~/anaconda3/etc/profile.d/conda.sh
conda activate diffwhatever

# Run python script
srun python make_diffsky_mock.py fixedAmp_001_hlist_0.67120.list.hdf5 UM_SMDPL_a0p6643_mock.hdf5 mock_z0p5 0.5053 0.4899 fixedAmp_001_galsampled_mock_67120.hdf5

conda deactivate
