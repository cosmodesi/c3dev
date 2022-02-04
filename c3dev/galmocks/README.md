## Mock Galaxy Challenge

This sub-package contains source code used to make mock galaxy catalogs for the C3 Mock Challenge.

### Installation

To set up a typical working environment:

$ conda create -n c3gmc python=3.9 numpy scipy matplotlib cython numba h5py astropy ipython jupyter halotools jax corrfunc pytest asdf pytest flake8
$ conda activate c3gmc
$ cd /path/to/c3dev
$ python setup.py install

If you want to create a jupyter kernel to use via https://jupyter.nersc.gov/, you will need to create and activate the environment as above, and then do:

$ python -m ipykernel install —user —name c3gmc —display-name c3gmc_jkernel

See https://docs.nersc.gov/services/jupyter/#conda-environments-as-kernels for further details about configuring a jupyter kernel on NERSC.
