from setuptools import find_packages, setup

PACKAGENAME = "c3dev"
VERSION = "0.1.0"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author=["Chris Blake", "Andrew Hearin", "Alexie Leauthaud", "John Moustakas"],
    author_email=["ahearin@anl.gov"],
    description="Development code and documentation for the DESI C3 Working Group",
    long_description="Development code and documentation for the DESI C3 Working Group",
    install_requires=[
        "numpy",
        "scipy",
        "healpy",
        "astropy",
        "numba",
        "halotools",
        "h5py",
        "jax",
    ],
    packages=find_packages(),
    url="https://github.com/cosmodesi/c3dev",
    package_data={
        "c3dev": (
            "galmocks/galhalo_models/baryonic_effects/ds_fit_data/*/*.txt",
            "galmocks/galhalo_models/baryonic_effects/tests/testing_data/*.txt",
            "galmocks/galhalo_models/baryonic_effects/tests/testing_data/*.npy",
        )
    },
)
