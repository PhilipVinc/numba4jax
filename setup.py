from setuptools import setup, find_packages

BASE_DEPENDENCIES = [
    "numpy~=1.18",
    "numba>=0.52, <0.55",
    "cffi>=1.12",
    "jax>0.2.16, <0.3",
    "jaxlib>=0.1.69",
]

DEV_DEPENDENCIES = [
    "pytest>=6",
    "pytest-xdist>=2",
    "coverage>=5",
    "pytest-cov>=2.10.1",
    "networkx~=2.4",
    "flaky>=3.7",
    "pre-commit",
    "black==21.6b0",
    "flakehell>=0.9",
]

setup(
    name="numba4jax",
    author="Filippo Vicentini",
    url="http://github.com/PhilipVinc/numba4jax",
    author_email="filippovicentini@gmail.com",
    license="MIT",
    description="Usa numba in jax-compiled kernels.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
    ],
    packages=find_packages(),
    install_requires=BASE_DEPENDENCIES,
    python_requires=">=3.7",
    extras_require={"dev": DEV_DEPENDENCIES},
)
