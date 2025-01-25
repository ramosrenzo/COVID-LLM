# AAM (Attention All Microbes)

## Setup Environment

Create and activate a virtual python environment:

```
conda create --name aam -c conda-forge -c bioconda unifrac python=3.9 cython

conda activate aam

conda install -c conda-forge gxx_linux-64 hdf5 mkl-include lz4 hdf5-static libcblas liblapacke make

```

## Installing AAM

```

pip install git+https://github.com/kwcantrell/attention-all-microbes.git@s2s-updated

```
