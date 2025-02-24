# Assessing LLMs to Improve the Prediction of COVID-19 Status
Official website: <a href="https://ramosrenzo.github.io/COVID-LLM/">Assessing LLMs to Improve the Prediction of COVID-19 Status</a>

## Clone the Repository
```python
git clone https://github.com/ramosrenzo/COVID-LLM.git
```

## DNABERT, DNABERT-2, GROVER

Note: DNABERT-2 will require a GPU to run.

### Setup Environment

Create and activate a virtual python environment:

<<<<<<< HEAD
```python
conda create -n covid_llms python=3.8
conda activate covid_llms
```

Install required packages:

```python
python -m pip install -r requirements.txt
```

Please ensure that the `triton` package is not installed in your environment:

```python
pip uninstall triton
```

### Run Model

Specify which model(s) to run with a target:

`all` runs DNABERT, DNABERT-2, and GROVER.

`dnabert` runs DNABERT model.

`dnabert-2` runs DNABERT-2 model.

`grover` runs GROVER model.
<br />
<br />
Run the build script with one or multiple targets:

```python
python run.py <target>
```

## AAM

### Setup Environment

Note: Ran on 64GB of CPU memory and on a NVIDIA 2080ti and ran on Linux Virtual Machine

Create and activate a virtual python environment:

=======
>>>>>>> b1b428ee88abc243b6fb151e97710a73e86c46a3
```
conda create --name aam -c conda-forge -c bioconda unifrac python=3.9 cython

conda activate aam

conda install -c conda-forge gxx_linux-64 hdf5 mkl-include lz4 hdf5-static libcblas liblapacke make
```

<<<<<<< HEAD
Install AAM:
=======
## Installing AAM
>>>>>>> b1b428ee88abc243b6fb151e97710a73e86c46a3

```
pip install git+https://github.com/kwcantrell/attention-all-microbes.git@s2s-updated
```

<<<<<<< HEAD
### Run Model

Run the build script:

```python
python run_aam.py
```
=======
## Running Model

#### Training the models
```
python run.py
```
>>>>>>> b1b428ee88abc243b6fb151e97710a73e86c46a3
