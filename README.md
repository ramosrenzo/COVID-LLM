# Assessing LLMs to Improve the Prediction of COVID-19 Status
## DNABERT, DNABERT-2, GROVER
Note that DNABERT-2 will require a GPU to run.

### Setup Environment
Create and activate a virtual python environment:
```python
conda create -n covid_llms python=3.8
conda activate covid_llms
```

Install required packages:
```python
python3 -m pip install -r requirements.txt
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
Create and activate a virtual python environment:
```
conda create --name aam -c conda-forge -c bioconda unifrac python=3.9 cython

conda activate aam

conda install -c conda-forge gxx_linux-64 hdf5 mkl-include lz4 hdf5-static libcblas liblapacke make
```

Install AAM:
```
pip install git+https://github.com/kwcantrell/attention-all-microbes.git@s2s-updated
```

### Run Model
Run the build script:
```python
python run_aam.py
```
