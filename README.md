# Assessing LLMs to Improve the Prediction of COVID-19 Status
## Setup Environment
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

## Run Model
Run the build script with one or multiple targets:
```python
python run.py [target 1] [target 2]
```

**Targets**

`all`: runs all models. Note that DNABERT-2 requires a GPU.

`dnabert`: runs DNABERT model.

`dnabert-2`: runs DNABERT-2 model. Note that DNABERT-2 requires a GPU.

`grover`: runs grover model.