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
Note: DNABERT-2 requires a GPU to run.

Run the build script with a target to run one or multiple models:
```python
python run.py [target 1] [target 2]
```
**Targets**

`all`: runs all models.

`dnabert`: runs DNABERT model.

`dnabert-2`: runs DNABERT-2 model.

`grover`: runs grover model.