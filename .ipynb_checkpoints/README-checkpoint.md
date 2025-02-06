# DNABERT and DNABERT-2
## Setup Environment

Create and activate a virtual python environment:
```python
conda create -n dnabert python=3.8
conda activate dnabert
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

**Note**: DNABERT-2 requires a GPU to run.

Run the model on a simple test:

```python
python run.py
```