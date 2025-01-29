# DNABERT-2

## Setup Environment

Create and activate a virtual python environment:
```python
conda create -n dnabert2 python=3.8
conda activate dnabert2
```

Install required packages:
```python
python3 -m pip install -r requirements.txt
```

Please ensure that the `triton` package is uninstalled in your environment:
```python
pip uninstall triton
``` 

## Running Code

**Note**: DNABERT-2 requires a GPU to run.

To run the model on a simple test:

```python
python run.py
```


