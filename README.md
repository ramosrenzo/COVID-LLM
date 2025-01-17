# DNABERT-2

## Setup environment

### create and activate virtual python environment
```python
conda create -n dnabert2 python=3.8
conda activate dnabert2
```

### install packages
```python
python3 -m pip install -r requirements.txt
```

### additional note
Ensure that the `triton` package is uninstalled in your virtual environment.
```python
pip uninstall triton
``` 

## Running Code

**Note**: We recommend running DNABERT-2 with a GPU.

To run the model on a simple test:

```python
python run.py
```


