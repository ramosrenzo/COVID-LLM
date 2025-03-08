# Assessing LLMs to Improve the Prediction of COVID-19 Status
Official website: <a href="https://ramosrenzo.github.io/COVID-LLM/">Assessing LLMs to Improve the Prediction of COVID-19 Status</a>

## Clone the Repository
```python
git clone https://github.com/ramosrenzo/COVID-LLM.git
```

## Checkout DNABERT-2 branch
```python
cd COVID-LLM

git checkout dnabert-2
```

## DNABERT-2

### Setup Environment

Create and activate a virtual python environment:

```python
conda create --name covid_llms -c conda-forge -c bioconda unifrac python=3.9 cython

conda activate covid_llms
```

Install required packages:

```python
pip install git+https://github.com/kwcantrell/attention-all-microbes.git@capstone-2025

python -m pip install -r requirements.txt
```

Please ensure that the `triton` package is not installed in your environment:

```python
pip uninstall triton
```

### Run Model

Run the build script:

```python
python run.py
```