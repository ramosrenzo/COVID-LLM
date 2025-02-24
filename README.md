# Assessing LLMs to Improve the Prediction of COVID-19 Status
Official website: <a href="https://ramosrenzo.github.io/COVID-LLM/">Assessing LLMs to Improve the Prediction of COVID-19 Status</a>

## Clone the Repository
```python
git clone https://github.com/ramosrenzo/COVID-LLM.git
```

## DNABERT, DNABERT-2, GROVER

Note: Ran on 64GB of CPU memory and on a NVIDIA 2080ti and ran on Linux Virtual Machine

### Setup Environment

Create and activate a virtual python environment:

```
conda create --name covid_llms -c conda-forge -c bioconda unifrac python=3.9 cython

conda activate covid_llms
```

## Install AAM

```
pip install git+https://github.com/kwcantrell/attention-all-microbes.git@sequence-regressor

pip install -r requirements.txt
```

## Run Model
```
python run_dnabert_2.py
```