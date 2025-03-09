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
Note: Ran on 64GB of CPU memory and on a NVIDIA 2080ti and on Linux Virtual Machine.

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
The build script `run_embedding.py` generates and stores embeddings from DNABERT-2 in the `data/input` folder. Run it with:

```python
python run_embedding.py
```
The build script `run.py` handles training, testing, and plotting of AUROC and AUPRC scores. Use a target argument to specify which part of the pipeline to execute:

- `all` - Runs training, testing, and plotting. If your system runs out of memory during testing, consider running the `test` target separately.

- `training` – Runs only the training process.

- `test` – Runs testing and plots AUROC and AUPRC.

Run the build script with one target:

```python
python run.py <target>
```
