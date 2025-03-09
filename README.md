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
Note: Ran on 64GB of CPU memory and on a NVIDIA 2080ti and on Linux Virtual Machine.

The build script `run_embedding.py` stores embeddings from DNABERT-2 in the `data/input` folder:

```python
python run_embedding.py
```

The build script`run.py` runs the training, testing, and plotting for AUROC and AUPRC.
Specify which part of the pipeline to run with a target:

`all` runs the training, test, and plotting for AUROC and AUPRC. If your system runs out of memory during the test, then run the `test` target on its own.

`training` runs the training.

`test` runs the test and plotting for AUROC and AUPRC.

<br />
<br />
Run the build script with one target:
```python
python run.py <target>
```