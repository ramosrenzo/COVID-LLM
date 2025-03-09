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

Please ensure that the `triton` package is not installed in your environment, as it may cause errors when running DNABERT-2:

```python
pip uninstall triton
```

### Run Data Preprocessing and Get Embeddings
The build script `run_data.py` handles sample data preprocessing and generates embeddings from DNABERT-2. Data is stored in the `data/input` folder. Use a target argument to specify which stage of the pipeline to execute:
- `all` - Preprocesses sample data and generates embeddings from DNABERT-2.

- `samples` – Preprocesses sample data.

- `embedding` – Generates embeddings from DNABERT-2.

Run the build script with one target:

```python
python run_data.py <target>
```

### Run Classifier
The build script `run.py` handles training, testing, and plotting of AUROC and AUPRC scores. Trained models are stored in the `trained_models_dnabert_2` folder and plots are stored in the `figures` folder. Use a target argument to specify which stage of the pipeline to execute:

- `all` - Runs training, testing, and plotting. If your system runs out of memory during testing, consider running the `test` target separately.

- `training` – Runs the training process.

- `test` – Runs testing and plots AUROC and AUPRC.

Run the build script with one target:

```python
python run.py <target>
```
