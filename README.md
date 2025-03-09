# Assessing LLMs to Improve the Prediction of COVID-19 Status
Official website: <a href="https://ramosrenzo.github.io/COVID-LLM/">Assessing LLMs to Improve the Prediction of COVID-19 Status</a>

## Clone the Repository
Our work was run on a Linux Virtual Machine with 64GB of CPU memory and a NVIDIA 2080 Ti GPU. We used <a href="https://git-lfs.com/" target="_blank" rel="noopener noreferrer">Git Large File Storage (LFS)</a> to upload our AAM, DNABERT, DNABERT-2, and GROVER embeddings, along with trained Keras models. Please ensure Git LFS is installed. Alternatively, you can preprocess the data and generate embeddings locally using the `run_data.py` script.

Clone the repository:
```python
git clone https://github.com/ramosrenzo/COVID-LLM.git
cd COVID-LLM
```

## AAM, DNABERT, DNABERT-2, and GROVER

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
The build script `run_data.py` handles data preprocessing and generates embeddings from the large language models. Data is stored in the `data/input` folder. Use a target argument to specify which model to run:
- `dnabert` - Runs DNABERT.

- `dnabert-2` – Runs DNABERT-2.

- `grover` – Runs GROVER.

Use a second target argument to specify which stage of the pipeline to execute:

- `all` - Preprocesses sample data and generates embeddings from DNABERT-2.

- `samples` – Preprocesses sample data.

- `embedding` – Generates embeddings from DNABERT-2.

Run the build script with two targets:

```python
python run_data.py <target_1> <target_2>
```

### Run Classifier
The build script `run.py` handles training, testing, and plotting of AUROC and AUPRC scores. Trained models are stored in the `trained_models_<model>` folder and plots are stored in the `figures` folder. Use a target argument to specify which model to run:
- `aam` - Runs AAM.

- `dnabert` - Runs DNABERT.

- `dnabert-2` – Runs DNABERT-2.

- `grover` – Runs GROVER.

Use a second target argument to specify which stage of the pipeline to execute:

- `all` - Runs training, testing, and plotting. If your system runs out of memory during testing, consider running the test target separately.

- `training` – Runs the training process.

- `test` – Runs testing and plots AUROC and AUPRC.

Run the build script with two targets:

```python
python run.py <target_1> <target_2>
```