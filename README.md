# Assessing LLMs to Improve the Prediction of COVID-19 Status
Official website: <a href="https://ramosrenzo.github.io/COVID-LLM/">Assessing LLMs to Improve the Prediction of COVID-19 Status</a>

## Clone the Repository
The large language models (LLMs) was run on a Linux Virtual Machine with 64GB of CPU memory and a NVIDIA 2080 Ti GPU. We used <a href="https://git-lfs.com/" target="_blank" rel="noopener noreferrer">Git Large File Storage (LFS)</a> to upload our embeddings from AAM, DNABERT, DNABERT-2, and GROVER, along with our trained Keras classifiers. Please ensure Git LFS is installed. Alternatively, you can preprocess the data and generate embeddings locally using the `run_data.py` script.

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

conda install -c conda-forge gxx_linux-64 hdf5 mkl-include lz4 hdf5-static libcblas liblapacke make
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

To run our Jupyter notebooks, use the following commands to add the `covid_llms` Conda environment:

```python
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=covid_llms
```

### Run Data Preprocessing and Get Embeddings
The build script `run_data.py` handles data preprocessing and generates embeddings from the LLMs. If Git LFS is not installed, then this script is necessary to run before moving on to the classifer stage. Data is stored in the `data/input` folder. Use a target argument to specify which LLM to run:

- `dnabert` - Runs DNABERT.

- `dnabert-2` – Runs DNABERT-2.

- `grover` – Runs GROVER.

Use a second target argument to specify which stage of the pipeline to execute:

- `all` - Preprocesses sample data and generates embeddings.

- `samples` – Preprocesses sample data.

- `embedding` – Generates embeddings.

Run the build script with two targets:

```python
python run_data.py <target_1> <target_2>
```

### Run Classifier
The build script `run.py` handles training, testing, and plotting of AUROC and AUPRC scores for COVID-19 status classification ("Positve" or "Not detected"). Trained classifiers are stored in the `trained_models_<LLM>` folder and plots are stored in the `figures` folder. Use a target argument to specify which LLM's embeddings to use for classification:
- `aam` - Uses AAM embeddings.

- `dnabert` - Uses DNABERT embeddings.

- `dnabert-2` – Uses DNABERT-2 embeddings.

- `grover` – Uses GROVER embeddings.

Use a second target argument to specify which stage of the pipeline to execute:

- `all` - Runs training, testing, and plotting. If your system runs out of memory during testing, consider running the test target separately.

- `train` – Runs the training process.

- `test` – Runs testing and plots AUROC and AUPRC scores.

Run the build script with two targets:

```python
python run.py <target-1> <target-2>
```
