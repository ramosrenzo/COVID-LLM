# Assessing LLMs to Improve the Prediction of COVID-19 Status Using Microbiome Data
Official website: <a href="https://ramosrenzo.github.io/COVID-LLM/">Assessing LLMs to Improve the Prediction of COVID-19 Status</a>

We evaluated the performance of four large language models (LLMs)—DNABERT, DNABERT-2, GROVER and AAM—in predicting COVID-19 status
from microbiome data. These four models were chosen for their
distinct pre-training strategies: DNABERT and GROVER were trained on the
human genome, DNABERT-2 incorporated multi-species genomes, and AAM
was trained on 16S ribosomal RNA (rRNA) sequencing data. We assessed
each model’s performance by using embeddings extracted from
hospital-derived 16S data labeled with COVID-19 status ("Positive" or "Not detected"). For our evaluation metrics, we used AUROC and AUPRC to benchmark.


## Clone the Repository
The LLMs was run on a Linux Virtual Machine with 64GB of CPU memory and a NVIDIA 2080 Ti GPU. We used <a href="https://git-lfs.com/" target="_blank" rel="noopener noreferrer">Git Large File Storage (LFS)</a> to upload our embeddings from DNABERT, DNABERT-2, and GROVER, along with our trained Keras classifiers. Please ensure Git LFS is installed. Alternatively, you can preprocess the data and generate embeddings locally using the `run_data.py` script.

If Git LFS is downloaded, install the package:
```python
git lfs install
```

Clone the repository:
```python
git clone https://github.com/ramosrenzo/COVID-LLM.git
cd COVID-LLM
```

## Running AAM, DNABERT, DNABERT-2, and GROVER

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
The build script `run_data.py` handles data preprocessing and generates embeddings from the LLMs. Preprocessed data and model embeddings are stored in the `data/input` folder. If Git LFS is not installed, then this script is necessary to run for DNABERT, DNABERT-2, and GROVER before moving on to the classifer stage. The embeddings for AAM was uploaded in the standard way so this section is not necessary. Use a target argument to specify which LLM to run:

- `dnabert` - Runs DNABERT.

- `dnabert-2` – Runs DNABERT-2.

- `grover` – Runs GROVER.

Use a second target argument to specify which stage of the pipeline to execute:

- `all` - Preprocesses sample data and generates embeddings.

- `samples` – Preprocesses sample data.

- `embedding` – Generates embeddings.

Run the build script with two targets:

```python
python run_data.py <target-1> <target-2>
```

### Run Classifier
The build script `run.py` handles training, testing, and plotting of AUROC and AUPRC scores for COVID-19 status classification ("Positive" or "Not detected"). Trained classifiers for each LLM are stored in their respective `trained_models_<LLM>` folder. Plots are stored in the `figures` folder. Use a target argument to specify which LLM's embeddings to use for classification:
- `aam` - Uses AAM embeddings.

- `dnabert` - Uses DNABERT embeddings.

- `dnabert-2` – Uses DNABERT-2 embeddings.

- `grover` – Uses GROVER embeddings.

Use a second target argument to specify which stage of the pipeline to execute:

- `all` - Runs training, testing, and plotting. If your system runs out of memory during testing, consider running the test target separately.

- `train` – Runs the training process.

- `test` – Runs the testing process and plots AUROC and AUPRC scores. If Git LFS is not installed, then training must be done locally before testing. 

Run the build script with two targets:

```python
python run.py <target-1> <target-2>
```

## References
**AAM**

Cantrell, Kalen. "Attention All Microbes (AAM)." (2025). https://github.com/kwcantrell/attention-all-microbes

**DNABERT**

Ji, Yanrong, et al. "DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome." Bioinformatics 37.15 (2021): 2112-2120.

**DNABERT-2**

Zhou, Zhihan, et al. "Dnabert-2: Efficient foundation model and benchmark for multi-species genome." arXiv preprint arXiv:2306.15006 (2023).

**GROVER**

Sanabria, Melissa, et al. "DNA language model GROVER learns sequence context in the human genome." Nature Machine Intelligence 6.8 (2024): 911-923.
