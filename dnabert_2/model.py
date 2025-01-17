
import torch
from transformers import AutoTokenizer, AutoModel, BertConfig, logging
import gc
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
logging.set_verbosity_error()

def run_model():
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def clean_gpu():
        torch.cuda.empty_cache()
        gc.collect()
    clean_gpu()

    # initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config).to(device)
    print("Model successfully loaded.")

    # get dna embeddings
    dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
    inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"].to(device)
    hidden_states = model(inputs)[0]
    print("Hidden states successfully loaded.")

    embedding_mean = torch.mean(hidden_states[0], dim=0)
    print(f"Embeddings successfully loaded.")
