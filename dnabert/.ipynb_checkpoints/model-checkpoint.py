import torch
from transformers import AutoTokenizer, AutoModel, BertConfig, logging
import gc
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
logging.set_verbosity_error()

def run_dnabert():
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def clean_gpu():
        torch.cuda.empty_cache()
        gc.collect()
    clean_gpu()

    # initialize tokenizer and model
    config = BertConfig.from_pretrained('https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/config.json')
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNA_bert_6", config=config, trust_remote_code=True).to(device)
    print("DNABERT: Model successfully loaded.")

    # get dna embeddings
    dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
    inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"].to(device)
    hidden_states = model(inputs)[0]
    print("DNABERT: Hidden states successfully loaded.")

    embedding_mean = torch.mean(hidden_states[0], dim=0)
    print(f"DNABERT: Embeddings successfully loaded.")

    model_input = tokenizer.encode_plus(dna, add_special_tokens=True, max_length=512)["input_ids"]
    model_input = torch.tensor(model_input, dtype=torch.long).to(device)
    model_input = model_input.unsqueeze(0)   # to generate a fake batch with batch size one
    
    output = model(model_input)
    print(output[1])
    print(f"DNABERT: Complete.")