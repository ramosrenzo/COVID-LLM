import pandas as pd
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings('ignore')

def run_model():
    tokenizer = AutoTokenizer.from_pretrained("PoetschLab/GROVER")
    model = AutoModel.from_pretrained("PoetschLab/GROVER")
    print("Model successfully loaded.")

    dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
    inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
    hidden_states = model(inputs)[0]
    print("Hidden states successfully loaded.")

    embedding_mean = torch.mean(hidden_states[0], dim=0)
    print("Embeddings successfully loaded.")
    print(embedding_mean.shape)