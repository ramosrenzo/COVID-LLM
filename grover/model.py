import pandas as pd
import numpy as np
import torch
import warnings
import tqdm
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings('ignore')

def run_model():
    # for GPU usage (if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("PoetschLab/GROVER")
    model = AutoModel.from_pretrained("PoetschLab/GROVER").to(device)
    print("Model successfully loaded.")

    dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
    inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"].to(device)
    hidden_states = model(inputs)[0]
    print("Hidden states successfully loaded.")

    embedding_mean = torch.mean(hidden_states[0], dim=0)
    print("Embeddings successfully loaded.")
    print(embedding_mean.shape)
