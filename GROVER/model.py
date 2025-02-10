import pandas as pd
import numpy as np
import torch
import gc
import warnings
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, logging

warnings.filterwarnings('ignore')
logging.set_verbosity_error()

def run_grover():
    # for GPU usage (if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using: {device}")

    def clean_gpu():
        torch.cuda.empty_cache()
        gc.collect()
    clean_gpu()

    # initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("PoetschLab/GROVER")
    model = AutoModel.from_pretrained("PoetschLab/GROVER").to(device)
    print("GROVER: Model successfully loaded.")

    # data usage
    data_df = pd.read_csv('data/fecal/fecal_by_sample.csv')
    data = data_df.drop(columns=['sample']).to_numpy()
    print("GROVER: Data successfully loaded.")

    # get dna embeddings with mean pooling
    def calc_embedding_mean(seq):
        inputs = tokenizer(seq, return_tensors = 'pt', padding=True, truncation=True)["input_ids"].to(device)
        hidden_states = model(inputs)[0]

        # embedding with mean pooling
        embedding_mean = torch.mean(hidden_states[0], dim=0)

        return embedding_mean
    
    embeddings_list = []

    for sample in tqdm(data):
        sample_embeddings = []
        for seq in tqdm(sample):
            sample_embeddings.append(calc_embedding_mean(seq).detach().cpu().numpy())
        embeddings_list.append(sample_embeddings)
    embeddings = torch.tensor(embeddings_list)

    print("GROVER: Embeddings successfully loaded.")
    print("GROVER: Completed.")
