import torch
from transformers import AutoTokenizer, AutoModel, BertConfig, logging
import gc
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.set_verbosity_error()

def run_model():
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Running with {device}.')
    
    def clean_gpu():
        torch.cuda.empty_cache()
        gc.collect()
    clean_gpu()

    # initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config).to(device)
    print("Model successfully loaded.")

    # get data
    data_df = pd.read_csv('data/fecal_by_sample.csv')
    data = data_df.drop(columns=['sample']).to_numpy()
    print("Data successfully loaded.")

    # get dna embeddings with mean pooling
    def calc_embedding_mean(seq):
        inputs = tokenizer(seq, return_tensors = 'pt')["input_ids"].to(device)
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
    print(f"Embeddings successfully loaded.")