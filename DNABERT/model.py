import torch
from transformers import AutoTokenizer, AutoModel, BertConfig, logging
import gc
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

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
    config = BertConfig.from_pretrained('https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/config.json', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True, config=config).to(device)
    print("DNABERT: Model successfully loaded.")

    # get data
    data_df = pd.read_csv('data/fecal/fecal_by_sample.csv')
    data = data_df.drop(columns=['sample']).to_numpy()
    print("DNABERT: Data successfully loaded.")

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
    print(f"DNABERT: Embeddings successfully loaded.")
    print(f"DNABERT: Completed.")