import torch
from transformers import AutoTokenizer, AutoModel, BertConfig, logging
import numpy as np
from tqdm import tqdm
from biom import load_table, Table

def asv_embedding():
    '''
    Saves the ASV embeddings of DNABERT-2 to a file in the data folder called 'asv_embeddings_dnabert_2.npy'.
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    table = load_table("data/input/merged_biom_table.biom")
    table.ids(axis="observation")
    
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config).to(device)
    
    def calc_embedding_mean(asvs):
        inputs = [tokenizer(asv, return_tensors = 'pt')["input_ids"].to(device) for asv in asvs]
        hidden_states = [model(input) for input in inputs]
        embedding_mean = [torch.mean(byte_pair, dim=1) for byte_pair, class_token in hidden_states]
        return torch.concat(embedding_mean, dim=0).cpu().detach().numpy()
    
    embeddings = []
    for asv in tqdm(table.ids(axis="observation")):
        embeddings.append(calc_embedding_mean([asv]))
    embeddings = np.array(embeddings).reshape((61974, 768))
    
    np.save("data/input/asv_embeddings_dnabert_2.npy", embeddings)
    np.save("data/input/asv_embedding_ids.npy", table.ids(axis="observation"))
    print(f'DNABERT-2: ASV embeddings saved.')