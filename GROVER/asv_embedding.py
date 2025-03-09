import torch
from transformers import AutoTokenizer, AutoModel, BertConfig, logging
import numpy as np
from tqdm import tqdm
from biom import load_table, Table

def asv_embedding():
    '''
    Saves ASV embeddings of GROVER as 'asv_embeddings_grover.npy' in 'data/input' folder(s).
    '''

    device = "cuda" if torch.cuda.is_available() else "cpu"

    table = load_table("data/input/merged_biom_table.biom")
    table.ids(axis="observation")

    config = BertConfig.from_pretrained("PoetschLab/GROVER")
    tokenizer = AutoTokenizer.from_pretrained("PoetschLab/GROVER")
    model = AutoModel.from_pretrained("PoetschLab/GROVER", config=config).to(device)

    def calc_embedding_mean(asvs):
        inputs = [tokenizer(asv, return_tensors = 'pt')["input_ids"].to(device) for asv in asvs]
        hidden_states = [np.mean(model(input)[0].cpu().detach().numpy(), axis=1) for input in tqdm(inputs)]
        return np.vstack(hidden_states)
    
    embeddings = calc_embedding_mean(table.ids(axis="observation"))

    np.save("data/input/asv_embeddings_grover.npy", embeddings)
    np.save("data/input/asv_embedding_ids.npy", table.ids(axis="observation"))
    print('GROVER: ASV embeddings and ids saved.')