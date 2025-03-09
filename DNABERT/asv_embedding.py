import torch
from transformers import AutoTokenizer, AutoModel, BertConfig, logging
import numpy as np
from tqdm import tqdm
from biom import load_table, Table

def asv_embedding():
    '''
    Saves the ASV embeddings of DNABERT to a file in the data folder called 'asv_embeddings.npy'.
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    table = load_table("data/input/merged_biom_table.biom")
    table.ids(axis="observation")
    
    config = BertConfig.from_pretrained('https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-5/config.json', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_5", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNA_bert_5", trust_remote_code=True, config=config).to(device)

    def get_kmer_sentence(original_string, kmer=5, stride=1):
        if kmer == -1:
            return original_string
        sentence = ""
        original_string = original_string.replace("\n", "")
        i = 0
        while i < len(original_string)-kmer:
            sentence += original_string[i:i+kmer] + " "
            i += stride
        return sentence[:-1].strip("\"")
    
    def calc_embedding_mean(asvs):
        inputs = tokenizer(get_kmer_sentence(asvs), return_tensors = 'pt')["input_ids"].to(device)
        hidden_states = model(inputs)[0]
        embedding_mean = hidden_states.mean(axis=1)
        return embedding_mean.cpu().detach()
    
    embeddings = []
    for asv in tqdm(table.ids(axis="observation")):
        embeddings.append(calc_embedding_mean(asv))
    embeddings = np.array(embeddings).reshape((61974, 768))
    
    np.save("data/input/asv_embeddings.npy", embeddings)
    print(f'DNABERT: ASV embeddings saved.')