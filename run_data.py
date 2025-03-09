from DNABERT_2.asv_embedding import asv_embedding as asv_embedding_dnabert_2
from GROVER.asv_embedding import asv_embedding as asv_embedding_grover
from src.data_preprocessing import preprocess_data

import sys

if __name__ == "__main__":
    try:
        model = sys.argv[1]
        target = sys.argv[2]

        if model not in ['dnabert', 'dnabert-2', 'grover']:
            raise Exception("Incorrect model name. Available models: 'dnabert', 'dnabert-2', and 'grover'")
        if target not in ['samples', 'embedding', 'all']:
            raise Exception("Incorrect target name. Available targets: 'samples', 'embedding', and 'all'")
        
        if target in ['samples', 'all']:
            preprocess_data()
        
        if target in ['embedding', 'all']:
            if model == 'dnabert-2':
                asv_embedding_dnabert_2()
            if model == 'grover'
                asv_embedding_grover()
    except Exception as e:
        print(f"An error occurred during the process: {e}")