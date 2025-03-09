from DNABERT.asv_embedding import asv_embedding
from src.data_preprocessing import preprocess_data
import sys

if __name__ == "__main__":
    try:
        target = sys.argv[1]
        if target not in ['samples', 'embedding', 'all']:
            raise Exception("Incorrect target name. Available targets: 'samples', 'embedding', and 'all'")
        if target in ['samples', 'all']:
            preprocess_data()
        if target in ['embedding', 'all']:
            asv_embedding()
    except Exception as e:
        print(f"An error occurred during the process: {e}")