import sys
from DNABERT.model import run_dnabert
from DNABERT_2.model import run_dnabert_2
from GROVER.model import run_grover
from src.data_preprocessing import preprocess_data

if __name__ == "__main__":
    try:
        args = sys.argv[1:]

        preprocess_data()

        if 'dnabert' in args or 'all' in args:
            run_dnabert()
        if 'dnabert-2' in args or 'all' in args:
            run_dnabert_2()
        if 'grover' in args or 'all' in args:
            run_grover()
    except Exception as e:
        print(f"An error occurred during the process: {e}")