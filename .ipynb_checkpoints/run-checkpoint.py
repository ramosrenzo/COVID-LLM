from dnabert.model import run_dnabert
from dnabert_2.model import run_dnabert_2

if __name__ == "__main__":
    try:
        run_dnabert()
        run_dnabert_2()
    except Exception as e:
        print(f"An error occurred during the process: {e}")