import sys
from DNABERT.model import run_dnabert

if __name__ == "__main__":
    try:
        run_dnabert()
    except Exception as e:
        print(f"An error occurred during the process: {e}")