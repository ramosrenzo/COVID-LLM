import sys
from AAM.model import run_model
from data_preprocessing import preprocess_data

if __name__ == "__main__":
    try:
        preprocess_data()
        run_model()
    except Exception as e:
        print(f"An error occurred during the process: {e}")