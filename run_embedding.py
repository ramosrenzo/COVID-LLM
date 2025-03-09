from DNABERT_2.asv_embedding import asv_embedding

import sys

if __name__ == "__main__":
    try:
        asv_embedding()
    except Exception as e:
        print(f"An error occurred during the process: {e}")