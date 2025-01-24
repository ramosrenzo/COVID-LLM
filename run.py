from dnabert.model import run_model

if __name__ == "__main__":
    try:
        run_model()
    except Exception as e:
        print(f"An error occurred during the process: {e}")