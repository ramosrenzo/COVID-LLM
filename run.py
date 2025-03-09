from DNABERT.model import train_model
import sys

if __name__ == "__main__":
    try:
        train_fp = sys.argv[1]
        large = sys.argv[2].lower() == 'true'
        opt_type = sys.argv[3]
        hidden_dim = int(sys.argv[4])
        num_hidden_layers = int(sys.argv[5])
        dropout_rate = float(sys.argv[6])
        learning_rate = float(sys.argv[7])
        beta_1 = None
        beta_2 = None
        weight_decay = None
        momentum = None
        if opt_type == 'adam':
            beta_1 = float(sys.argv[8])
            beta_2 = float(sys.argv[9])
            weight_decay = float(sys.argv[10])
        else:
            momentum = float(sys.argv[8])
        train_model(train_fp, opt_type, hidden_dim, num_hidden_layers, dropout_rate, learning_rate, beta_1=beta_1, beta_2=beta_2, weight_decay=weight_decay, momentum=momentum, model_fp=None, large=large)
    except Exception as e:
        print(f"An error occurred during the process: {e}")