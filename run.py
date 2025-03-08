from DNABERT_2.asv_embedding import asv_embedding
from DNABERT_2.train import train_model
from DNABERT_2.test import test_model
from DNABERT_2.plot_auroc_and_auprc import plot_auroc_auprc
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

        # get embeddings from DNABERT-2
        asv_embedding()
    
        # train model
        train_model(train_fp, opt_type, hidden_dim, num_hidden_layers, dropout_rate, learning_rate, beta_1=beta_1, beta_2=beta_2, weight_decay=weight_decay, momentum=momentum, model_fp=None, large=large)

        # test model
        forehead_predictions = test_model('data/input/test_metadata_forehead.tsv', model_fp='DNABERT_2/trained_models_dnabert_2/forehead/forehead_best_model.keras')
        inside_floor_predictions = test_model('data/input/test_metadata_inside_floor.tsv', model_fp='DNABERT_2/trained_models_dnabert_2/inside_floor/inside_floor_best_model.keras')
        stool_predictions = test_model('data/input/test_metadata_stool.tsv', model_fp='DNABERT_2/trained_models_dnabert_2/stool/stool_best_model.keras')
        nares_predictions = test_model('data/input/test_metadata_nares.tsv', model_fp='DNABERT_2/trained_models_dnabert_2/nares/nares_best_model.keras')

        # plot auroc and auprc
        sample_data = {
            "Nares": nares_predictions[0][:2],
            "Forehead": forehead_predictions[0][:2],
            "Stool": stool_predictions[0][:2],
            "Inside floor": inside_floor_predictions[0][:2],
        }
        plot_auroc_auprc()
        
    except Exception as e:
        print(f"An error occurred during the process: {e}")