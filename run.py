from DNABERT_2.training import train_model
from DNABERT_2.test import test_model
from DNABERT_2.plot_auroc_auprc import plot_auroc_auprc

import sys

if __name__ == "__main__":
    try:
        target = sys.argv[1]
        if target not in ['training', 'test', 'all']:
            raise Exception("Incorrect target name. Available targets: 'embedding', 'training', 'test', and 'all'")
        if target in ['training','all']:
            train_model(train_fp ='data/input/training_metadata_forehead.tsv', large=False, opt_type='adam', hidden_dim=768, num_hidden_layers=10, dropout_rate=0, learning_rate=0.0001, use_cova=False, beta_1=0.9, beta_2=0.99, weight_decay=0.0001 )

            train_model(train_fp ='data/input/training_metadata_inside_floor.tsv', large=False, opt_type='adam', hidden_dim=768, num_hidden_layers=10, dropout_rate=0.1, learning_rate=0.0001, use_cova=False, beta_1=0.9, beta_2=0.999, weight_decay=0.0001)
    
            train_model(train_fp ='data/input/training_metadata_nares.tsv', large=False, opt_type='adam', hidden_dim=768, num_hidden_layers=10, dropout_rate=0.1, learning_rate=0.001, use_cova=False, beta_1=0.5, beta_2=0.999, weight_decay=0.0001)
    
            train_model(train_fp ='data/input/training_metadata_stool.tsv', large=False, opt_type='adam', hidden_dim=768, num_hidden_layers=5, dropout_rate=0, learning_rate=0.001, use_cova=False, beta_1=0.5, beta_2=0.999, weight_decay=0.001)
        if target in ['test','all']:
            forehead_predictions = test_model('data/input/test_metadata_forehead.tsv', model_fp='trained_models_dnabert_2/forehead/forehead_best_model.keras')
            inside_floor_predictions = test_model('data/input/test_metadata_inside_floor.tsv', model_fp='trained_models_dnabert_2/inside_floor/inside_floor_best_model.keras')
            stool_predictions = test_model('data/input/test_metadata_stool.tsv', model_fp='trained_models_dnabert_2/stool/stool_best_model.keras')
            nares_predictions = test_model('data/input/test_metadata_nares.tsv', model_fp='trained_models_dnabert_2/nares/nares_best_model.keras')
            
            sample_data = {
                "Nares": nares_predictions[0],
                "Forehead": forehead_predictions[0],
                "Stool": stool_predictions[0],
                "Inside floor": inside_floor_predictions[0],
            }
            plot_auroc_auprc(sample_data)
    
    except Exception as e:
        print(f"An error occurred during the process: {e}")