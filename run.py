from AAM.training import train_model as train_model_aam
from AAM.test import test_model as test_model_aam
from AAM.plot_auroc_auprc import plot_auroc_auprc as plot_auroc_auprc_aam

from DNABERT_2.training import train_model as train_model_dnabert_2
from DNABERT_2.test import test_model as test_model_dnabert_2
from DNABERT_2.plot_auroc_auprc import plot_auroc_auprc as plot_auroc_auprc_dnabert_2
 
from GROVER.training import train_model as train_model_grover
from GROVER.test import test_model as test_model_grover
from GROVER.plot_auroc_auprc import plot_auroc_auprc as plot_auroc_auprc_grover

import sys

if __name__ == "__main__":
    try:
        model = sys.argv[1]
        target = sys.argv[2]
        
        if model not in ['aam', 'dnabert', 'dnabert-2', 'grover']:
            raise Exception("Incorrect model name. Available models: 'aam', 'dnabert', 'dnabert-2', and 'grover'")
        elif target not in ['training', 'test', 'all']:
            raise Exception("Incorrect target name. Available targets: 'training', 'test', and 'all'")
        
        if model == 'aam':
            if target in ['training','all']:
                train_model_aam(train_fp ='data/input/training_metadata_forehead.tsv', large=False, opt_type='adam', hidden_dim=256, num_hidden_layers=10, dropout_rate=0, learning_rate=0.0001, use_cova=False, beta_1=0.9, beta_2=0.99, weight_decay=0.0001 )
                train_model_aam(train_fp ='data/input/training_metadata_inside_floor.tsv', large=False, opt_type='adam', hidden_dim=256, num_hidden_layers=10, dropout_rate=0.1, learning_rate=0.0001, use_cova=False, beta_1=0.9, beta_2=0.999, weight_decay=0.0001)
                train_model_aam(train_fp ='data/input/training_metadata_nares.tsv', large=False, opt_type='adam', hidden_dim=256, num_hidden_layers=10, dropout_rate=0.1, learning_rate=0.001, use_cova=False, beta_1=0.5, beta_2=0.999, weight_decay=0.0001)
                train_model_aam(train_fp ='data/input/training_metadata_stool.tsv', large=False, opt_type='adam', hidden_dim=256, num_hidden_layers=5, dropout_rate=0, learning_rate=0.001, use_cova=False, beta_1=0.5, beta_2=0.999, weight_decay=0.001)
            if target in ['test','all']:
                forehead_predictions = test_model_aam('data/input/test_metadata_forehead.tsv', model_fp=f'trained_models_aam/forehead/forehead_best_model.keras')
                inside_floor_predictions = test_model_aam('data/input/test_metadata_inside_floor.tsv', model_fp=f'trained_models_aam/inside_floor/inside_floor_best_model.keras')
                stool_predictions = test_model_aam('data/input/test_metadata_stool.tsv', model_fp=f'trained_models_aam/stool/stool_best_model.keras')
                nares_predictions = test_model_aam('data/input/test_metadata_nares.tsv', model_fp=f'trained_models_aam/nares/nares_best_model.keras')
                plot_auroc_auprc_aam(nares_predictions, forehead_predictions, stool_predictions, inside_floor_predictions)
        
        if model == 'dnabert-2':
            if target in ['training','all']:
                train_model_dnabert_2(train_fp ='data/input/training_metadata_forehead.tsv', large=False, opt_type='adam', hidden_dim=768, num_hidden_layers=10, dropout_rate=0, learning_rate=0.0001, use_cova=False, beta_1=0.9, beta_2=0.99, weight_decay=0.0001 )
                train_model_dnabert_2(train_fp ='data/input/training_metadata_inside_floor.tsv', large=False, opt_type='adam', hidden_dim=768, num_hidden_layers=10, dropout_rate=0.1, learning_rate=0.0001, use_cova=False, beta_1=0.9, beta_2=0.999, weight_decay=0.0001)
                train_model_dnabert_2(train_fp ='data/input/training_metadata_nares.tsv', large=False, opt_type='adam', hidden_dim=768, num_hidden_layers=10, dropout_rate=0.1, learning_rate=0.001, use_cova=False, beta_1=0.5, beta_2=0.999, weight_decay=0.0001)
                train_model_dnabert_2(train_fp ='data/input/training_metadata_stool.tsv', large=False, opt_type='adam', hidden_dim=768, num_hidden_layers=5, dropout_rate=0, learning_rate=0.001, use_cova=False, beta_1=0.5, beta_2=0.999, weight_decay=0.001)        
            if target in ['test','all']:
                forehead_predictions = test_model_dnabert_2('data/input/test_metadata_forehead.tsv', model_fp=f'trained_models_dnabert_2/forehead/forehead_best_model.keras')
                inside_floor_predictions = test_model_dnabert_2('data/input/test_metadata_inside_floor.tsv', model_fp=f'trained_models_dnabert_2/inside_floor/inside_floor_best_model.keras')
                stool_predictions = test_model_dnabert_2('data/input/test_metadata_stool.tsv', model_fp=f'trained_models_dnabert_2/stool/stool_best_model.keras')
                nares_predictions = test_model_dnabert_2('data/input/test_metadata_nares.tsv', model_fp=f'trained_models_dnabert_2/nares/nares_best_model.keras')
                plot_auroc_auprc_dnabert_2(nares_predictions, forehead_predictions, stool_predictions, inside_floor_predictions)
        
        if model == 'grover':
            if target in ['training','all']:
                train_model_grover(train_fp ='data/input/training_metadata_forehead.tsv', large=False, opt_type='adam', hidden_dim=768, num_hidden_layers=10, dropout_rate=0, learning_rate=0.0001, use_cova=False, beta_1=0.9, beta_2=0.99, weight_decay=0.0001 )
                train_model_grover(train_fp ='data/input/training_metadata_inside_floor.tsv', large=False, opt_type='adam', hidden_dim=768, num_hidden_layers=10, dropout_rate=0.1, learning_rate=0.0001, use_cova=False, beta_1=0.9, beta_2=0.999, weight_decay=0.0001)
                train_model_grover(train_fp ='data/input/training_metadata_nares.tsv', large=False, opt_type='adam', hidden_dim=768, num_hidden_layers=10, dropout_rate=0.1, learning_rate=0.001, use_cova=False, beta_1=0.5, beta_2=0.999, weight_decay=0.0001)
                train_model_grover(train_fp ='data/input/training_metadata_stool.tsv', large=False, opt_type='adam', hidden_dim=768, num_hidden_layers=5, dropout_rate=0, learning_rate=0.001, use_cova=False, beta_1=0.5, beta_2=0.999, weight_decay=0.001)

            if target in ['test','all']:
                forehead_predictions = test_model_grover('data/input/test_metadata_forehead.tsv', model_fp='trained_models_grover/forehead/forehead_best_model.keras')
                inside_floor_predictions = test_model_grover('data/input/test_metadata_inside_floor.tsv', model_fp='trained_models_grover/inside_floor/inside_floor_best_model.keras')
                stool_predictions = test_model_grover('data/input/test_metadata_stool.tsv', model_fp='trained_models_grover/stool/stool_best_model.keras')
                nares_predictions = test_model_grover('data/input/test_metadata_nares.tsv', model_fp='trained_models_grover/nares/nares_best_model.keras')
                plot_auroc_auprc_grover(nares_predictions, forehead_predictions, stool_predictions, inside_floor_predictions)

    except Exception as e:
        print(f"An error occurred during the process: {e}")