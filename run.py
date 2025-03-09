from AAM.model import train_model, test_model, create_figures
import sys

if __name__ == "__main__":
    try:
        run_type = sys.argv[1]
        if run_type not in ['training', 'testing', 'all']:
            raise Exception('Incorrect run type format')
        elif run_type in ['training','all']:
            train_model(train_fp ='data/input/training_metadata_forehead.tsv', large=False, opt_type='adam', hidden_dim=256, num_hidden_layers=10, dropout_rate=0, learning_rate=0.0001, use_cova=False, beta_1=0.9, beta_2=0.99, weight_decay=0.0001 )
        
            train_model(train_fp ='data/input/training_metadata_inside_floor.tsv', large=False, opt_type='adam', hidden_dim=256, num_hidden_layers=10, dropout_rate=0.1, learning_rate=0.0001, use_cova=False, beta_1=0.9, beta_2=0.999, weight_decay=0.0001)
            
            train_model(train_fp ='data/input/training_metadata_nares.tsv', large=False, opt_type='adam', hidden_dim=256, num_hidden_layers=10, dropout_rate=0.1, learning_rate=0.001, use_cova=False, beta_1=0.5, beta_2=0.999, weight_decay=0.0001)
            
            train_model(train_fp ='data/input/training_metadata_stool.tsv', large=False, opt_type='adam', hidden_dim=256, num_hidden_layers=5, dropout_rate=0, learning_rate=0.001, use_cova=False, beta_1=0.5, beta_2=0.999, weight_decay=0.001)

        elif run_type in ['testing', 'all']:
            forehead_predictions_small = test_model('data/input/test_metadata_forehead.tsv', model_fp='trained_models_aam/forehead')
            inside_floor_predictions_small = test_model('data/input/test_metadata_inside_floor.tsv', model_fp='trained_models_aam/inside_floor')
            stool_predictions_small = test_model('data/input/test_metadata_stool.tsv', model_fp='trained_models_aam/stool')
            nares_predictions_small = test_model('data/input/test_metadata_nares.tsv', model_fp='trained_models_aam/nares')
            
            create_figures(forehead_predictions_small, inside_floor_predictions_small, nares_predictions_small, stool_predictions_small)

    except Exception as e:
        print(f"An error occurred during the process: {e}")