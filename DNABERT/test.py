from DNABERT.model import GeneratorDataset
import tensorflow as tf
if tf.config.list_physical_devices("GPU"):
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import biom
import os

def get_sample_type(file_path):
    '''
    Returns the name of the sample environment from the test metadata file path.
    '''
    filename = os.path.basename(file_path)
    if filename.startswith('test_metadata_'):
        sample_type = filename[len('test_metadata_'):]
        sample_type = os.path.splitext(sample_type)[0]
        return sample_type
    return "Unknown"

def test_model(test_fp, model_fp, ensemble=False):
    sample_type = get_sample_type(test_fp)
    test_metadata = pd.read_csv(test_fp, sep='\t', index_col=0)
    X_test = test_metadata.drop(columns=['study_sample_type', 'has_covid'], axis=1)
    y_test = test_metadata[['study_sample_type', 'has_covid']]
    
    if sample_type == 'stool':
        rarefy_depth = 4000
    else:
        rarefy_depth = 1000

    sequence_embedding_fp = 'data/input/asv_embeddings.npy'
    sequence_labels_fp = 'data/input/asv_embeddings_ids.npy'

    embed_test = [GeneratorDataset(
        table='data/input/merged_biom_table.biom',
        metadata=y_test,
        metadata_column='has_covid',
        shuffle=False,
        is_categorical=False,
        shift=0,
        rarefy_depth = rarefy_depth,
        scale=1,
        batch_size = 32,
        epochs=1,
        gen_new_tables = False,
        sequence_embeddings = sequence_embedding_fp,
        sequence_labels = sequence_labels_fp,
        upsample=False,
        drop_remainder=False,
        gen_new_table_frequency = 1,
        rarefy_seed = 42 + i
        ) for i in range(69)
    ]
    
    if '.keras' in model_fp:
        model=tf.keras.models.load_model(model_fp, compile=False)
        predictions = [model.predict(ds, steps=ds.steps_per_epoch) for ds in embed_test]
        y_pred, y_true = [], []
        for y_p, y_t, _ in predictions:
            y_pred.append(y_p)
            y_true.append(y_t)
        y_pred = np.hstack(y_pred)
        y_true = np.hstack(y_true)
        auc_score = 0
        return (y_pred, y_true), auc_score
    else:
        models = [tf.keras.models.load_model(f'{model_fp}/{sample_type}_{i}_model.keras', compile=False) for i in range(5)]
        predictions = []
        for model in models:
            predictions.append([model.predict(ds, steps=ds.steps_per_epoch) for ds in embed_test])
        ensemble_y_pred, ensemble_y_true = [], []
        for model_predictions in predictions:
            y_pred, y_true = [], []
            for y_p, y_t, _ in model_predictions:
                y_pred.append(y_p)
                y_true.append(y_t)
            y_pred = np.hstack(y_pred)
            y_true = np.hstack(y_true)
            ensemble_y_pred.append(y_pred)
            ensemble_y_true.append(y_true)
        ensemble_y_pred = np.vstack(ensemble_y_pred).mean(axis=0)
        ensemble_y_true = np.vstack(ensemble_y_true).mean(axis=0)
        auc_score = 0
        return (ensemble_y_pred, ensemble_y_true), auc_score