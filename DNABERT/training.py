from DNABERT.model import GeneratorDataset
from DNABERT.model import Classifier
import tensorflow as tf
if tf.config.list_physical_devices("GPU"):
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings('ignore')

def get_sample_type(file_path):
    '''
    Returns the name of the sample environment from the training metadata file path.
    '''
    filename = os.path.basename(file_path)
    if filename.startswith('training_metadata_'):
        sample_type = filename[len('training_metadata_'):]
        sample_type = os.path.splitext(sample_type)[0]
        return sample_type
    return "Unknown"


def train_model(train_fp, large, opt_type, hidden_dim, num_hidden_layers, dropout_rate, learning_rate, use_cova=False, beta_1=None, beta_2=None, weight_decay=None):
    training_metadata = pd.read_csv(train_fp, sep='\t', index_col=0)
    X = training_metadata.drop(columns=['study_sample_type', 'has_covid'], axis=1)
    y = training_metadata[['study_sample_type', 'has_covid']]
    sample_type = get_sample_type(train_fp)

    dir_path = f'trained_models_dnabert/{sample_type}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    sequence_embedding_fp = 'data/input/asv_embeddings.npy'
    sequence_labels_fp = 'data/input/asv_embeddings_ids.npy'
    sequence_embedding_dim = 768
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    curr_best_val_loss = np.inf
    curr_best_model = None
    for i, (train_index, valid_index) in enumerate(skf.split(y, y['has_covid'])):
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        if sample_type == 'stool':
            rarefy_depth = 4000
        else:
            rarefy_depth = 1000
    
        gd_train = GeneratorDataset(
            table='data/input/merged_biom_table.biom',
            metadata=y_train,
            metadata_column='has_covid',
            shuffle=True,
            is_categorical=False,
            shift=0,
            rarefy_depth = rarefy_depth,
            scale=1,
            batch_size = 4,
            gen_new_tables = True,
            sequence_embeddings = sequence_embedding_fp,
            sequence_labels = sequence_labels_fp,
            upsample=False,
            drop_remainder=False
        )
    
        gd_valid = GeneratorDataset(
            table='data/input/merged_biom_table.biom',
            metadata=y_valid,
            metadata_column='has_covid',
            shuffle=False,
            is_categorical=False,
            shift=0,
            rarefy_depth = rarefy_depth,
            scale=1,
            batch_size = 4,
            sequence_embeddings = sequence_embedding_fp,
            sequence_labels = sequence_labels_fp,
            upsample=False,
            drop_remainder=False,
            rarefy_seed = 42
        )

        model = Classifier(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate)
        asv_embedding_shape = tf.TensorShape([None, None, sequence_embedding_dim])
        count_shape = tf.TensorShape([None, None, 1])
        model.build([asv_embedding_shape, count_shape])

        if opt_type == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate = 0.0,
                warmup_target = learning_rate,
                warmup_steps=0,
                decay_steps=100_000,
                ),
                use_ema = True,
                beta_1 = beta_1,
                beta_2 = beta_2,
                weight_decay = weight_decay
            )
            early_stop = EarlyStopping(patience=100, start_from_epoch=50, restore_best_weights=False)
        else:
            optimizer = tf.keras.optimizers.legacy.SGD(
                learning_rate=tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate = 0.0,
                warmup_target = learning_rate,
                warmup_steps=0,
                decay_steps=100_000,
                ),
                momentum = momentum
            )
            early_stop = EarlyStopping(patience=100, start_from_epoch=50, restore_best_weights=True)
            
        model.compile(optimizer=optimizer, run_eagerly=False)
        history = model.fit(gd_train, 
                  validation_data = gd_valid, 
                  validation_steps=gd_valid.steps_per_epoch, 
                  epochs=10_000,
                  steps_per_epoch=gd_train.steps_per_epoch, 
                  callbacks=[
                      early_stop
                   ])
        if opt_type == 'adam':
            model.optimizer.finalize_variable_values(model.trainable_variables)

        validation_loss = history.history['val_loss']
        train_loss = history.history['loss']
        epochs = np.array(range(len(validation_loss)))
        
        min_val_loss = np.min(history.history['val_loss'])
        if min_val_loss < curr_best_val_loss:
            curr_best_model = model
            curr_best_val_loss = min_val_loss

        plt.plot(epochs, validation_loss, color='blue')
        plt.title(f'Validation Loss Per Epoch, Best: {curr_best_val_loss} Final: {min_val_loss}')
        plt.plot(epochs, train_loss, color='red')
        plt.savefig(os.path.join(dir_path, f'{sample_type}_{i}_model_loss.png'))
        plt.close()
        model.save(os.path.join(dir_path, f'{sample_type}_{i}_model.keras'), save_format='keras')
    curr_best_model.save(os.path.join(dir_path, f'{sample_type}_best_model.keras'), save_format='keras')
    print(f"\nDNABERT: Best model saved for {get_sample_type(train_fp)} samples.")