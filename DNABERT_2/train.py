from DNABERT_2.model import GeneratorDataset
from DNABERT_2.model import Classifier
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
if tf.config.list_physical_devices("GPU"):
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

from aam.models.sequence_regressor import SequenceRegressor
from aam.models.sequence_regressor_v2 import SequenceRegressorV2
from aam.callbacks import SaveModel
from keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
from biom import Table, load_table
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

def train_model(train_fp, opt_type, hidden_dim, num_hidden_layers, dropout_rate, learning_rate, beta_1=None, beta_2=None, weight_decay=None, momentum=None, model_fp=None, large=True, use_cova=False):
    training_metadata = pd.read_csv(train_fp, sep='\t', index_col=0)
    X = training_metadata.drop(columns=['study_sample_type', 'has_covid'], axis=1)
    y = training_metadata[['study_sample_type', 'has_covid']]
    sample_type = get_sample_type(train_fp)
    
    dir_path = f'trained_models/{sample_type}_{opt_type}_{"large" if large else "small"}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    sequence_embedding_fp = 'data/input/asv_embeddings.npy'
    sequence_embedding_dim = 768

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    curr_best_val_loss = np.inf
    curr_best_model = None
    for i, (train_index, valid_index) in enumerate(skf.split(y, y['has_covid'])):
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        if sample_type == 'stool':
            rarefy_depth = 1000
        else:
            rarefy_depth = 1000
        dataset_train = GeneratorDataset(
            table='data/input/merged_biom_table.biom',
            metadata=y_train,
            metadata_column='has_covid',
            shuffle=True,
            is_categorical=False,
            shift=0,
            rarefy_depth = rarefy_depth,
            scale=1,
            epochs=100_000,
            batch_size = 4,
            gen_new_tables = True, #only in training dataset
            sequence_embeddings = sequence_embedding_fp,
            sequence_labels = 'data/input/asv_embeddings_ids.npy',
            upsample=False,
            drop_remainder=False
        )
    
        dataset_valid = GeneratorDataset(
            table='data/input/merged_biom_table.biom',
            metadata=y_valid,
            metadata_column='has_covid',
            shuffle=False,
            is_categorical=False,
            shift=0,
            rarefy_depth = rarefy_depth,
            scale=1,
            epochs=100000,
            batch_size = 4,
            sequence_embeddings = sequence_embedding_fp,
            sequence_labels = 'data/input/asv_embeddings_ids.npy',
            upsample=False,
            drop_remainder=False,
            rarefy_seed = 42
        )

        if model_fp is None:
            model = Classifier(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate, use_cova=use_cova)
        else:
            model = tf.keras.models.load_model(model_fp, compile=False)
        
        token_shape = tf.TensorShape([None, 768])
        batch_indicies = tf.TensorShape([None, 2])
        indicies_shape = tf.TensorShape([None])
        count_shape = tf.TensorShape([None, 1])
        model.build(
             [token_shape, batch_indicies, indicies_shape, count_shape]
        )
        model.summary()
        if opt_type == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate = 0.0,
                warmup_target = learning_rate, # maybe change
                warmup_steps=0,
                decay_steps=250000,
                ),
                use_ema = True,
                beta_1 = beta_1,
                beta_2 = beta_2,
                weight_decay = weight_decay
                )
            early_stop = EarlyStopping(patience=250, start_from_epoch=250, restore_best_weights=False)
        else:
            optimizer = tf.keras.optimizers.legacy.SGD(
                learning_rate=tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate = 0.0,
                warmup_target = learning_rate, # maybe change
                warmup_steps=0,
                decay_steps=250000,
                ),
                momentum = momentum
            )
            early_stop = EarlyStopping(patience=250, start_from_epoch=250, restore_best_weights=True)
    
        model.compile(optimizer=optimizer, run_eagerly=False)
        #switch loss to val loss 
        #pass early stopping for callbacks
        history = model.fit(dataset_train, 
                  validation_data = dataset_valid, 
                  validation_steps=dataset_valid.steps_per_epoch, 
                  epochs=100_000, 
                  steps_per_epoch=dataset_train.steps_per_epoch, 
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
    print(f"\nDNABERT-2: Best model saved for {sample_type} samples.")