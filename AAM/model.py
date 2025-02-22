from aam.data_handlers.generator_dataset import GeneratorDataset
from aam.models.transformers import TransformerEncoder
from aam.models.multihead_attention_pooling import MultiHeadAttentionPooling
from aam.models.sequence_regressor import SequenceRegressor
from aam.callbacks import SaveModel
from keras.callbacks import EarlyStopping

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import biom
import os
import warnings

warnings.filterwarnings('ignore')

K = tf.keras

class Classifier(K.Model):
    def __init__(self, feature_extractor, **kwargs):
        super().__init__(**kwargs)

        self.feature_extractor = feature_extractor
        self.feature_extractor.trainable = False

        self.model = SequenceRegressor(2048, dropout_rate=0.1, intermediate_size=512, intermediate_activation='gelu', base_model = feature_extractor, out_dim = 1)

        self.loss_fn = K.losses.BinaryCrossentropy(from_logits=True)

        self.auc_tracker = K.metrics.AUC(from_logits=True)
        self.loss_tracker = K.metrics.Mean()

    def call(self, inputs, mask=None, training=False):
        """
        inputs: [B, A, N], B: batch_dim, A: # ASV in sample, N: nuctides,
        string tensor
        """
        return self.model(inputs, training=training)


    def train_step(self, data):
        x, y = data

        # create attention mask
        # example [["ACTG"], [""]]
        with tf.GradientTape() as tape:
            sample_embeddings, output = self(x, training=True)
            loss = self.loss_fn(y, output)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.auc_tracker.update_state(y, output)
        return {"loss": self.loss_tracker.result(), "auc": self.auc_tracker.result()}

    def test_step(self, data):
        x, y = data
        sample_embeddings, output = self(x, training=False)
        loss = self.loss_fn(y, output)

        self.loss_tracker.update_state(loss)
        self.auc_tracker.update_state(y, output)

        return {"loss": self.loss_tracker.result(), "auc": self.auc_tracker.result()}

    def predict_step(self, data):
        x, y = data
        sample_embeddings, output = self(x, training=False)
        predictions = tf.reshape(tf.keras.activations.sigmoid(output), shape=[-1])
        y = tf.reshape(y, shape=[-1])
        y = tf.cast(y, dtype=tf.float32)
        pred_label = tf.cast(predictions >= 0.5, dtype=tf.float32)
        correct = tf.reshape(tf.cast(pred_label == y, dtype=tf.float32), shape=[-1])
        correct = tf.reduce_mean(correct, axis=0)
        return pred_label, y, correct
    def get_config(self):
        config = super(Classifier, self).get_config()
        config.update({'feature_extractor': tf.keras.saving.serialize_keras_object(self.feature_extractor), 'build_input_shape': self.get_build_config()})
        return config
    def build(self, input_shape):
        if self.built:
            return
        super(Classifier, self).build(input_shape)
    @classmethod
    def from_config(cls, config):
        build_input_shape = config.pop('build_input_shape')
        input_shape = build_input_shape['input_shape']
        config['feature_extractor'] = tf.keras.saving.deserialize_keras_object(config['feature_extractor'])
        model = cls(**config)
        model.build(input_shape)
        return model

def get_dataset(gen):
        enqueuer = tf.keras.utils.OrderedEnqueuer(gen, use_multiprocessing=True)
        enqueuer.start(workers=2, max_queue_size=gen.steps_per_epoch)
        gen.stop = lambda: enqueuer.stop(0.1)
        y_type = tf.TensorSpec(shape=(gen.samples_per_minibatch, 1), dtype=tf.string)
        dataset = tf.data.Dataset.from_generator(
            enqueuer.get,
            output_signature=(
                (
                    tf.TensorSpec(shape=[None, 150], dtype=tf.int32),
                    tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
                    tf.TensorSpec(shape=[None], dtype=tf.int32),
                    tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
                ),
                tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
            ),
        )
        return dataset
def get_sample_type(file_path):
            filename = os.path.basename(file_path)
            # Remove the 'training_metadata_' prefix and the file extension
            if filename.startswith('training_metadata_'):
                sample_type = filename[len('training_metadata_'):]
                sample_type = os.path.splitext(sample_type)[0]
                return sample_type
            return "Unknown"
    
#function that creates training and valid split and trains each model
def train_model(train_fp, model=None):
    print()
    training_metadata = pd.read_csv(train_fp, sep='\t', index_col=0)
    X = training_metadata.drop(columns=['study_sample_type', 'has_covid'], axis=1)
    y = training_metadata[['study_sample_type', 'has_covid']]
    
    _, _, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    curr_best_val_loss = np.inf
    curr_best_model = None
    for i, (train_index, valid_index) in enumerate(skf.split(y, y['has_covid'])):
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]
        
        gd_train = GeneratorDataset(
            table='data/input/merged_biom_table.biom',
            metadata=y_train,
            metadata_column='has_covid',
            shuffle=False,
            is_categorical=False,
            shift=0,
            rarefy_depth = 4000,
            scale=1,
            batch_size = 4,
            gen_new_tables = True #only in training dataset
        )
    
        gd_valid = GeneratorDataset(
            table='data/input/merged_biom_table.biom',
            metadata=y_valid,
            metadata_column='has_covid',
            shuffle=False,
            is_categorical=False,
            shift=0,
            rarefy_depth = 4000,
            scale=1,
            batch_size = 4,
        )
    
        dataset_train = get_dataset(gd_train)
        dataset_valid = get_dataset(gd_valid)
  
        def get_sample_type(file_path):
            filename = os.path.basename(file_path)
            # Remove the 'training_metadata_' prefix and the file extension
            if filename.startswith('training_metadata_'):
                sample_type = filename[len('training_metadata_'):]
                sample_type = os.path.splitext(sample_type)[0]
                return sample_type
            return "Unknown"

        if model is None:
            base_model = tf.keras.models.load_model('model.keras', compile=False)
            model = Classifier(base_model)
        
        token_shape = tf.TensorShape([None, 150])
        batch_indicies = tf.TensorShape([None, 2])
        indicies_shape = tf.TensorShape([None])
        count_shape = tf.TensorShape([None, 1])
        model.build([token_shape, batch_indicies, indicies_shape, count_shape])
    
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate = 0.0,
            warmup_target = 0.0003, # maybe change
            warmup_steps=0,
            decay_steps=100000
        )
)
    
        model.compile(optimizer=optimizer, run_eagerly=False)
        #switch loss to val loss 
        #pass early stopping for callbacks
        history = model.fit(dataset_train, 
                  validation_data = dataset_valid, 
                  validation_steps=gd_valid.steps_per_epoch, 
                  epochs=10000, 
                  steps_per_epoch=gd_train.steps_per_epoch, 
                  callbacks=[
                             EarlyStopping(patience=250, start_from_epoch=0, restore_best_weights=True)
                            ])

        validation_loss = history.history['val_loss']
        epochs = np.array(range(len(validation_loss)))
        plt.plot(epochs, validation_loss)
        plt.title(f'Validation Loss Per Epoch {validation_loss[-1]}')
        plt.savefig(f'trained_models/{get_sample_type(train_fp)}/{get_sample_type(train_fp)}_{i}_model_loss.png')
        plt.close()
        if history.history['val_loss'][-1] < curr_best_val_loss:
            curr_best_model = model
        model.save(f'trained_models/{get_sample_type(train_fp)}/{get_sample_type(train_fp)}_{i}_model.keras', save_format='keras')
    curr_best_model.save(f'trained_models/{get_sample_type(train_fp)}/{get_sample_type(train_fp)}_best_model.keras', save_format='keras')

def run_model():
    #get data

    train_model('data/input/training_metadata_nares.tsv')
    train_model('data/input/training_metadata_stool.tsv')
    train_model('data/input/training_metadata_inside_floor.tsv')
    train_model('data/input/training_metadata_forehead.tsv')