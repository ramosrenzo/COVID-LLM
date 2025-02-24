import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow import keras as K
from aam.models.sequence_regressor import SequenceRegressor
from aam.callbacks import SaveModel
from keras.callbacks import EarlyStopping

from transformers import AutoTokenizer, AutoModel, BertConfig, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from biom import load_table, Table
from sklearn.model_selection import train_test_split, StratifiedKFold
import json
import os

warnings.filterwarnings('ignore')
logging.set_verbosity_error()

class GeneratorEmbedding(tf.keras.utils.Sequence):
    def __init__(
        self,
        table = None,
        metadata = None,
        metadata_column = None,
        shift = None,
        scale = "minmax",
        max_token_per_sample: int = 1024,
        shuffle: bool = False,
        rarefy_depth: int = 5000,
        epochs: int = 1000,
        gen_new_tables: bool = False,
        batch_size: int = 8,
        max_bp: int = 150,
        is_16S: bool = True,
        is_categorical = None,
        gen_new_table_frequency=3,
        return_sample_ids=False,
        tree_path=None,
        seed=None,
        asv_embeddings_fp=None
    ):      
        if isinstance(table, str):
            table = load_table(table)
        self.table: Table = table
        self.tree_path = tree_path
        self.is_categorical: bool = is_categorical
        asv_embeddings = np.load(asv_embeddings_fp)
        obs_ids = self.table.ids(axis="observation")
        self.asv_embeddings_dict = {k:v for k,v in zip(obs_ids, asv_embeddings)}
        self.metadata_column: str = metadata_column
        self.shift = shift
        self.scale = scale
        self.metadata: pd.Series = metadata
        self.rarefy_depth: int = rarefy_depth
        self.max_token_per_sample: int = max_token_per_sample
        self.return_sample_ids: bool = return_sample_ids
        self.include_sample_weight: bool = is_categorical
        self.shuffle = shuffle
        self.epochs = epochs
        self.gen_new_tables = gen_new_tables
        self.samples_per_minibatch = batch_size
        self.batch_size = batch_size
        self.max_bp = max_bp
        self.is_16S = is_16S
        self.seed = seed
        self.gen_new_table_frequency = gen_new_table_frequency
        self.epochs_since_last_table = 0
        self.encoder_target = None
        self.encoder_dtype = None
        self.encoder_output_type = None
        self.sample_ids = None
        self.asv_ids = None
        if self.tree_path is not None:
            self.tree = to_skbio_treenode(parse_newick(open(self.tree_path).read()))
            self.postorder_pos = {n.name: i for i, n in enumerate(self.tree.postorder()) if n.is_tip()}
        print("rarefy table...")
        self.rarefied_table: Table = self.table.subsample(rarefy_depth)
        self.size = self.rarefied_table.shape[1]
        self.steps_per_epoch = self.size // self.batch_size
        self.y_data = self.metadata.loc[self._rarefied_table.ids()]
        self.on_epoch_end()

    def __len__(self):
        return self.steps_per_epoch
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        sample_indices = self.sample_indices[start:end]
        batch_sample_ids = self.sample_ids[sample_indices]
        return self._batch_data(batch_sample_ids)
    
    def _batch_data(self, batch_sample_ids):
        num_unique_asvs, sparse_indices, obs_indices, counts = [], [], [], []
        cur_row_indx = 0
        for s_id in batch_sample_ids:
            sample_data = self.rarefied_table.data(s_id, dense=False).tocoo()
            obs_idx, sample_counts = sample_data.row, sample_data.data
            # remove zeros
            non_zero_mask = sample_counts > 0.0
            obs_idx = obs_idx[non_zero_mask]
            sample_counts = sample_counts[non_zero_mask]
            num_unique_asvs.append(len(obs_idx))
            sparse_indices.append(([[cur_row_indx, i] for i in range(len(obs_idx))]))
            obs_indices.append(obs_idx) 
            counts.append(sample_counts)
            cur_row_indx += 1
        num_unique_asvs = np.array(num_unique_asvs, dtype=np.int32)
        sparse_indices = np.vstack(sparse_indices, dtype=np.int32)
        obs_indices = np.hstack(obs_indices, dtype=np.int32)
        counts = np.hstack(counts, dtype=np.float32)[:, np.newaxis]
        
        # get list of unique observations in batch
        unique_obs, obs_indices = np.unique(obs_indices, return_inverse=True)
        obs = self.rarefied_table.ids(axis="observation")
        asvs = obs[unique_obs]
        y_true = self.y_data.loc[batch_sample_ids].to_numpy()[:, np.newaxis]
        asvs_array = np.array([self.asv_embeddings_dict[asv] for asv in asvs])
        batch_embeddings, counts = self.batch_embeddings(asvs_array, sparse_indices, counts, obs_indices)
        return (batch_embeddings, counts), y_true
    
    def batch_embeddings(self, asv_embeddings, batch_indicies, counts, asv_indices=None):
        emb_dim = tf.shape(asv_embeddings)[-1]
        if asv_indices is not None:
            asv_embeddings = tf.gather(asv_embeddings, asv_indices)
        batch_shape = tf.reduce_max(batch_indicies[:, 0]) + 1
        max_unique = tf.reduce_max(batch_indicies[:, 1]) + 1
        batch_embeddings = tf.scatter_nd(
            batch_indicies, asv_embeddings, shape=[batch_shape, max_unique, emb_dim]
        )
        counts = tf.scatter_nd(
            batch_indicies, counts, shape=[batch_shape, max_unique, 1]
        )
        return batch_embeddings.numpy(), counts.numpy()
        
    def sort_using_counts(self, tensor, counts):
        sorted_indices = tf.argsort(tf.squeeze(counts, axis=-1), axis=1, direction="DESCENDING")
        sorted_tensor = tf.gather(tensor, sorted_indices, axis=1, batch_dims=1)
        sorted_counts = tf.gather(counts, sorted_indices, axis=1, batch_dims=1)
        return sorted_tensor, sorted_counts

    def on_epoch_end(self):
        if self.gen_new_tables and self.epochs_since_last_table > self.gen_new_table_frequency:
            print("resampling dataset...")
            self.rarefied_table = self.table.subsample(self.rarefy_depth)
            self.epochs_since_last_table = 0
        if self.shuffle:
            np.random.shuffle(self.sample_indices)
        self.epochs_since_last_table += 1
    
    @property
    def rarefied_table(self):
        return self._rarefied_table
    
    @rarefied_table.setter
    def rarefied_table(self, table: Table):
        self._rarefied_table = table
        print("removing empty sample/obs from table")
        self._rarefied_table.remove_empty()
        if self.tree_path is not None:
            def sort_obs(obs):
                post_pos = [self.postorder_pos[ob] for ob in obs]
                sorted_indices = np.argsort(post_pos)
                return obs[sorted_indices]
            self._rarefied_table = self._rarefied_table.sort(sort_obs, axis="observation")
        self.sample_ids = self._rarefied_table.ids()
        self.asv_ids = self._rarefied_table.ids(axis="observation")
        self.sample_indices = np.arange(len(self.sample_ids))
        print("creating encoder target...")
        self.encoder_target = self._create_encoder_target()
        print("encoder target created")

    def _create_encoder_target(self) -> None:
        return None
    
    def _encoder_output(self, sample_ids):
        return None
    
    @property
    def table(self) -> Table:
        return self._table
    @table.setter
    def table(self, table):
        self._table = table
    @property
    def metadata(self) -> pd.Series:
        return self._metadata
    @metadata.setter
    def metadata(self, metadata):
        if metadata is None:
            return
        if isinstance(metadata, str):
            metadata = pd.read_csv(metadata, sep="\t", index_col=0, dtype={0: str})
        if self.metadata_column not in metadata.columns:
            raise Exception(f"Invalid metadata column {self.metadata_column}")
        print("aligning table with metadata")
        samp_ids = np.intersect1d(self.table.ids(axis="sample"), metadata.index)
        self.table.filter(samp_ids, axis="sample", inplace=True)
        self.table.remove_empty()
        metadata = metadata.loc[self.table.ids(), self.metadata_column]
        print(f"aligned table shape: {self.table.shape}")
        print(f"aligned metadata shape: {metadata.shape}")
        metadata = metadata.astype(np.int32)
        self._metadata = metadata.reindex(self.table.ids())
        print("done preprocessing metadata")

class Classifier(K.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = SequenceRegressor(
            2048, dropout_rate=0.1, embedding_dim = 768, intermediate_size = 3072, 
            intermediate_activation = "gelu", use_residual_connections = False, out_dim = 1
        )

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
        config.update({'build_input_shape': self.get_build_config()})
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
    
        embed_train = GeneratorEmbedding(
            table='data/input/merged_biom_table.biom',
            metadata=y_train,
            metadata_column='has_covid',
            shuffle=False,
            is_categorical=False,
            shift=0,
            rarefy_depth = 4000,
            scale=1,
            batch_size = 4,
            gen_new_tables = True, #only in training dataset
            asv_embeddings_fp = "asv_embeddings.npy"
            
        )
    
        embed_valid = GeneratorEmbedding(
            table='data/input/merged_biom_table.biom',
            metadata=y_valid,
            metadata_column='has_covid',
            shuffle=False,
            is_categorical=False,
            shift=0,
            rarefy_depth = 4000,
            scale=1,
            batch_size = 4,
            asv_embeddings_fp = "asv_embeddings.npy"
        )

        def get_sample_type(file_path):
            filename = os.path.basename(file_path)
            # Remove the 'training_metadata_' prefix and the file extension
            if filename.startswith('training_metadata_'):
                sample_type = filename[len('training_metadata_'):]
                sample_type = os.path.splitext(sample_type)[0]
                return sample_type
            return "Unknown"

        model = Classifier()
        asv_embedding_shape = tf.TensorShape([None, None, 768])
        count_shape = tf.TensorShape([None, None, 1])
        model.build([asv_embedding_shape, count_shape])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate = 0.0,
            warmup_target = 0.0003,
            warmup_steps=0,
            decay_steps=100_000
        )
)
        model.compile(optimizer=optimizer, run_eagerly=False)
        #switch loss to val loss 
        #pass early stopping for callbacks
        history = model.fit(embed_train, 
                  validation_data = embed_valid, 
                  validation_steps=embed_valid.steps_per_epoch, 
                  epochs=10, 
                  steps_per_epoch=embed_train.steps_per_epoch, 
                  callbacks=[
                          SaveModel("model_test.keras", report_back=1),
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
    print(f"\nDNABERT-2: Best model saved for {get_sample_type(train_fp)} samples.")

def run_dnabert_2():
    train_model('data/input/training_metadata_forehead.tsv')
    train_model('data/input/training_metadata_inside_floor.tsv')
    train_model('data/input/training_metadata_stool.tsv')
    train_model('data/input/training_metadata_nares.tsv')
    print(f"\nDNABERT-2: Training complete.")