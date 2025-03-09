import tensorflow as tf
if tf.config.list_physical_devices("GPU"):
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

from aam.models.sequence_regressor import SequenceRegressor
from aam.models.sequence_regressor_v2 import SequenceRegressorV2
import pandas as pd
import numpy as np
from biom import load_table, Table
K = tf.keras

class GeneratorDataset(tf.keras.utils.Sequence):
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
        sequence_embeddings = None,
        sequence_labels = None,
        upsample = True,
        drop_remainder = True,
        num_init_tables = 1,
        rarefy_seed = None
    ):      
        if batch_size % 2 != 0:
            raise Exception("Batch size must be even")
            
        if isinstance(table, str):
            table = load_table(table)
        if sequence_embeddings == None:
            raise Exception("Must pass in sequence embeddings")
        if sequence_labels == None:
            raise Exception("Must pass in sequence labels")
        self.table: Table = table
        self.tree_path = tree_path
        self.is_categorical: bool = is_categorical
        obs_ids = self.table.ids(axis="observation")
        self.metadata_column: str = metadata_column
        self.shift = shift
        self.scale = scale
        self.sequence_embeddings = sequence_embeddings
        self.sequence_labels = sequence_labels
        if self.sequence_embeddings is not None:
            emb = np.load(self.sequence_embeddings)
            emb_mean = np.mean(emb, axis=0)
            emb_std = np.std(emb, axis=0)
            self.sequence_embeddings = (emb - emb_mean) / (emb_std + 1e-8)
            self.sequence_labels = np.load(self.sequence_labels, allow_pickle=True)
            self.sequence_labels = self.sequence_labels.astype(np.str_)
            self.sequence_labels = np.char.encode(
                self.sequence_labels, encoding="utf-8"
            )
        self.metadata: pd.Series = metadata
        self.rarefy_depth: int = rarefy_depth
        self.max_token_per_sample: int = max_token_per_sample
        self.return_sample_ids: bool = return_sample_ids
        self.include_sample_weight: bool = is_categorical
        self.shuffle = shuffle
        self.epochs = epochs
        self.gen_new_tables = gen_new_tables
        self.num_init_tables = num_init_tables
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
        self.rarefy_seed = rarefy_seed
        if self.num_init_tables > 1:
            tables = [self.table.subsample(rarefy_depth, seed=rarefy_seed+i) for i in range(self.num_init_tables)]
            self.rarefied_table: Table = tables[0].concat(tables[1:])
        else:
            self.rarefied_table: Table = self.table.subsample(rarefy_depth, seed=rarefy_seed)
        self.upsample = upsample
        self.size = self.rarefied_table.shape[1]
        self.steps_per_epoch = self.size // self.batch_size
        if not drop_remainder and (self.steps_per_epoch * self.batch_size) != self.size:
            self.steps_per_epoch += 1
        self.y_data = self.metadata.loc[self._rarefied_table.ids(), self.metadata_column]
        self.sample_weights = self.y_data.copy()
        class_0_p = len(self.sample_weights[self.sample_weights == 0])
        class_1_p = len(self.sample_weights[self.sample_weights == 1])
        self.sample_weights[self.sample_weights == 0] = self.sample_weights.size / (2 * class_0_p)
        self.sample_weights[self.sample_weights == 1] = self.sample_weights.size / (2 * class_1_p)
        # self.sample_weights = [class_0_p, class_1_p]
        self.on_epoch_end()

    def __len__(self):
        return self.steps_per_epoch
    
    def __getitem__(self, idx):
        # start = idx * self.batch_size
        # end = start + self.batch_size
        # sample_indices = self.sample_indices[start:end]
        # batch_sample_ids = self.sample_ids[sample_indices]
        if self.upsample == True:
            batch_metadata = self.metadata.groupby(self.metadata_column).sample(self.batch_size//2, replace=True)
            batch_sample_ids = batch_metadata.index.to_numpy()
        else:
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
        if self.sequence_embeddings is None:
            lookup = {
            "a": 1,
            "c": 2,
            "g": 3,
            "t": 4,
            }

            def map(asv):
                asv = asv.lower()
                return np.array([lookup[c] for c in asv], dtype=np.int32)[np.newaxis, :]
    
            tokens = np.concatenate(
            [map(asv) for asv in self.asv_ids[unique_obs]], axis=0
            )
        else:
            asvs, asv_ids_idx, sequence_labels_idx = np.intersect1d(
            self.asv_ids[unique_obs],
            self.sequence_labels,
            assume_unique=True,
            return_indices=True,
            )
            tokens = self.sequence_embeddings[sequence_labels_idx]
        y_true = self.y_data.loc[batch_sample_ids].to_numpy()[:, np.newaxis]
        sample_weights = self.sample_weights.loc[batch_sample_ids].to_numpy()[:, np.newaxis]
        return (tokens, sparse_indices, obs_indices, counts), (y_true, sample_weights)
        
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
        self._metadata = self.metadata.loc[self.sample_ids]
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
        metadata = metadata.loc[self.table.ids(), [self.metadata_column]]
        print(f"aligned table shape: {self.table.shape}")
        print(f"aligned metadata shape: {metadata.shape}")
        metadata = metadata.astype(np.int32)
        self._metadata = metadata.reindex(self.table.ids())
        print("done preprocessing metadata")
        
class Classifier(K.Model):
    def __init__(self, num_hidden_layers, hidden_dim=None, dropout_rate=None, use_cova=False, **kwargs):
        super().__init__(**kwargs)
        
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.use_cova = use_cova

        if not use_cova:
            self.model = SequenceRegressor(hidden_dim=hidden_dim, num_hidden_layers=self.num_hidden_layers, dropout_rate=self.dropout_rate, intermediate_activation='gelu', out_dim = 1)
        else:
            self.model = SequenceRegressorV2(rarefy_depth=1000)
        self.output_activation = tf.keras.layers.Activation('sigmoid', dtype=tf.float64)
        self.output_activation.build([None, 1])
        self.loss_fn = K.losses.BinaryCrossentropy(from_logits=False, reduction='none')

        self.auc_tracker = K.metrics.AUC(from_logits=False)
        self.loss_tracker = K.metrics.Mean()
    
    def build(self, input_shape):
        if self.built:
            return
        token_shape = tf.TensorShape([None, 768])
        batch_indicies = tf.TensorShape([None, 2])
        indicies_shape = tf.TensorShape([None])
        count_shape = tf.TensorShape([None, 1])
        self.model.build(
             [token_shape, batch_indicies, indicies_shape, count_shape]
        )
        super(Classifier, self).build(input_shape)
        
    def call(self, inputs, mask=None, training=False):
        """
        inputs: [B, A, N], B: batch_dim, A: # ASV in sample, N: nuctides,
        string tensor
        """
        sample_embeddings, output = self.model(inputs, training=training)
        return sample_embeddings, self.output_activation(output)

    def compute_loss(self, y, y_pred):
        y_true, sample_weights = y
        y_true = tf.reshape(y_true, shape=[-1])
        y_pred = tf.reshape(y_pred, shape=[-1])
        sample_weights = tf.reshape(sample_weights, shape=[-1])
        sample_weights = tf.cast(sample_weights, dtype=tf.float64)
        loss = self.loss_fn(y_true, y_pred)
        return tf.reduce_mean(loss * sample_weights)

    def compute_metric(self, y, y_pred):
        y_true, sample_weights = y
        y_true = tf.reshape(y_true, shape=[-1])
        y_pred = tf.reshape(y_pred, shape=[-1])
        sample_weights = tf.reshape(sample_weights, shape=[-1])
        sample_weights = tf.cast(sample_weights, dtype=tf.float64)
        self.auc_tracker.update_state(y_true, y_pred)
        
    def train_step(self, data):
        x, y = data

        # create attention mask
        # example [["ACTG"], [""]]
        with tf.GradientTape() as tape:
            sample_embeddings, output = self(x, training=True)
            loss = self.compute_loss(y, output)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.compute_metric(y, output)
        return {"loss": self.loss_tracker.result(), "auc": self.auc_tracker.result()}

    def test_step(self, data):
        x, y = data
        sample_embeddings, output = self(x, training=False)
        loss = self.compute_loss(y, output)

        self.loss_tracker.update_state(loss)
        self.compute_metric(y, output)

        return {"loss": self.loss_tracker.result(), "auc": self.auc_tracker.result()}

    def predict_step(self, data):
        x, y = data
        y, _ = y
        sample_embeddings, output = self(x, training=False)
        predictions = tf.reshape(output, shape=[-1])
        y = tf.reshape(y, shape=[-1])
        y = tf.cast(y, dtype=tf.float64)
        pred_label = tf.cast(predictions >= 0.5, dtype=tf.float64)
        correct = tf.reshape(tf.cast(pred_label == y, dtype=tf.float64), shape=[-1])
        correct = tf.reduce_mean(correct, axis=0)
        return predictions, y, correct
    def get_config(self):
        config = super(Classifier, self).get_config()
        config.update({'build_input_shape': self.get_build_config(), 'hidden_dim': self.hidden_dim, 'num_hidden_layers': self.num_hidden_layers, 'dropout_rate': self.dropout_rate, 'use_cova': self.use_cova})
        return config
    @classmethod
    def from_config(cls, config):
        build_input_shape = config.pop('build_input_shape')
        input_shape = build_input_shape['input_shape']
        if not 'hidden_dim' in config:
            config['hidden_dim'] = 32
            config['num_hidden_layers'] = 2
            config['dropout_rate'] = 0.1
        model = cls(**config)
        return model
        