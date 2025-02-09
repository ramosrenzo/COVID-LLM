from aam.data_handlers.generator_dataset import GeneratorDataset
from aam.models.transformers import TransformerEncoder
from aam.models.multihead_attention_pooling import MultiHeadAttentionPooling
import tensorflow as tf
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

K = tf.keras

class ASVWrapper(tf.keras.layers.Layer):
    def __init__(self, asv_encoder, **kwargs):
        super(ASVWrapper, self).__init__(**kwargs)
        self.asv_encoder = asv_encoder
        self.asv_encoder.trainable = False
        self.embedding_dim = self.asv_encoder.embedding_dim
    def call(self, inputs, training=False):
        embeddings, counts = self.asv_encoder(inputs, return_asv_embeddings=True, training=False)
        return embeddings, counts
    def get_config(self):
        config = super().get_config()
        config.update({"asv_encoder": tf.keras.saving.serialize_keras_object(self.asv_encoder)})
    @classmethod
    def from_config(cls, config):
        config["asv_encoder"] = tf.keras.saving.deserialize_keras_object(config["asv_encoder"])
        model = cls(**config)
        return model

class Classifier(K.Model):
    def __init__(self, feature_extractor, **kwargs):
        super().__init__(**kwargs)

        self.feature_extractor = feature_extractor
        self.feature_extractor.trainable = False

        self.encoder = TransformerEncoder(
            num_layers=4, 
            num_attention_heads=4, 
            intermediate_size=512, 
            dropout_rate=0.1, 
            normalize_outputs=False,
            use_residual_connections=True,
            use_linear_bias=False
        )  # transformer (or anything else)
        self.pooling = MultiHeadAttentionPooling(normalize_output=False,  use_linear_bias=False)  # extracts sample embeddings from our ASV embeddings
        self.dense_ff = K.layers.Dense(1)  # sample_embedings => classifys covid +/-

        self.loss_fn = K.losses.BinaryCrossentropy(from_logits=True)

        self.auc_tracker = K.metrics.Mean()
        self.loss_tracker = K.metrics.Mean()

    def call(self, inputs, mask=None, training=False):
        """
        inputs: [B, A, N], B: batch_dim, A: # ASV in sample, N: nuctides,
        string tensor
        """


        # aam case
        features, counts = self.feature_extractor(inputs)
        mask =  tf.cast(counts >  0, dtype="float32")

        encodig_output = self.encoder(features, mask=mask, training=training)  # [B, A, N]
        pooling_output = self.pooling(encodig_output, mask=mask, training=training)  # [B, N]
        return self.dense_ff(pooling_output)

    def train_step(self, data):
        x, y = data

        # create attention mask
        # example [["ACTG"], [""]]
        with tf.GradientTape() as tape:
            output = self(x, training=True)
            loss = self.loss_fn(y, output)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.auc_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(), "auc": self.auc_tracker.result()}

    def test_step(self, data):
        x, y = data
        output = self(x, training=False)
        loss = self.loss_fn(y, output)

        self.loss_tracker.update_state(loss)
        self.auc_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result(), "auc": self.auc_tracker.result()}

    def predict_step(self, data):
        x, y = data
        return self(x)

def run_model():
    #get data
    fecal = pd.read_csv('data/fecal.tsv', sep='\t')

    fecal_filtered = fecal[['sample_name', 'covid_positive']]
    
    fecal_filtered
    
    def check_covid_positive(row):
        if row =='yes':
            return 1
        else:
            return 0

    fecal_filtered['has_covid'] = fecal_filtered['covid_positive'].apply(check_covid_positive)
    fecal_filtered.set_index('sample_name', inplace=True)

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

    gd = GeneratorDataset(
        table='/home/swchan/DSC170/data/input/merged_biom_table.biom',
        metadata='/home/swchan/DSC170/data/input/training_metadata.tsv',
        metadata_column='has_covid',
        shuffle=False,
        is_categorical=False,
        shift=0,
        rarefy_depth = 5000,
        scale=1,
        batch_size=4
    )

    dataset = get_dataset(gd)

    base_model = tf.keras.models.load_model('/home/swchan/DSC170/model.keras', compile=False)
    base_model = ASVWrapper(base_model)

    model = Classifier(base_model)

    optimizer = K.optimizers.Adam(
        learning_rate=1e-4
    )

    model.compile(optimizer=optimizer, run_eagerly=False)

    model.fit(dataset, epochs=10, steps_per_epoch=gd.steps_per_epoch)
