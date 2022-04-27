import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from modules import Encoder, padding_mask
from modules import ProcessContext, ProcessSparseID, ProcessSequence, ProcessUser, ProcessGoods, ProcessPairs


def get_weight_sum_embed(seq_len, embed_dim, name):
    embed_target_in = tf.keras.Input(shape=(1, embed_dim), dtype=tf.float32)  ## (None, 100, 1)
    seq_embed_in = tf.keras.Input(shape=(seq_len, embed_dim), dtype=tf.float32)  ## (None, 100, 32)
    embed_target_tile = tf.tile(embed_target_in, [1, seq_len, 1])  ## (None, 100, 32)

    data_concat = tf.concat(
        [embed_target_tile, seq_embed_in, embed_target_tile - seq_embed_in, embed_target_tile * seq_embed_in],
        axis=2)  ## DIN 中的attention
    att_w = layers.Dense(20, activation=None, use_bias=True)(data_concat)  ## (None, 100, 128) ## 减少参数所以用了个8
    att_w = layers.PReLU()(att_w)  ### DIN论文中说这个对离散embedding特征学的比较好
    att_w = layers.Dense(1, activation="sigmoid", use_bias=True)(att_w)  ## (None, 100, 1)
    seq_embed = att_w * seq_embed_in  #### (None, 100, 1) * (None, 100, 32)
    seq_embed = tf.reduce_sum(seq_embed, axis=1, keepdims=True)  ## (None, 32)

    return tf.keras.Model(inputs=[embed_target_in, seq_embed_in], outputs=seq_embed, name=name)


