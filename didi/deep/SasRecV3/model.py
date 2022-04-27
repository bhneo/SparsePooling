import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Embedding, Input, PReLU
from modules import *
from tensorflow import keras
from tensorflow.keras.models import Model

# api functional model
def get_sasrec(maxlen, item_fea_col, embed_dim, embed_reg, dropout, num_heads, blocks, ffn_hidden_unit,
               use_norm = True, causality = True):

    # train_inputs = Input(shape=(206,), dtype=tf.float32, name = 'model_inputs') # (None, 206)
    user_inputs = Input(shape=(170,), dtype=tf.float32, name='user_inputs')  # (None, 170)
    item_inputs = Input(shape=(36,), dtype=tf.float32, name='item_inputs')  # (None, 36)

    # split
    tmp = tf.split(user_inputs, axis=1, num_or_size_splits=[20, 150])
    user_feat_inputs, user_seq_inputs= tmp  # (None,20) (None,150)
    tmp = tf.split(item_inputs, axis=1, num_or_size_splits=[1, 35])
    sample_cspuidx_inputs, item_feat_inputs = tmp  # (None,1) (None,35)

    ### ********** ###
    #   user part
    ### ********** ###
    new_seq_inputs = tf.cast(user_seq_inputs, dtype = tf.int32)
    mask = tf.expand_dims(tf.cast(tf.not_equal(new_seq_inputs, 0), dtype=tf.float32), axis=-1)  # (None, maxlen, 1)
    item_embedding = Embedding(input_dim=item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=embed_dim,
                                        mask_zero=True,
                                        embeddings_initializer='random_uniform',
                                        embeddings_regularizer=l2(embed_reg))
    seq_embed = item_embedding(new_seq_inputs)   # (None, 150, dim=50)

    pos_embedding = Embedding(input_dim=maxlen,
                              input_length=1,
                              output_dim=embed_dim,
                              mask_zero=False,
                              embeddings_initializer='random_uniform',
                              embeddings_regularizer=l2(embed_reg))
    pos_encoding = tf.expand_dims(pos_embedding(tf.range(maxlen)), axis=0)
    seq_embed += pos_encoding

    seq_embed = Dropout(dropout)(seq_embed)
    att_outputs = seq_embed   # (None, maxlen, dim)
    att_outputs *= mask

    encoder_layer = [EncoderLayer(embed_dim, num_heads, ffn_hidden_unit,
                                  dropout, use_norm, causality) for _ in range(blocks)]
    for block in encoder_layer:
        att_outputs = block([att_outputs, mask])  # (None, maxlen, dim)
        att_outputs *= mask  # (None, maxlen, dim)

    seq_outputs = att_outputs[:, -1]  # (None, dim) remain the embedding of the last item

    # concat
    user_feat_vec = tf.concat([user_feat_inputs, seq_outputs], -1)  # (None,20+50)

    # MLP
    ffn_1 = Dense(units=60, activation='relu', use_bias=True, kernel_initializer=keras.initializers.he_uniform())
    ffn_2 = Dense(units=50, activation='relu', use_bias=True, kernel_initializer=keras.initializers.he_uniform())
    norm1 = LayerNormalization(epsilon=1e-6, trainable=use_norm)
    norm2 = LayerNormalization(epsilon=1e-6, trainable=use_norm)

    feat_vec = ffn_1(user_feat_vec)
    feat_vec = norm1(feat_vec)
    feat_vec = ffn_2(feat_vec)  # (None,50)
    user_vec = norm2(feat_vec)

    ### ********** ###
    #   item part
    ### ********** ###
    item_info = item_embedding(sample_cspuidx_inputs)  # (None, 1, dim)
    item_emb_vec = item_info[:, -1] # (None, dim)

    item_feat_vec = tf.concat([item_emb_vec, item_feat_inputs], -1)  #(None, dim+35)
    ffn_3 = Dense(units=50, activation='relu', use_bias=True, kernel_initializer=keras.initializers.he_uniform())
    norm3 = LayerNormalization(epsilon=1e-6, trainable=use_norm)
    item_vec = ffn_3(item_feat_vec)  #(None, 50)
    item_vec = norm3(item_vec)

    # compute logits
    logits = tf.reduce_sum(user_vec * item_vec, axis=-1, keepdims=True) # (None, 1)
    logits = tf.nn.sigmoid(logits)

    model = Model(inputs=[user_inputs, item_inputs], outputs=[logits])

    model.__setattr__("user_inputs", user_inputs)
    model.__setattr__("user_outputs", user_vec)

    model.__setattr__("item_inputs", item_inputs)
    model.__setattr__("item_outputs", item_vec)

    return model






