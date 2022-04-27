"""
    deepFM:
    样本：首页+去除无行为用户+去leader、仓店+负样本20%随机采样（正负样本1：30）；
    特征：93维实时特征+passtime+user_group+context+sequence+goods_sparse_v3.1+user分桶+goods分桶+pair分桶+cspu特征+用户偏好特征+商品毛利特征；
    deep：128+64+32，隐向量=10，去除L2和dropout；
    batch_size=10000
    利用transformer对用户行为序列进行建模
"""

# coding: utf-8
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import (layers, initializers, metrics)
from tensorflow.keras import backend as K
import re
import datetime
import argparse
import random
import os
import subprocess
import json
import sys

path_append = os.path.join(os.path.abspath(os.path.dirname(__file__)), "./")
sys.path.append(path_append)
from modules import Encoder, padding_mask

print("run: ", os.path.abspath(__file__))
print("tensorflow version: ", tf.__version__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    # 设置内存增长方式 自增长
    tf.config.experimental.set_memory_growth(gpu, True)

print(tf.config.list_physical_devices('GPU'))


def get_list_by_feature_list(feature_list, dims_dict):
    rst = []
    for k in feature_list:
        rst.append(dims_dict[k])
    return rst


def read_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


class Config(object):
    ## ---train compile----
    workspace_dir = None

    epochs = 100000
    batch_size = 1024
    prefetch_buffer_size = 200000
    shuffle_buffer_size = 20000
    steps_per_epoch = 5000
    validation_steps = 500
    init_lr = 1e-3
    embed_dim = 8

    embed_init = initializers.glorot_normal()  ## also called Xavier normal initializer.
    kernel_init = initializers.glorot_normal()  ## also called Xavier normal initializer.
    kernel_regular = None

    ## 特征解析
    conf = read_json("./config.json")
    field_length = conf["field_length_dict"]
    dims_dict = conf["dims_dict"]["others"]
    short_seq_len = conf["field_length_dict"]["short_behavior"]
    long_seq_len = conf["field_length_dict"]["long_behavior"]

    feature_concat_col = conf["stride"]["inputs"]
    features_name = conf["stride"]["inputs"]  ## 特征索引的顺序
    stride = np.cumsum(get_list_by_feature_list(conf["stride"]["inputs"], field_length))  ## 特征在向量中的索引
    sequence_stride = np.cumsum(get_list_by_feature_list(conf["stride"]["sequence"], field_length))

    all_len = sum(get_list_by_feature_list(conf["stride"]["inputs"], field_length))

    goods_feature_selected = ["level1_idx", "level3_idx", "level4_idx", "goods_unit_idx", "origin_idx", "selling_points_idx", "brand_idx", "entity_idx"]
    pair_field_dims = get_list_by_feature_list(conf["fmap"]["fm_pair"], conf["dims_dict"]["others"])
    goods_field_dims = get_list_by_feature_list(goods_feature_selected, conf["dims_dict"]["others"])
    realtime_back_category_dims = [10] * field_length['realtime_back_category']
    realtime_goods_dims = [32] * field_length['realtime_goods']
    bucket_user_dims = get_list_by_feature_list(conf["fmap"]["bucket_user"], conf["dims_dict"]["bucket_user"])
    bucket_goods_dims = get_list_by_feature_list(conf["fmap"]["bucket_goods"], conf["dims_dict"]["bucket_goods"])
    bucket_pair_dims = get_list_by_feature_list(conf["fmap"]["bucket_pair"], conf["dims_dict"]["bucket_pair"])
    context_dims = get_list_by_feature_list(conf["fmap"]["context"], conf["dims_dict"]["others"])[:2]
    realtime_passtime_dims = [24 + 90] * 21
    realtime_user_group_dims = [32] * field_length['realtime_user_group']
    goods_sparse_id_fmap = dict(zip(conf["fmap"]["goods_idx"], range(0, len(conf["fmap"]["goods_idx"]))))
    bucket_user_cspu_dims = get_list_by_feature_list(conf["fmap"]["bucket_user_cspu"],
                                                     conf["dims_dict"]["bucket_user_cspu"])
    bucket_ozid_cspu_dims = get_list_by_feature_list(conf["fmap"]["bucket_ozid_cspu"],
                                                     conf["dims_dict"]["bucket_ozid_cspu"])
    bucket_user_behavior_dims = get_list_by_feature_list(conf["fmap"]["bucket_user_behavior"],
                                                         conf["dims_dict"]["bucket_user_behavior"])
    bucket_goods_gross_dims = get_list_by_feature_list(conf["fmap"]["bucket_goods_gross"],
                                                       conf["dims_dict"]["bucket_goods_gross"])


cfg = Config()


def get_sum_pooling_embed(seq_len, embed_dim):
    seq_in = keras.Input(shape=(seq_len,), dtype=tf.int32)
    seq_embed_in = keras.Input(shape=(seq_len, embed_dim), dtype=tf.float32)
    g = tf.greater(seq_in, tf.zeros((1,), dtype=tf.int32))
    data_mask = tf.cast(g, tf.float32)  ## (None, 100)
    data_mask = tf.expand_dims(data_mask, axis=2)  ## (None, 100, 1)
    data_embed_masked = tf.multiply(seq_embed_in, data_mask)  ## (None, 100, 1) *  (None, 100, 32), broadcast
    data_embed_masked = tf.reduce_sum(data_embed_masked, axis=1)  ## (None, 32)
    data_embed_masked = tf.expand_dims(data_embed_masked, axis=1)  ## (None, 1, 32)
    return keras.Model(inputs=[seq_in, seq_embed_in], outputs=data_embed_masked)


def get_weight_sum_embed(seq_len, embed_dim):
    embed_target_in = keras.Input(shape=(1, embed_dim), dtype=tf.float32)  ## (None, 100, 1)
    seq_embed_in = keras.Input(shape=(seq_len, embed_dim), dtype=tf.float32)  ## (None, 100, 32)
    embed_target_tile = tf.tile(embed_target_in, [1, seq_len, 1])  ## (None, 100, 32)
    # data_concat = tf.concat([embed_target_tile, seq_embed_in], axis=2) ## (None, 100, 64)
    data_concat = tf.concat(
        [embed_target_tile, seq_embed_in, embed_target_tile - seq_embed_in, embed_target_tile * seq_embed_in],
        axis=2)  ## DIN 中的attention
    att_w = layers.Dense(20, activation=None, use_bias=True)(data_concat)  ## (None, 100, 128) ## 减少参数所以用了个8
    att_w = layers.PReLU()(att_w)  ### DIN论文中说这个对离散embedding特征学的比较好
    att_w = layers.Dense(1, activation="sigmoid", use_bias=True)(att_w)  ## (None, 100, 1)
    seq_embed = att_w * seq_embed_in  #### (None, 100, 1) * (None, 100, 32)
    seq_embed = tf.reduce_sum(seq_embed, axis=1)  ## (None, 32)
    seq_embed = tf.expand_dims(seq_embed, axis=1)  ## (None, 1, 32)

    return keras.Model(inputs=[embed_target_in, seq_embed_in], outputs=seq_embed)


def get_masked_embed(seq_len, embed_dim):
    seq_in = keras.Input(shape=(seq_len,), dtype=tf.int32)
    seq_embed_in = keras.Input(shape=(seq_len, embed_dim), dtype=tf.float32)
    g = tf.greater(seq_in, tf.zeros((1,), dtype=tf.int32))
    data_mask = tf.cast(g, tf.float32)  ## (None, 100)
    data_mask = tf.expand_dims(data_mask, axis=2)  ## (None, 100, 1)
    data_embed_masked = tf.multiply(seq_embed_in, data_mask)  ## (None, 100, 1) *  (None, 100, 32), broadcast
    return keras.Model(inputs=[seq_in, seq_embed_in], outputs=data_embed_masked)


def get_fm_model():
    ### -------------------------------------------------------------
    ###                     一、   input
    ### -------------------------------------------------------------
    inputs = keras.Input(shape=(cfg.all_len,), dtype=tf.float32, name="inputs")
    print(inputs)
    ## 1 特征拆解
    realtime_features = inputs[:, :cfg.stride[0]]
    pair_feature = tf.cast(inputs[:, cfg.stride[0]:cfg.stride[1]], tf.int32)
    goods_sparse_features = tf.cast(inputs[:, cfg.stride[1]:cfg.stride[2]], tf.int32)
    sequence = tf.cast(inputs[:, cfg.stride[2]:cfg.stride[3]], tf.int32)

    ## 1.1 非实时分桶特征
    bucket_user_features = inputs[:, cfg.stride[3]:cfg.stride[4]]
    bucket_goods_features = inputs[:, cfg.stride[4]:cfg.stride[5]]
    bucket_pair_features = inputs[:, cfg.stride[5]:cfg.stride[6]]
    context_features = tf.cast(inputs[:, cfg.stride[6]:cfg.stride[7]], tf.int32)
    bucket_user_cspu_obj = tf.cast(inputs[:, cfg.stride[7]:cfg.stride[8]], tf.int32)
    bucket_ozid_cspu_obj = tf.cast(inputs[:, cfg.stride[8]:cfg.stride[9]], tf.int32)
    bucket_user_behavior_obj = tf.cast(inputs[:, cfg.stride[9]:cfg.stride[10]], tf.int32)
    bucket_goods_gross_obj = tf.cast(inputs[:, cfg.stride[10]:cfg.stride[11]], tf.int32)

    context_features = context_features[:, 0:2]  ## 只取星期和小时；
    bucket_pair_box_obj = tf.cast(tf.concat([bucket_pair_features[:, :10], bucket_pair_features[:, 20:]], axis=-1),
                                  tf.int32)  ## 离散值
    # bucket_pair_box_num = tf.cast(bucket_pair_features[:, 10:20], tf.float32)  ## 连续值

    bucket_user_box_obj = tf.cast(bucket_user_features[:, :66], tf.int32)  ## 离散值
    # bucket_user_box_num = tf.cast(bucket_user_features[:, 66:120], tf.float32)  ## 连续值
    bucket_user_raw = tf.cast(bucket_user_features[:, 120:], tf.float32)  ## 连续值

    bucket_goods_box_obj = tf.cast(bucket_goods_features[:, 4:53], tf.int32)  ## 离散值,去除后端类目，goods_sparse_features已经有了
    # bucket_goods_box_num = tf.cast(bucket_goods_features[:, 53:98], tf.float32)  ## 连续值
    bucket_goods_raw = tf.cast(bucket_goods_features[:, 98:], tf.float32)  ## 连续值

    # bucket_pair_box_num = tf.clip_by_value(bucket_pair_box_num, clip_value_min=0, clip_value_max=1)
    # bucket_user_box_num = tf.clip_by_value(bucket_user_box_num, clip_value_min=0, clip_value_max=1)
    bucket_user_raw = tf.clip_by_value(bucket_user_raw, clip_value_min=0, clip_value_max=1)
    # bucket_goods_box_num = tf.clip_by_value(bucket_goods_box_num, clip_value_min=0, clip_value_max=1)
    bucket_goods_raw = tf.clip_by_value(bucket_goods_raw, clip_value_min=0, clip_value_max=1)

    ## 1.2 实时
    # realtime_user = realtime_features[:, :5]
    realtime_pair_click = realtime_features[:, 5]
    ## realtime_cross_category_front = features[:,6:6+24]  ## 前端类目pair不要了
    realtime_back_category = realtime_features[:, 30:30 + 4 * 12]  # 分桶后传入fm
    realtime_goods = realtime_features[:, 30 + 4 * 12:30 + 4 * 12 + 15]
    realtime_passtime = realtime_features[:, 30 + 4 * 12 + 15:30 + 4 * 12 + 15 + 21]
    realtime_user_group = realtime_features[:, 30 + 4 * 12 + 15 + 21:30 + 4 * 12 + 15 + 21 + 36]

    ## 1.3 商品id类特征
    print("sequence: ", sequence)
    lv1_idx = tf.reshape(
        tf.cast(goods_sparse_features[:, cfg.goods_sparse_id_fmap['back_lv1']], tf.int32), shape=(-1, 1))
    lv2_idx = tf.reshape(
        tf.cast(goods_sparse_features[:, cfg.goods_sparse_id_fmap['back_lv2']], tf.int32), shape=(-1, 1))
    lv3_idx = tf.reshape(
        tf.cast(goods_sparse_features[:, cfg.goods_sparse_id_fmap['back_lv3']], tf.int32), shape=(-1, 1))
    lv4_idx = tf.reshape(
        tf.cast(goods_sparse_features[:, cfg.goods_sparse_id_fmap['back_lv4']], tf.int32), shape=(-1, 1))
    cspu_idx = tf.reshape(tf.cast(goods_sparse_features[:, cfg.goods_sparse_id_fmap['cspu_id']], tf.int32),
                          shape=(-1, 1))
    supplier_idx = tf.reshape(tf.cast(goods_sparse_features[:, cfg.goods_sparse_id_fmap['supplier_id']], tf.int32),
                              shape=(-1, 1))
    goods_unit_idx = tf.reshape(tf.cast(goods_sparse_features[:, cfg.goods_sparse_id_fmap['goods_unit']], tf.int32),
                                shape=(-1, 1))
    origin_idx = tf.reshape(tf.cast(goods_sparse_features[:, cfg.goods_sparse_id_fmap['origin']], tf.int32),
                            shape=(-1, 1))
    selling_points_idx = tf.reshape(tf.cast(goods_sparse_features[:, cfg.goods_sparse_id_fmap['selling_points']], tf.int32),
                                    shape=(-1, 1))
    brand_idx = tf.reshape(tf.cast(goods_sparse_features[:, cfg.goods_sparse_id_fmap['brand']], tf.int32),
                                   shape=(-1, 1))
    entity_idx = tf.reshape(tf.cast(goods_sparse_features[:, cfg.goods_sparse_id_fmap['goods_entity']], tf.int32),
                            shape=(-1, 1))
    print("lv1_idx 1111: ", lv1_idx)

    lv1_idx = tf.clip_by_value(lv1_idx, clip_value_min=0, clip_value_max=cfg.dims_dict['level1_idx'] - 1)
    lv2_idx = tf.clip_by_value(lv2_idx, clip_value_min=0, clip_value_max=cfg.dims_dict['level2_idx'] - 1)
    lv3_idx = tf.clip_by_value(lv3_idx, clip_value_min=0, clip_value_max=cfg.dims_dict['level3_idx'] - 1)
    lv4_idx = tf.clip_by_value(lv4_idx, clip_value_min=0, clip_value_max=cfg.dims_dict['level4_idx'] - 1)
    cspu_idx = tf.clip_by_value(cspu_idx, clip_value_min=0, clip_value_max=cfg.dims_dict['cspu_idx'] - 1)
    supplier_idx = tf.clip_by_value(supplier_idx, clip_value_min=0, clip_value_max=cfg.dims_dict['supplier_idx'] - 1)
    goods_unit_idx = tf.clip_by_value(goods_unit_idx, clip_value_min=0,
                                      clip_value_max=cfg.dims_dict['goods_unit_idx'] - 1)
    origin_idx = tf.clip_by_value(origin_idx, clip_value_min=0, clip_value_max=cfg.dims_dict['origin_idx'] - 1)
    selling_points_idx = tf.clip_by_value(selling_points_idx, clip_value_min=0, clip_value_max=cfg.dims_dict['selling_points_idx'] - 1)
    brand_idx = tf.clip_by_value(brand_idx, clip_value_min=0, clip_value_max=cfg.dims_dict['brand_idx'] - 1)
    entity_idx = tf.clip_by_value(entity_idx, clip_value_min=0, clip_value_max=cfg.dims_dict['entity_idx'] - 1)
    print("lv1_idx: ", lv1_idx)

    ## concat 商品离散idx
    goods_sparse = tf.concat([lv1_idx, lv3_idx, lv4_idx, goods_unit_idx, origin_idx, selling_points_idx, brand_idx, entity_idx],
                             axis=1, name="goods_sparse_idx_concat")
    print("goods_sparse: ", goods_sparse)

    ## 1.4 序列拆解
    long_click = sequence[:, :cfg.sequence_stride[0]]
    long_cart = sequence[:, cfg.sequence_stride[0]: cfg.sequence_stride[1]]
    long_buy = sequence[:, cfg.sequence_stride[1]:cfg.sequence_stride[2]]
    # long_buy_supplier = sequence[:, cfg.sequence_stride[2]:cfg.sequence_stride[3]]
    long_buy_level2 = sequence[:, cfg.sequence_stride[3]:cfg.sequence_stride[4]]
    long_cart_level2 = sequence[:, cfg.sequence_stride[4]:cfg.sequence_stride[5]]
    long_click_level2 = sequence[:, cfg.sequence_stride[5]:cfg.sequence_stride[6]]

    short_click = sequence[:, cfg.sequence_stride[6]:cfg.sequence_stride[7]]
    short_cart = sequence[:, cfg.sequence_stride[7]:cfg.sequence_stride[8]]
    short_buy = sequence[:, cfg.sequence_stride[8]:cfg.sequence_stride[9]]
    # short_click_supplier = sequence[:, cfg.sequence_stride[9]:cfg.sequence_stride[10]]
    short_click_level2 = sequence[:, cfg.sequence_stride[10]:cfg.sequence_stride[11]]
    short_cart_level2 = sequence[:, cfg.sequence_stride[11]:cfg.sequence_stride[12]]
    short_buy_level2 = sequence[:, cfg.sequence_stride[12]:]

    print("long_click: ", long_click)
    print("short_click: ", short_click)
    ### -------------------------------------------------------------
    ###                        input done！！！！
    ### -------------------------------------------------------------

    ### -------------------------------------------------------------
    ###                        二、 embedding layer
    ### -------------------------------------------------------------
    ## 2.1 sequence attention
    ## lookup table
    cspu_embed_lookup = layers.Embedding(input_dim=cfg.dims_dict['cspu_idx'],
                                         output_dim=cfg.embed_dim,
                                         embeddings_initializer=cfg.embed_init,
                                         embeddings_regularizer=cfg.kernel_regular,
                                         input_length=None)
    supplier_embed_lookup = layers.Embedding(input_dim=cfg.dims_dict['supplier_idx'],
                                             output_dim=cfg.embed_dim,
                                             embeddings_initializer=cfg.embed_init,
                                             embeddings_regularizer=cfg.kernel_regular,
                                             input_length=None)
    level2_embed_lookup = layers.Embedding(input_dim=cfg.dims_dict['level2_idx'],
                                           output_dim=cfg.embed_dim,
                                           embeddings_initializer=cfg.embed_init,
                                           embeddings_regularizer=cfg.kernel_regular,
                                           input_length=None)

    ## 商品id特征，共享的特征
    cspu_idx = tf.clip_by_value(cspu_idx, clip_value_min=0, clip_value_max=cfg.dims_dict['cspu_idx'] - 1)
    supplier_idx = tf.clip_by_value(supplier_idx, clip_value_min=0, clip_value_max=cfg.dims_dict['supplier_idx'] - 1)
    lv2_idx = tf.clip_by_value(lv2_idx, clip_value_min=0, clip_value_max=cfg.dims_dict['level2_idx'] - 1)
    cspu_embed = cspu_embed_lookup(cspu_idx)
    supplier_embed = supplier_embed_lookup(supplier_idx)
    lv2_embed = level2_embed_lookup(lv2_idx)

    long_click = tf.clip_by_value(long_click, clip_value_min=0, clip_value_max=cfg.dims_dict['cspu_idx'] - 1)
    long_cart = tf.clip_by_value(long_cart, clip_value_min=0, clip_value_max=cfg.dims_dict['cspu_idx'] - 1)
    long_buy = tf.clip_by_value(long_buy, clip_value_min=0, clip_value_max=cfg.dims_dict['cspu_idx'] - 1)

    long_buy_level2 = tf.clip_by_value(long_buy_level2, clip_value_min=0,
                                       clip_value_max=cfg.dims_dict['level2_idx'] - 1)
    long_cart_level2 = tf.clip_by_value(long_cart_level2, clip_value_min=0,
                                        clip_value_max=cfg.dims_dict['level2_idx'] - 1)
    long_click_level2 = tf.clip_by_value(long_click_level2, clip_value_min=0,
                                         clip_value_max=cfg.dims_dict['level2_idx'] - 1)

    long_click_embed = cspu_embed_lookup(long_click)
    long_cart_embed = cspu_embed_lookup(long_cart)
    long_buy_embed = cspu_embed_lookup(long_buy)

    long_buy_level2_embed = level2_embed_lookup(long_buy_level2)
    long_cart_level2_embed = level2_embed_lookup(long_cart_level2)
    long_click_level2_embed = level2_embed_lookup(long_click_level2)

    short_click = tf.clip_by_value(short_click, clip_value_min=0, clip_value_max=cfg.dims_dict['cspu_idx'] - 1)
    short_cart = tf.clip_by_value(short_cart, clip_value_min=0, clip_value_max=cfg.dims_dict['cspu_idx'] - 1)
    short_buy = tf.clip_by_value(short_buy, clip_value_min=0, clip_value_max=cfg.dims_dict['cspu_idx'] - 1)

    short_click_level2 = tf.clip_by_value(short_click_level2, clip_value_min=0,
                                          clip_value_max=cfg.dims_dict['level2_idx'] - 1)
    short_cart_level2 = tf.clip_by_value(short_cart_level2, clip_value_min=0,
                                         clip_value_max=cfg.dims_dict['level2_idx'] - 1)
    short_buy_level2 = tf.clip_by_value(short_buy_level2, clip_value_min=0,
                                        clip_value_max=cfg.dims_dict['level2_idx'] - 1)

    short_click_embed = cspu_embed_lookup(short_click)
    short_cart_embed = cspu_embed_lookup(short_cart)
    short_buy_embed = cspu_embed_lookup(short_buy)

    short_click_level2_embed = level2_embed_lookup(short_click_level2)
    short_cart_level2_embed = level2_embed_lookup(short_cart_level2)
    short_buy_level2_embed = level2_embed_lookup(short_buy_level2)

    ## MASK
    sum_p_long_click_embed_masked = get_masked_embed(cfg.long_seq_len, cfg.embed_dim)([long_click, long_click_embed])
    sum_p_long_cart_embed_masked = get_masked_embed(cfg.long_seq_len, cfg.embed_dim)([long_cart, long_cart_embed])
    sum_p_long_buy_embed_masked = get_masked_embed(cfg.long_seq_len, cfg.embed_dim)([long_buy, long_buy_embed])

    sum_p_long_buy_level2_embed_masked = get_masked_embed(cfg.long_seq_len, cfg.embed_dim)(
        [long_buy_level2, long_buy_level2_embed])
    sum_p_long_cart_level2_embed_masked = get_masked_embed(cfg.long_seq_len, cfg.embed_dim)(
        [long_cart_level2, long_cart_level2_embed])
    sum_p_long_click_level2_embed_masked = get_masked_embed(cfg.long_seq_len, cfg.embed_dim)(
        [long_click_level2, long_click_level2_embed])

    sum_p_short_click_embed_masked = get_masked_embed(cfg.short_seq_len, cfg.embed_dim)(
        [short_click, short_click_embed])
    sum_p_short_cart_embed_masked = get_masked_embed(cfg.short_seq_len, cfg.embed_dim)([short_cart, short_cart_embed])
    sum_p_short_buy_embed_masked = get_masked_embed(cfg.short_seq_len, cfg.embed_dim)([short_buy, short_buy_embed])

    sum_p_short_click_level2_embed_masked = get_masked_embed(cfg.short_seq_len, cfg.embed_dim)(
        [short_click_level2, short_click_level2_embed])
    sum_p_short_cart_level2_embed_masked = get_masked_embed(cfg.short_seq_len, cfg.embed_dim)(
        [short_cart_level2, short_cart_level2_embed])
    sum_p_short_buy_level2_embed_masked = get_masked_embed(cfg.short_seq_len, cfg.embed_dim)(
        [short_buy_level2, short_buy_level2_embed])

    ## ATTENTION
    target_embs = tf.concat([cspu_embed, lv2_embed], -1)
    # 1. long sequence
    long_click_embs = tf.concat([sum_p_long_click_embed_masked, sum_p_long_click_level2_embed_masked], -1)
    long_cart_embs = tf.concat([sum_p_long_cart_embed_masked, sum_p_long_cart_level2_embed_masked], -1)
    long_buy_embs = tf.concat([sum_p_long_buy_embed_masked, sum_p_long_buy_level2_embed_masked], -1)

    # transformer layer
    d_model = long_cart_embs.shape[-1]
    long_cart_padding_mask_list = padding_mask(long_cart)
    long_buy_padding_mask_list = padding_mask(long_buy)

    long_cart_transformer = Encoder(1, d_model, 4, 256, cfg.long_seq_len, True)
    long_buy_transformer = Encoder(1, d_model, 4, 256, cfg.long_seq_len, True)

    long_cart_output = long_cart_transformer(long_cart_embs, long_cart_padding_mask_list)
    long_buy_output = long_buy_transformer(long_buy_embs, long_buy_padding_mask_list)
    print("long_buy_output", long_buy_output)

    # 2. short sequence
    short_click_embs = tf.concat([sum_p_short_click_embed_masked, sum_p_short_click_level2_embed_masked], -1)
    short_cart_embs = tf.concat([sum_p_short_cart_embed_masked, sum_p_short_cart_level2_embed_masked], -1)
    short_buy_embs = tf.concat([sum_p_short_buy_embed_masked, sum_p_short_buy_level2_embed_masked], -1)

    d_model = short_cart_embs.shape[-1]
    short_cart_padding_mask_list = padding_mask(short_cart)
    short_buy_padding_mask_list = padding_mask(short_buy)

    short_cart_transformer = Encoder(1, d_model, 4, 256, cfg.short_seq_len, True)
    short_buy_transformer = Encoder(1, d_model, 4, 256, cfg.short_seq_len, True)

    short_cart_output = short_cart_transformer(short_cart_embs, short_cart_padding_mask_list)
    short_buy_output = short_buy_transformer(short_buy_embs, short_buy_padding_mask_list)
    print("short_buy_output", short_buy_output)

    # target attention
    long_click_din = get_weight_sum_embed(cfg.long_seq_len, cfg.embed_dim * 2)([target_embs, long_click_embs])
    long_cart_din = get_weight_sum_embed(cfg.long_seq_len, cfg.embed_dim * 2)([target_embs, long_cart_output])
    long_buy_din = get_weight_sum_embed(cfg.long_seq_len, cfg.embed_dim * 2)([target_embs, long_buy_output])
    short_click_din = get_weight_sum_embed(cfg.short_seq_len, cfg.embed_dim * 2)([target_embs, short_click_embs])
    short_cart_din = get_weight_sum_embed(cfg.short_seq_len, cfg.embed_dim * 2)([target_embs, short_cart_output])
    short_buy_din = get_weight_sum_embed(cfg.short_seq_len, cfg.embed_dim * 2)([target_embs, short_buy_output])

    # for dense
    pooling_long_click_din = tf.squeeze(long_click_din, 1)
    pooling_long_cart_din = tf.squeeze(long_cart_din, 1)
    pooling_long_buy_din = tf.squeeze(long_buy_din, 1)
    pooling_short_click_din = tf.squeeze(short_click_din, 1)
    pooling_short_cart_din = tf.squeeze(short_cart_din, 1)
    pooling_short_buy_din = tf.squeeze(short_buy_din, 1)
    print("pooling_short_buy_din", pooling_short_buy_din)

    ##  ----------------- 行为序列 end！！ -----------------

    ## ----------------- 2.2 sparse embedding ---------------
    ## 离散特征处理
    pair_offsets = tf.expand_dims(tf.cumsum([0] + cfg.pair_field_dims[:-1], axis=0), axis=0)  ## [1, n]
    goods_offsets = tf.expand_dims(tf.cumsum([0] + cfg.goods_field_dims[:-1], axis=0), axis=0)  ## [1, n]
    bucket_user_offsets = tf.expand_dims(tf.cumsum([0] + cfg.bucket_user_dims[:-1], axis=0), axis=0)  ## [1, n]
    bucket_goods_offsets = tf.expand_dims(tf.cumsum([0] + cfg.bucket_goods_dims[:-1], axis=0), axis=0)  ## [1, n]
    bucket_pair_offsets = tf.expand_dims(tf.cumsum([0] + cfg.bucket_pair_dims[:-1], axis=0), axis=0)  ## [1, n]
    context_offsets = tf.expand_dims(tf.cumsum([0] + cfg.context_dims[:-1], axis=0), axis=0)  ## [1, n]
    bucket_user_cspu_offsets = tf.expand_dims(tf.cumsum([0] + cfg.bucket_user_cspu_dims[:-1], axis=0),
                                              axis=0)  ## [1, n]
    bucket_ozid_cspu_offsets = tf.expand_dims(tf.cumsum([0] + cfg.bucket_ozid_cspu_dims[:-1], axis=0),
                                              axis=0)  ## [1, n]
    bucket_user_behavior_offsets = tf.expand_dims(tf.cumsum([0] + cfg.bucket_user_behavior_dims[:-1], axis=0),
                                                  axis=0)  ## [1, n]
    bucket_goods_gross_offsets = tf.expand_dims(tf.cumsum([0] + cfg.bucket_goods_gross_dims[:-1], axis=0),
                                                axis=0)  ## [1, n]

    pair_feature = pair_feature + pair_offsets
    goods_sparse = goods_sparse + goods_offsets
    bucket_user_box_obj = bucket_user_box_obj + bucket_user_offsets
    bucket_goods_box_obj = bucket_goods_box_obj + bucket_goods_offsets
    bucket_pair_box_obj = bucket_pair_box_obj + bucket_pair_offsets
    context_features = context_features + context_offsets
    bucket_user_cspu_obj = bucket_user_cspu_obj + bucket_user_cspu_offsets
    bucket_ozid_cspu_obj = bucket_ozid_cspu_obj + bucket_ozid_cspu_offsets
    bucket_user_behavior_obj = bucket_user_behavior_obj + bucket_user_behavior_offsets
    bucket_goods_gross_obj = bucket_goods_gross_obj + bucket_goods_gross_offsets

    pair_feature = tf.clip_by_value(pair_feature, clip_value_min=0, clip_value_max=sum(cfg.pair_field_dims) - 1)
    goods_sparse = tf.clip_by_value(goods_sparse, clip_value_min=0, clip_value_max=sum(cfg.goods_field_dims) - 1)
    bucket_user_box_obj = tf.clip_by_value(bucket_user_box_obj, clip_value_min=0,
                                           clip_value_max=sum(cfg.bucket_user_dims) - 1)
    bucket_goods_box_obj = tf.clip_by_value(bucket_goods_box_obj, clip_value_min=0,
                                            clip_value_max=sum(cfg.bucket_goods_dims) - 1)
    bucket_pair_box_obj = tf.clip_by_value(bucket_pair_box_obj, clip_value_min=0,
                                           clip_value_max=sum(cfg.bucket_pair_dims) - 1)
    bucket_user_cspu_obj = tf.clip_by_value(bucket_user_cspu_obj, clip_value_min=0,
                                            clip_value_max=sum(cfg.bucket_user_cspu_dims) - 1)
    bucket_ozid_cspu_obj = tf.clip_by_value(bucket_ozid_cspu_obj, clip_value_min=0,
                                            clip_value_max=sum(cfg.bucket_ozid_cspu_dims) - 1)
    bucket_user_behavior_obj = tf.clip_by_value(bucket_user_behavior_obj, clip_value_min=0,
                                                clip_value_max=sum(cfg.bucket_user_behavior_dims) - 1)
    bucket_goods_gross_obj = tf.clip_by_value(bucket_goods_gross_obj, clip_value_min=0,
                                              clip_value_max=sum(cfg.bucket_goods_gross_dims) - 1)

    ## ----------------- 2.3 实时特征 ---------------

    ## 实时离散特征处理
    ### ------------ 实时特征等距分箱处理 ---------
    ##   商品特征进行log分桶，用户和pair都进行等距分桶，间隔3为一个桶，最大10个桶。
    ### -----------------------------------------
    # realtime_user = tf.clip_by_value(realtime_user, clip_value_min=0, clip_value_max=29)
    realtime_pair_click = tf.cast(tf.clip_by_value(realtime_pair_click, clip_value_min=0, clip_value_max=2), tf.int32)
    # realtime_cross_category_front = tf.clip_by_value(realtime_cross_category_front, clip_value_min=0, clip_value_max=29)
    realtime_back_category = tf.clip_by_value(realtime_back_category, clip_value_min=0, clip_value_max=29)
    realtime_goods = tf.clip_by_value(realtime_goods, clip_value_min=0, clip_value_max=100000000)
    realtime_passtime = tf.clip_by_value(realtime_passtime, clip_value_min=0, clip_value_max=129600)
    realtime_user_group = tf.clip_by_value(realtime_user_group, clip_value_min=0, clip_value_max=129600)

    # realtime_user = tf.cast(tf.math.floordiv(realtime_user, 5), tf.int32)
    # realtime_cross_category_front = tf.cast(tf.floordiv(realtime_cross_category_front, 5), tf.int32)
    realtime_back_category = tf.cast(tf.math.floordiv(realtime_back_category, 5), tf.int32)
    realtime_goods = tf.cast(tf.math.log(realtime_goods + 1), tf.int32)
    realtime_goods = tf.clip_by_value(realtime_goods, clip_value_min=0, clip_value_max=30)
    realtime_passtime_hour = tf.math.floordiv(realtime_passtime, 60, name="pst_hour")
    realtime_passtime_day = tf.add(tf.math.floordiv(realtime_passtime, 1440), 23, name="pst_day")
    realtime_passtime = tf.cast(tf.where(realtime_passtime < 1440.0, realtime_passtime_hour, realtime_passtime_day),
                                tf.int32)
    realtime_user_group = tf.cast(tf.math.log(realtime_user_group + 1), tf.int32)
    realtime_user_group = tf.clip_by_value(realtime_user_group, clip_value_min=0, clip_value_max=30)

    ### ------------ 实时特征等距分箱处理 done ---------

    realtime_back_category_offsets = tf.expand_dims(tf.cumsum([0] + cfg.realtime_back_category_dims[:-1], axis=0),
                                                    axis=0)  ## [1, n]
    realtime_back_category = realtime_back_category + realtime_back_category_offsets
    realtime_goods_offsets = tf.expand_dims(tf.cumsum([0] + cfg.realtime_goods_dims[:-1], axis=0), axis=0)
    realtime_goods = realtime_goods + realtime_goods_offsets
    realtime_passtime_offsets = tf.expand_dims(tf.cumsum([0] + cfg.realtime_passtime_dims[:-1], axis=0), axis=0)
    realtime_passtime = realtime_passtime + realtime_passtime_offsets
    realtime_user_group_offsets = tf.expand_dims(tf.cumsum([0] + cfg.realtime_user_group_dims[:-1], axis=0), axis=0)
    realtime_user_group = realtime_user_group + realtime_user_group_offsets

    ## ----------------- 2.4 各种embedding ---------------

    ## embedding
    pair_feature_embed = layers.Embedding(input_dim=sum(cfg.pair_field_dims),
                                          output_dim=cfg.embed_dim,
                                          embeddings_initializer=cfg.embed_init,
                                          embeddings_regularizer=cfg.kernel_regular,
                                          input_length=None)(pair_feature)

    goods_sparse_embed = layers.Embedding(input_dim=sum(cfg.goods_field_dims),
                                          output_dim=cfg.embed_dim,
                                          embeddings_initializer=cfg.embed_init,
                                          embeddings_regularizer=cfg.kernel_regular,
                                          input_length=None)(goods_sparse)

    bucket_user_box_obj_embed = layers.Embedding(input_dim=sum(cfg.bucket_user_dims),
                                                 output_dim=cfg.embed_dim,
                                                 embeddings_initializer=cfg.embed_init,
                                                 embeddings_regularizer=cfg.kernel_regular,
                                                 input_length=None)(bucket_user_box_obj)
    bucket_goods_box_obj_embed = layers.Embedding(input_dim=sum(cfg.bucket_goods_dims),
                                                  output_dim=cfg.embed_dim,
                                                  embeddings_initializer=cfg.embed_init,
                                                  embeddings_regularizer=cfg.kernel_regular,
                                                  input_length=None)(bucket_goods_box_obj)
    bucket_pair_box_obj_embed = layers.Embedding(input_dim=sum(cfg.bucket_pair_dims),
                                                 output_dim=cfg.embed_dim,
                                                 embeddings_initializer=cfg.embed_init,
                                                 embeddings_regularizer=cfg.kernel_regular,
                                                 input_length=None)(bucket_pair_box_obj)

    realtime_back_category_embed = layers.Embedding(input_dim=sum(cfg.realtime_back_category_dims),
                                                    output_dim=cfg.embed_dim,
                                                    embeddings_initializer=cfg.embed_init,
                                                    embeddings_regularizer=cfg.kernel_regular,
                                                    input_length=None)(realtime_back_category)

    realtime_goods_embed = layers.Embedding(input_dim=sum(cfg.realtime_goods_dims),
                                            output_dim=cfg.embed_dim,
                                            embeddings_initializer=cfg.embed_init,
                                            embeddings_regularizer=cfg.kernel_regular,
                                            input_length=None)(realtime_goods)

    realtime_pair_click_embed = layers.Embedding(input_dim=3,
                                                 output_dim=cfg.embed_dim,
                                                 embeddings_initializer=cfg.embed_init,
                                                 embeddings_regularizer=cfg.kernel_regular,
                                                 input_length=None)(realtime_pair_click)
    realtime_pair_click_embed = tf.expand_dims(realtime_pair_click_embed, 1)

    context_embed = layers.Embedding(input_dim=sum(cfg.context_dims),
                                     output_dim=cfg.embed_dim,
                                     embeddings_initializer=cfg.embed_init,
                                     embeddings_regularizer=cfg.kernel_regular,
                                     input_length=None)(context_features)

    realtime_passtime_embed = layers.Embedding(input_dim=sum(cfg.realtime_passtime_dims),
                                               output_dim=cfg.embed_dim,
                                               embeddings_initializer=cfg.embed_init,
                                               embeddings_regularizer=cfg.kernel_regular,
                                               input_length=None)(realtime_passtime)

    realtime_user_group_embed = layers.Embedding(input_dim=sum(cfg.realtime_user_group_dims),
                                                 output_dim=cfg.embed_dim,
                                                 embeddings_initializer=cfg.embed_init,
                                                 embeddings_regularizer=cfg.kernel_regular,
                                                 input_length=None)(realtime_user_group)

    bucket_user_cspu_obj_embed = layers.Embedding(input_dim=sum(cfg.bucket_user_cspu_dims),
                                                  output_dim=cfg.embed_dim,
                                                  embeddings_initializer=cfg.embed_init,
                                                  embeddings_regularizer=cfg.kernel_regular,
                                                  input_length=None)(bucket_user_cspu_obj)
    bucket_ozid_cspu_obj_embed = layers.Embedding(input_dim=sum(cfg.bucket_ozid_cspu_dims),
                                                  output_dim=cfg.embed_dim,
                                                  embeddings_initializer=cfg.embed_init,
                                                  embeddings_regularizer=cfg.kernel_regular,
                                                  input_length=None)(bucket_ozid_cspu_obj)

    bucket_user_behavior_obj_embed = layers.Embedding(input_dim=sum(cfg.bucket_user_behavior_dims),
                                                      output_dim=cfg.embed_dim,
                                                      embeddings_initializer=cfg.embed_init,
                                                      embeddings_regularizer=cfg.kernel_regular,
                                                      input_length=None)(bucket_user_behavior_obj)
    bucket_goods_gross_obj_embed = layers.Embedding(input_dim=sum(cfg.bucket_goods_gross_dims),
                                                    output_dim=cfg.embed_dim,
                                                    embeddings_initializer=cfg.embed_init,
                                                    embeddings_regularizer=cfg.kernel_regular,
                                                    input_length=None)(bucket_goods_gross_obj)

    ## --------- all embedding concat -----------
    embed_fm = tf.concat([realtime_passtime_embed, context_embed, realtime_back_category_embed, realtime_goods_embed,
                          realtime_pair_click_embed, realtime_user_group_embed,
                          pair_feature_embed,
                          goods_sparse_embed,
                          bucket_user_box_obj_embed, bucket_goods_box_obj_embed, bucket_pair_box_obj_embed,
                          cspu_embed, supplier_embed, lv2_embed,
                          bucket_user_cspu_obj_embed, bucket_ozid_cspu_obj_embed, bucket_user_behavior_obj_embed,
                          bucket_goods_gross_obj_embed], axis=1, name="all_embed_concat")

    embed_deep = embed_fm
    deep_fc = tf.reshape(embed_deep, shape=(-1, embed_deep.shape[1] * embed_deep.shape[2]))
    deep_fc = tf.concat([deep_fc, pooling_long_click_din, pooling_long_cart_din, pooling_long_buy_din,
                         pooling_short_click_din, pooling_short_cart_din, pooling_short_buy_din], -1)
    print("concat embed", embed_fm)
    print("deep_fc", deep_fc)

    ### -------------------------------------------------------------
    ###                        三、 linear
    ### -------------------------------------------------------------
    linear1 = layers.Dense(1, activation=None, kernel_regularizer=cfg.kernel_regular, use_bias=True)(bucket_goods_raw)
    linear2 = layers.Embedding(input_dim=sum(cfg.pair_field_dims),
                               output_dim=1,
                               embeddings_initializer=cfg.embed_init,
                               embeddings_regularizer=cfg.kernel_regular,
                               input_length=None)(pair_feature)

    linear3 = layers.Embedding(input_dim=sum(cfg.realtime_back_category_dims),
                               output_dim=1,
                               embeddings_initializer=cfg.embed_init,
                               embeddings_regularizer=cfg.kernel_regular,
                               input_length=None)(realtime_back_category)

    linear4 = layers.Embedding(input_dim=sum(cfg.realtime_goods_dims),
                               output_dim=1,
                               embeddings_initializer=cfg.embed_init,
                               embeddings_regularizer=cfg.kernel_regular,
                               input_length=None)(realtime_goods)

    linear5 = layers.Embedding(input_dim=3,
                               output_dim=1,
                               embeddings_initializer=cfg.embed_init,
                               embeddings_regularizer=cfg.kernel_regular,
                               input_length=None)(realtime_pair_click)

    linear6 = layers.Embedding(input_dim=sum(cfg.bucket_pair_dims),
                               output_dim=1,
                               embeddings_initializer=cfg.embed_init,
                               embeddings_regularizer=cfg.kernel_regular,
                               input_length=None)(bucket_pair_box_obj)  ## [1, 62, embed_dim]

    linear7 = layers.Embedding(input_dim=sum(cfg.bucket_goods_dims),
                               output_dim=1,
                               embeddings_initializer=cfg.embed_init,
                               embeddings_regularizer=cfg.kernel_regular,
                               input_length=None)(bucket_goods_box_obj)  ## [1, 62, embed_dim]
    linear8 = layers.Embedding(input_dim=sum(cfg.realtime_passtime_dims),
                               output_dim=1,
                               embeddings_initializer=cfg.embed_init,
                               embeddings_regularizer=cfg.kernel_regular,
                               input_length=None)(realtime_passtime)
    linear9 = layers.Embedding(input_dim=sum(cfg.bucket_user_cspu_dims),
                               output_dim=1,
                               embeddings_initializer=cfg.embed_init,
                               embeddings_regularizer=cfg.kernel_regular,
                               input_length=None)(bucket_user_cspu_obj)
    linear10 = layers.Embedding(input_dim=sum(cfg.bucket_ozid_cspu_dims),
                                output_dim=1,
                                embeddings_initializer=cfg.embed_init,
                                embeddings_regularizer=cfg.kernel_regular,
                                input_length=None)(bucket_ozid_cspu_obj)
    linear11 = layers.Embedding(input_dim=sum(cfg.bucket_user_behavior_dims),
                                output_dim=1,
                                embeddings_initializer=cfg.embed_init,
                                embeddings_regularizer=cfg.kernel_regular,
                                input_length=None)(bucket_user_behavior_obj)
    linear12 = layers.Embedding(input_dim=sum(cfg.bucket_goods_gross_dims),
                                output_dim=1,
                                embeddings_initializer=cfg.embed_init,
                                embeddings_regularizer=cfg.kernel_regular,
                                input_length=None)(bucket_goods_gross_obj)
    linear13 = layers.Embedding(input_dim=sum(cfg.realtime_user_group_dims),
                                output_dim=1,
                                embeddings_initializer=cfg.embed_init,
                                embeddings_regularizer=cfg.kernel_regular,
                                input_length=None)(realtime_user_group)

    linear2 = tf.reduce_sum(tf.squeeze(linear2, 2), 1, keepdims=True)
    linear3 = tf.reduce_sum(tf.squeeze(linear3, 2), 1, keepdims=True)
    linear4 = tf.reduce_sum(tf.squeeze(linear4, 2), 1, keepdims=True)
    linear6 = tf.reduce_sum(tf.squeeze(linear6, 2), 1, keepdims=True)
    linear7 = tf.reduce_sum(tf.squeeze(linear7, 2), 1, keepdims=True)
    linear8 = tf.reduce_sum(tf.squeeze(linear8, 2), 1, keepdims=True)
    linear9 = tf.reduce_sum(tf.squeeze(linear9, 2), 1, keepdims=True)
    linear10 = tf.reduce_sum(tf.squeeze(linear10, 2), 1, keepdims=True)
    linear11 = tf.reduce_sum(tf.squeeze(linear11, 2), 1, keepdims=True)
    linear12 = tf.reduce_sum(tf.squeeze(linear12, 2), 1, keepdims=True)
    linear13 = tf.reduce_sum(tf.squeeze(linear13, 2), 1, keepdims=True)

    linear = tf.concat(
        [linear1, linear2, linear3, linear4, linear5, linear6, linear7, linear8, linear9, linear10, linear11, linear12, linear13],
        axis=1, name="linear_concat")  ## 不要想加

    ### -------------------------------------------------------------
    ###                        四、 fm  & deep
    ### -------------------------------------------------------------
    ## ---------------------- fm --------------------
    summed_features_emb = tf.reduce_sum(embed_fm, 1)
    summed_features_emb_square = tf.square(summed_features_emb)
    # square_sum part
    squared_features_emb = tf.square(embed_fm)
    squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)
    # second order
    fm = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  ## (None, embed_dim)

    ## --------------- deep ------------------
    deep_fc = tf.concat([bucket_user_raw, bucket_goods_raw, deep_fc], axis=-1, name="deep_input")

    print("deep 输入tensor： ", deep_fc)
    # deep_fc = layers.Dropout(rate=0.3)(deep_fc)
    deep_fc = layers.Dense(128, activation=None, kernel_regularizer=cfg.kernel_regular, use_bias=True)(deep_fc)
    deep_fc = layers.BatchNormalization(axis=-1, momentum=0.99)(deep_fc)
    deep_fc = layers.ReLU()(deep_fc)
    # deep_fc = layers.Dropout(rate=0.2)(deep_fc)
    deep_fc = layers.Dense(64, activation=None, kernel_regularizer=cfg.kernel_regular, use_bias=True)(deep_fc)
    deep_fc = layers.BatchNormalization(axis=-1, momentum=0.99)(deep_fc)
    deep_fc = layers.ReLU()(deep_fc)
    deep_fc = layers.Dense(32, activation=None, kernel_regularizer=cfg.kernel_regular, use_bias=True)(deep_fc)
    deep_fc = layers.BatchNormalization(axis=-1, momentum=0.99)(deep_fc)
    deep_fc = layers.ReLU()(deep_fc)

    concat_all = tf.concat([linear, fm, deep_fc], axis=1)
    concat_all = layers.Dense(1, kernel_regularizer=cfg.kernel_regular, activation=None, use_bias=True)(concat_all)
    output = tf.nn.sigmoid(concat_all, name="output")
    print("output:", output)
    return keras.Model(inputs=[inputs], outputs=[output])


def list_hdfs_tfrecords_file(path):
    cat = subprocess.Popen(["hdfs", "dfs", "-ls", "{}".format(path)], stdout=subprocess.PIPE)
    parquet_list = []
    # print(cat)
    pattern = re.compile(r"/user/.+part-r-\d+")
    for line in cat.stdout:
        if re.search(pattern, str(line)) is not None:
            # print(str(line))
            parquet_list.append(re.search(pattern, str(line)).group(0))
    return parquet_list


def dateRange(start, end, step=1, format="%Y-%m-%d"):
    ## dateRange("2017-01-01", "2017-01-03")
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days
    return [strftime(strptime(start, format) + datetime.timedelta(i), format) for i in range(0, days + 1, step)]


def get_hdfs_path_list(path, train_data_end_time, days):
    train_data_start_time = datetime.datetime.strptime(train_data_end_time,
                                                       '%Y-%m-%d') - datetime.timedelta(days)  ## 7天数据训练
    train_data_start_time = train_data_start_time.strftime('%Y-%m-%d')
    date_list = dateRange(train_data_start_time, train_data_end_time)
    dir_list = [os.path.join(path, t) for t in date_list]

    print("train parquet dataset dir: ", dir_list)
    hdfs_path = []
    for dir_tmp in dir_list:
        hdfs_path += list_hdfs_tfrecords_file(dir_tmp)
    hdfs_path = ["hdfs://difed{}".format(s) for s in hdfs_path]
    random.shuffle(hdfs_path)
    train_path = hdfs_path[:-10]  ## 训练集
    val_path = hdfs_path[-10:]  ## 验证集

    random.shuffle(train_path)
    random.shuffle(val_path)
    return train_path, val_path


def get_hdfs_path_list_test(path, train_data_end_time, days):
    train_data_start_time = datetime.datetime.strptime(train_data_end_time,
                                                       '%Y-%m-%d') - datetime.timedelta(days)  ## 7天数据训练
    train_data_start_time = train_data_start_time.strftime('%Y-%m-%d')
    date_list = dateRange(train_data_start_time, train_data_end_time)
    dir_list = [os.path.join(path, t) for t in date_list]

    print("train parquet dataset dir: ", dir_list)
    hdfs_path = []
    for dir_tmp in dir_list:
        hdfs_path += list_hdfs_tfrecords_file(dir_tmp)
    hdfs_path = ["hdfs://difed{}".format(s) for s in hdfs_path]
    random.shuffle(hdfs_path)
    test_path = hdfs_path[:-10]  ## 训练集
    val_path = hdfs_path[-10:]  ## 验证集
    return test_path, val_path


def dataset_pipeline(hdfs_path_list, epochs):
    features = {
        'user_id': tf.io.FixedLenFeature([1], tf.string),
        'goods_id': tf.io.FixedLenFeature([1], tf.string),
        'label': tf.io.FixedLenFeature([1], tf.float32),
        'context_feature': tf.io.FixedLenFeature([cfg.field_length['context_feature']], tf.float32),
        'realtime_features': tf.io.FixedLenFeature([cfg.field_length['realtime_features']], tf.float32),
        'features_pair_4': tf.io.FixedLenFeature([cfg.field_length['features_pair_4']], tf.int64),
        'goods_sparse_features': tf.io.FixedLenFeature([cfg.field_length['goods_sparse_features']], tf.int64),
        'sequence': tf.io.FixedLenFeature([cfg.field_length['sequence']], tf.int64),
        'bucket_user_features': tf.io.FixedLenFeature([cfg.field_length['bucket_user_features']], tf.float32),
        'bucket_goods_features': tf.io.FixedLenFeature([cfg.field_length['bucket_goods_features']], tf.float32),
        'bucket_pair_features': tf.io.FixedLenFeature([cfg.field_length['bucket_pair_features']], tf.float32),
        'bucket_user_cspu_features': tf.io.FixedLenFeature([cfg.field_length['bucket_user_cspu_features']], tf.float32),
        'bucket_ozid_cspu_features': tf.io.FixedLenFeature([cfg.field_length['bucket_ozid_cspu_features']], tf.float32),
        'bucket_user_behavior_features': tf.io.FixedLenFeature([cfg.field_length['bucket_user_behavior_features']],
                                                               tf.float32),
        'bucket_goods_gross_features': tf.io.FixedLenFeature([cfg.field_length['bucket_goods_gross_features']],
                                                             tf.float32)
    }

    def parse_record(record):
        parsed = tf.io.parse_single_example(record, features=features)
        label = parsed['label']

        features_all = []
        for col in cfg.feature_concat_col:
            tmp_feature = tf.cast(parsed[col], tf.float32)
            features_all.append(tmp_feature)

        feature_input = tf.concat(features_all, axis=-1)
        user_id = parsed['user_id']
        goods_id = parsed['goods_id']
        data = ({
                    "inputs": feature_input,
                    "user_id": user_id,
                    "goods_id": goods_id
                },
                {
                    "tf_op_layer_Sigmoid": label,
                })
        return data

    dataset = tf.data.TFRecordDataset(hdfs_path_list)
    dataset = dataset.map(parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=cfg.shuffle_buffer_size)
    dataset = dataset.repeat(epochs)  ## epoch
    dataset = dataset.prefetch(buffer_size=cfg.prefetch_buffer_size)
    dataset = dataset.batch(batch_size=cfg.batch_size)
    return dataset


def print_process_info(rst):
    if rst.returncode == 0:
        print("success:", rst)
    else:
        print("error:", rst)


def mkdirs(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)
        print("mkdirs: {} ,ok!".format(pth))


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

    return focal_loss_fixed


def dice_loss(smooth=1e-6):
    def dice_loss_fixed(y_true, y_pred):
        # flatten label and prediction tensors
        intersection = K.sum(y_true * y_pred)
        dice = (2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)

        return 1 - dice

    return dice_loss_fixed


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--embed_dim', type=int, help='隐向量维度')
    parser.add_argument('--train_time', type=str, help='训练数据时间')
    parser.add_argument('--test_time', type=str, help='测试数据时间')
    parser.add_argument('--name', type=str, help='模型备注')
    parser.add_argument("--tfrecords", type=str, help="训练数据的主目录hdfs路径")
    parser.add_argument("--workspace", type=str, help="workspace")
    parser.add_argument("--mode", type=str, help="train, test")
    parser.add_argument("--predict_hdfs_dir", type=str, help="训练数据hdfs文件夹")
    parser.add_argument("--steps_per_epoch", type=int, help="每个epoch的step数")
    parser.add_argument("--validation_steps", type=int, help="验证集的step数")
    parser.add_argument("--model_path", type=str, help="本地模型路径")
    parser.add_argument("--model_hdfs_dir", type=str, help="模型的hdfs路径")
    parser.add_argument("--predict_local_path", type=str, help="模型的hdfs路径")

    args = parser.parse_args()
    print("args: ", args)

    cfg.batch_size = args.batch_size
    cfg.init_lr = args.lr
    cfg.embed_dim = args.embed_dim
    cfg.workspace_dir = args.workspace
    cfg.steps_per_epoch = args.steps_per_epoch
    cfg.validation_steps = args.validation_steps
    mode = args.mode.split(",")

    log_dir = os.path.join(cfg.workspace_dir, 'tensorboard/{}_{}'.format(cfg.embed_dim, args.name))
    checkpoint_dir = os.path.join(cfg.workspace_dir, 'checkpoint')
    mkdirs(log_dir)
    mkdirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, 'model_{}_{}.h5'.format(cfg.embed_dim, args.name))

    tfrecords = args.tfrecords
    train_data_end_time = args.train_time
    test_data_end_time = args.test_time
    train_path, train_path_others = get_hdfs_path_list(tfrecords, train_data_end_time, days=21)
    test_path, val_path = get_hdfs_path_list_test(tfrecords, test_data_end_time, days=0)

    train_dataset = dataset_pipeline(hdfs_path_list=train_path + train_path_others, epochs=cfg.epochs)
    val_dataset = dataset_pipeline(hdfs_path_list=val_path, epochs=cfg.epochs * 10)
    test_dataset = dataset_pipeline(hdfs_path_list=test_path + val_path, epochs=1)

    # for feature, label in test_dataset.take(3):
    #     print(feature)

    my_callbacks = [
        keras.callbacks.EarlyStopping(patience=6),
        keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                        save_weights_only=True,
                                        monitor='val_auc',
                                        mode='max',
                                        save_best_only=True),
        keras.callbacks.TensorBoard(log_dir=log_dir),
        keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.2,
                                          patience=2, min_lr=1e-5, mode='max')
    ]

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = get_fm_model()
        ## load model
        # if os.path.exists(checkpoint_path):
        #     model.load_weights(checkpoint_path, by_name=True)
        #     print("load model: {}".format(checkpoint_path))

        model.compile(optimizer=keras.optimizers.Adam(cfg.init_lr),
                      loss={
                          "tf_op_layer_Sigmoid": keras.losses.BinaryCrossentropy(from_logits=False),
                      },
                      metrics={
                          "tf_op_layer_Sigmoid": [
                              keras.metrics.AUC(),
                              keras.metrics.BinaryAccuracy(),
                          ],
                      },
                      loss_weights={
                          "tf_op_layer_Sigmoid": 1.0,
                      },
                      )

        if "train" in mode:
            model.fit(train_dataset,
                      validation_data=val_dataset,
                      epochs=cfg.epochs,
                      steps_per_epoch=cfg.steps_per_epoch,
                      callbacks=my_callbacks,
                      validation_steps=cfg.validation_steps,
                      verbose=1)

            ## save pd格式
            model.load_weights(checkpoint_path, by_name=True)
            pb_save_path = os.path.join(checkpoint_dir, "{}/{}".format(args.name, args.train_time), "1")
            tf.keras.models.save_model(model, pb_save_path)

            ### 把model上传到hdfs
            model_hdfs_dir = args.model_hdfs_dir
            rst = subprocess.run(["hadoop", "fs", "-mkdir", "-p", "hdfs://difed{}".format(model_hdfs_dir)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print_process_info(rst)

            rst = subprocess.run(
                ["hadoop", "fs", "-put", "-f", pb_save_path.replace("/1", ""), "hdfs://difed{}".format(model_hdfs_dir)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print_process_info(rst)

        # model.load_weights(checkpoint_path, by_name=True) ## 加载存储的模型

        if "predict" in mode:
            test_dataset = dataset_pipeline(hdfs_path_list=test_path + val_path, epochs=1)
            model = keras.models.load_model(args.model_path)


            def test_step(features):
                predictions = model(features, training=False)
                return predictions


            @tf.function
            def distributed_test_step(dataset_inputs):
                return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))


            predict_local_path = args.predict_local_path
            file = open(predict_local_path, "w+")
            file.write("user_id,goods_id,label,probability\n")
            for step, (features, labels) in enumerate(test_dataset):
                features_inputs = {"inputs": features['inputs'], }
                user_id = features['user_id'].numpy()
                goods_id = features['goods_id'].numpy()
                label = labels['tf_op_layer_Sigmoid'].numpy()
                predictions = distributed_test_step(features_inputs)
                predictions = predictions.numpy()
                rst = ""
                for j in range(predictions.shape[0]):
                    prob = predictions[j][0]
                    uid = bytes.decode(user_id[j][0])
                    gid = bytes.decode(goods_id[j][0])
                    label_tmp = label[j][0]
                    rst += "{},{},{},{:.8f}\n".format(uid, gid, label_tmp, prob)
                file.write(rst)
                if step % 100 == 0:
                    print('[predict] sample num: {}, {}'.format(step * cfg.batch_size,
                                                                "{},{},{},{:.8f}\n".format(uid, gid, label_tmp, prob)))
            file.close()

            import pyarrow.csv as pv
            import pyarrow.parquet as pq
            import pyarrow as pa

            table = pv.read_csv(predict_local_path)
            schema = pa.schema([
                pa.field("user_id", "str", True),
                pa.field("goods_id", "str", True),
                pa.field("label", "float32", True),
                pa.field("probability", "float32", True)])
            t2 = table.cast(schema)
            parquet_local_path = predict_local_path + '.parquet'
            pq.write_table(t2, parquet_local_path)

            print(">>>>>>>>>>> predict done!")

            rst = subprocess.run(["hadoop", "fs", "-mkdir", "-p", "hdfs://difed{}".format(args.predict_hdfs_dir)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print_process_info(rst)
            rst = subprocess.run(
                ["hadoop", "fs", "-put", "-f", parquet_local_path, "hdfs://difed{}".format(args.predict_hdfs_dir)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print_process_info(rst)
            rst = subprocess.run(["rm", predict_local_path],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print_process_info(rst)
            rst = subprocess.run(["rm", parquet_local_path],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print_process_info(rst)

        if "pb" in mode:
            pb_save_path = os.path.join(checkpoint_dir, "{}_{}".format(args.train_time, args.name), "1")
            tf.keras.models.save_model(model, pb_save_path)

        if "evaluate" in mode:
            test_dataset = dataset_pipeline(hdfs_path_list=test_path + val_path, epochs=1)

            results = model.evaluate(test_dataset,
                                     verbose=1,
                                     steps=10000)
            print("test: ", results)
