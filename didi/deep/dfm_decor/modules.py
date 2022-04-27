# -*- coding:utf-8 -*-
# @Time: 2021/11/6 下午5:41
# @Author: Haiton

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


class ProcessInputs(tf.keras.Model):
    def __init__(self, cfg):
        super(ProcessInputs, self).__init__()
        self.cfg = cfg
        self.process_context = ProcessContext(self.cfg)
        self.process_sparse_id = ProcessSparseID(self.cfg)
        self.process_sequence = ProcessSequence(self.cfg)
        self.process_user = ProcessUser(self.cfg)
        self.process_goods = ProcessGoods(self.cfg)
        self.process_pairs = ProcessPairs(self.cfg)

    def call(self, x):
        realtime_features = x[:, :self.cfg.stride[0]]
        context_features = tf.cast(x[:, self.cfg.stride[6]:self.cfg.stride[7]], tf.int32, name='context_features')
        realtime_features, context_features = self.process_context((realtime_features, context_features))

        goods_sparse_features = tf.cast(x[:, self.cfg.stride[1]:self.cfg.stride[2]], tf.int32, name='goods_sparse_features')
        goods_sparse, cspu_idx, supplier_idx, lv2_idx = self.process_sparse_id(goods_sparse_features)

        sequence = tf.cast(x[:, self.cfg.stride[2]:self.cfg.stride[3]], tf.int32, name='sequence')
        long_seq, short_seq = self.process_sequence(sequence)

        bucket_user_raw, bucket_user_box_obj = self.process_user(x[:, self.cfg.stride[3]:self.cfg.stride[4]])

        bucket_goods_raw, bucket_goods_box_obj, bucket_goods_gross_obj = self.process_goods((x[:, self.cfg.stride[4]: self.cfg.stride[5]],
                                                                                             x[:, self.cfg.stride[10]: self.cfg.stride[11]]))

        pair_feature, \
        bucket_pair_box_obj, \
        bucket_user_cspu_obj, \
        bucket_ozid_cspu_obj, \
        bucket_user_behavior_obj = self.process_pairs((
            x[:, self.cfg.stride[0]: self.cfg.stride[1]],
            x[:, self.cfg.stride[5]: self.cfg.stride[6]],
            x[:, self.cfg.stride[7]: self.cfg.stride[8]],
            x[:, self.cfg.stride[8]: self.cfg.stride[9]],
            x[:, self.cfg.stride[9]: self.cfg.stride[10]]
        ))

        return (realtime_features, context_features), \
               (goods_sparse, cspu_idx, supplier_idx, lv2_idx), \
               (long_seq, short_seq), \
               (bucket_user_raw, bucket_user_box_obj), \
               (bucket_goods_raw, bucket_goods_box_obj, bucket_goods_gross_obj), \
               (pair_feature, bucket_pair_box_obj, bucket_user_cspu_obj, bucket_ozid_cspu_obj, bucket_user_behavior_obj)

    def get_config(self):
        config = super(ProcessInputs, self).get_config().copy()
        config.update({
            'process_context': self.process_context,
            'process_sparse_id': self.process_sparse_id,
            'process_sequence': self.process_sequence,
            'process_user': self.process_user,
            'process_goods': self.process_goods,
            'process_pairs': self.process_pairs
        })
        return config


class WeightedSum(tf.keras.Model):
    def __init__(self, cfg, seq_len, name):
        super(WeightedSum, self).__init__(name=name)
        self.cfg = cfg
        self.seq_len = seq_len
        self.dense1 = layers.Dense(20, activation=None, use_bias=True)
        self.dense2 = layers.Dense(1, activation="sigmoid", use_bias=True)
        self.activation = layers.PReLU()

    def call(self, x):
        embed_target_in, seq_embed_in = x
        embed_target_tile = tf.tile(embed_target_in, [1, self.seq_len, 1])  # (None, 100, 32)
        data_concat = tf.concat(
            [embed_target_tile, seq_embed_in, embed_target_tile - seq_embed_in, embed_target_tile * seq_embed_in],
            axis=2)  # DIN 中的attention
        att_w = self.dense1(data_concat)  # (None, 100, 128) ## 减少参数所以用了个8
        att_w = self.activation(att_w)  # DIN论文中说这个对离散embedding特征学的比较好
        att_w = self.dense2(att_w)  # (None, 100, 1)
        seq_embed = att_w * seq_embed_in  # (None, 100, 1) * (None, 100, 32)
        seq_embed = tf.reduce_sum(seq_embed, axis=1)  # (None, 32)
        seq_embed = tf.expand_dims(seq_embed, axis=1)  # (None, 1, 32)
        return seq_embed

    def get_config(self):
        config = super(WeightedSum, self).get_config().copy()
        config.update({
            'seq_len': self.seq_len,
            'dense1': self.dense1,
            'dense2': self.dense2,
            'activation': self.activation
        })
        return config


class MaskEmbedding(tf.keras.Model):
    def __init__(self, cfg, name):
        super(MaskEmbedding, self).__init__(name=name)
        self.cfg = cfg

    def call(self, x):
        seq_in, seq_embed_in = x
        seq_in = tf.cast(seq_in, tf.int32)
        g = tf.greater(seq_in, tf.zeros((1,), dtype=tf.int32))
        data_mask = tf.cast(g, tf.float32)  # (None, 100)
        data_mask = tf.expand_dims(data_mask, axis=2)  # (None, 100, 1)
        data_embed_masked = tf.multiply(seq_embed_in, data_mask)  # (None, 100, 1) *  (None, 100, 32), broadcast
        return data_embed_masked


class BuildSequence(tf.keras.Model):
    def __init__(self, cfg):
        super(BuildSequence, self).__init__()
        self.cfg = cfg
        self.mask_long_clk = MaskEmbedding(self.cfg, 'mask_long_clk')
        self.mask_long_cart = MaskEmbedding(self.cfg, 'mask_long_cart')
        self.mask_long_buy = MaskEmbedding(self.cfg, 'mask_long_buy')
        self.mask_long_clk_lv2 = MaskEmbedding(self.cfg, 'mask_long_clk_lv2')
        self.mask_long_cart_lv2 = MaskEmbedding(self.cfg, 'mask_long_cart_lv2')
        self.mask_long_buy_lv2 = MaskEmbedding(self.cfg, 'mask_long_buy_lv2')
        self.mask_short_clk = MaskEmbedding(self.cfg, 'mask_short_clk')
        self.mask_short_cart = MaskEmbedding(self.cfg, 'mask_short_cart')
        self.mask_short_buy = MaskEmbedding(self.cfg, 'mask_short_buy')
        self.mask_short_clk_lv2 = MaskEmbedding(self.cfg, 'mask_short_clk_lv2')
        self.mask_short_cart_lv2 = MaskEmbedding(self.cfg, 'mask_short_cart_lv2')
        self.mask_short_buy_lv2 = MaskEmbedding(self.cfg, 'mask_short_buy_lv2')
        d_model = 2 * cfg.embed_dim
        self.long_cart_transformer = Encoder(1, d_model, 4, 256, self.cfg.long_seq_len, True)
        self.long_buy_transformer = Encoder(1, d_model, 4, 256, self.cfg.long_seq_len, True)
        self.short_cart_transformer = Encoder(1, d_model, 4, 256, self.cfg.short_seq_len, True)
        self.short_buy_transformer = Encoder(1, d_model, 4, 256, self.cfg.short_seq_len, True)
        self.long_click_din = WeightedSum(self.cfg, self.cfg.long_seq_len, 'din_long_clk')
        self.long_cart_din = WeightedSum(self.cfg, self.cfg.long_seq_len, 'din_long_cart')
        self.long_buy_din = WeightedSum(self.cfg, self.cfg.long_seq_len, 'din_long_buy')
        self.short_click_din = WeightedSum(self.cfg, self.cfg.short_seq_len, 'din_short_clk')
        self.short_cart_din = WeightedSum(self.cfg, self.cfg.short_seq_len, 'din_short_cart')
        self.short_buy_din = WeightedSum(self.cfg, self.cfg.short_seq_len, 'din_short_buy')

    def call(self, x):
        long_click, long_click_embed, \
        long_cart, long_cart_embed, \
        long_buy, long_buy_embed, \
        long_buy_level2, long_buy_level2_embed, \
        long_cart_level2, long_cart_level2_embed, \
        long_click_level2, long_click_level2_embed, \
        short_click, short_click_embed, \
        short_cart, short_cart_embed, \
        short_buy, short_buy_embed, \
        short_click_level2, short_click_level2_embed, \
        short_cart_level2, short_cart_level2_embed, \
        short_buy_level2, short_buy_level2_embed, \
        cspu_embed, lv2_embed = x

        # 1. long sequence
        sum_p_long_click_embed_masked = self.mask_long_clk(
            [long_click, long_click_embed])
        sum_p_long_cart_embed_masked = self.mask_long_cart(
            [long_cart, long_cart_embed])
        sum_p_long_buy_embed_masked = self.mask_long_buy(
            [long_buy, long_buy_embed])
        sum_p_long_click_level2_embed_masked = self.mask_long_clk_lv2(
            [long_click_level2, long_click_level2_embed])
        sum_p_long_cart_level2_embed_masked = self.mask_long_cart_lv2(
            [long_cart_level2, long_cart_level2_embed])
        sum_p_long_buy_level2_embed_masked = self.mask_long_buy_lv2(
            [long_buy_level2, long_buy_level2_embed])

        long_click_embs = tf.concat([sum_p_long_click_embed_masked, sum_p_long_click_level2_embed_masked], -1)
        long_cart_embs = tf.concat([sum_p_long_cart_embed_masked, sum_p_long_cart_level2_embed_masked], -1)
        long_buy_embs = tf.concat([sum_p_long_buy_embed_masked, sum_p_long_buy_level2_embed_masked], -1)

        # transformer layer
        long_cart_padding_mask_list = padding_mask(long_cart)
        long_buy_padding_mask_list = padding_mask(long_buy)

        long_cart_output = self.long_cart_transformer(long_cart_embs, long_cart_padding_mask_list)
        long_buy_output = self.long_buy_transformer(long_buy_embs, long_buy_padding_mask_list)

        # 2. short sequence
        sum_p_short_click_embed_masked = self.mask_short_clk(
            [short_click, short_click_embed])
        sum_p_short_cart_embed_masked = self.mask_short_cart(
            [short_cart, short_cart_embed])
        sum_p_short_buy_embed_masked = self.mask_short_buy(
            [short_buy, short_buy_embed])
        sum_p_short_click_level2_embed_masked = self.mask_short_clk_lv2(
            [short_click_level2, short_click_level2_embed])
        sum_p_short_cart_level2_embed_masked = self.mask_short_cart_lv2(
            [short_cart_level2, short_cart_level2_embed])
        sum_p_short_buy_level2_embed_masked = self.mask_short_buy_lv2(
            [short_buy_level2, short_buy_level2_embed])

        short_click_embs = tf.concat([sum_p_short_click_embed_masked, sum_p_short_click_level2_embed_masked], -1)
        short_cart_embs = tf.concat([sum_p_short_cart_embed_masked, sum_p_short_cart_level2_embed_masked], -1)
        short_buy_embs = tf.concat([sum_p_short_buy_embed_masked, sum_p_short_buy_level2_embed_masked], -1)

        short_cart_padding_mask_list = padding_mask(short_cart)
        short_buy_padding_mask_list = padding_mask(short_buy)

        short_cart_output = self.short_cart_transformer(short_cart_embs, short_cart_padding_mask_list)
        short_buy_output = self.short_buy_transformer(short_buy_embs, short_buy_padding_mask_list)

        # ATTENTION
        target_embs = tf.concat([cspu_embed, lv2_embed], -1)
        # target attention
        long_click_din = self.long_click_din([target_embs, long_click_embs])
        long_cart_din = self.long_cart_din([target_embs, long_cart_output])
        long_buy_din = self.long_buy_din([target_embs, long_buy_output])
        short_click_din = self.short_click_din([target_embs, short_click_embs])
        short_cart_din = self.short_cart_din([target_embs, short_cart_output])
        short_buy_din = self.short_buy_din([target_embs, short_buy_output])

        # for dense
        pooling_long_click_din = tf.squeeze(long_click_din, 1)
        pooling_long_cart_din = tf.squeeze(long_cart_din, 1)
        pooling_long_buy_din = tf.squeeze(long_buy_din, 1)
        pooling_short_click_din = tf.squeeze(short_click_din, 1)
        pooling_short_cart_din = tf.squeeze(short_cart_din, 1)
        pooling_short_buy_din = tf.squeeze(short_buy_din, 1)

        return pooling_long_click_din, pooling_long_cart_din, pooling_long_buy_din, \
               pooling_short_click_din, pooling_short_cart_din, pooling_short_buy_din

    def get_config(self):
        config = super(BuildSequence, self).get_config().copy()
        config.update({
            'mask_long_clk': self.mask_long_clk,
            'mask_long_cart': self.mask_long_cart,
            'mask_long_buy': self.mask_long_buy,
            'mask_long_clk_lv2': self.mask_long_clk_lv2,
            'mask_long_cart_lv2': self.mask_long_cart_lv2,
            'mask_long_buy_lv2': self.mask_long_buy_lv2,
            'mask_short_clk': self.mask_short_clk,
            'mask_short_cart': self.mask_short_cart,
            'mask_short_buy': self.mask_short_buy,
            'mask_short_clk_lv2': self.mask_short_clk_lv2,
            'mask_short_cart_lv2': self.mask_short_cart_lv2,
            'mask_short_buy_lv2': self.mask_short_buy_lv2,
            'long_cart_transformer': self.long_cart_transformer,
            'long_buy_transformer': self.long_buy_transformer,
            'short_cart_transformer': self.short_cart_transformer,
            'short_buy_transformer': self.short_buy_transformer,
            'long_click_din': self.long_click_din,
            'long_cart_din': self.long_cart_din,
            'long_buy_din': self.long_buy_din,
            'short_click_din': self.short_click_din,
            'short_cart_din': self.short_cart_din,
            'short_buy_din': self.short_buy_din
        })
        return config


class MakeEmbedding(tf.keras.Model):
    def __init__(self, cfg):
        super(MakeEmbedding, self).__init__()
        self.cfg = cfg
        self.context_embed = layers.Embedding(input_dim=sum(self.cfg.context_dims),
                                              output_dim=self.cfg.embed_dim,
                                              embeddings_initializer=self.cfg.embed_init,
                                              embeddings_regularizer=self.cfg.kernel_regular,
                                              input_length=None,
                                              name='context_embedding')
        self.realtime_back_category_embed = layers.Embedding(input_dim=sum(self.cfg.realtime_back_category_dims),
                                                             output_dim=self.cfg.embed_dim,
                                                             embeddings_initializer=self.cfg.embed_init,
                                                             embeddings_regularizer=self.cfg.kernel_regular,
                                                             input_length=None,
                                                             name='realtime_back_category_embedding')
        self.realtime_goods_embed = layers.Embedding(input_dim=sum(self.cfg.realtime_goods_dims),
                                                     output_dim=self.cfg.embed_dim,
                                                     embeddings_initializer=self.cfg.embed_init,
                                                     embeddings_regularizer=self.cfg.kernel_regular,
                                                     input_length=None,
                                                     name='realtime_goods_embedding')
        self.realtime_pair_click_embed = layers.Embedding(input_dim=3,
                                                          output_dim=self.cfg.embed_dim,
                                                          embeddings_initializer=self.cfg.embed_init,
                                                          embeddings_regularizer=self.cfg.kernel_regular,
                                                          input_length=None,
                                                          name='realtime_pair_click_embedding')
        self.realtime_passtime_embed = layers.Embedding(input_dim=sum(self.cfg.realtime_passtime_dims),
                                                        output_dim=self.cfg.embed_dim,
                                                        embeddings_initializer=self.cfg.embed_init,
                                                        embeddings_regularizer=self.cfg.kernel_regular,
                                                        input_length=None,
                                                        name='realtime_passtime_embedding')
        self.realtime_user_group_embed = layers.Embedding(input_dim=sum(self.cfg.realtime_user_group_dims),
                                                          output_dim=self.cfg.embed_dim,
                                                          embeddings_initializer=self.cfg.embed_init,
                                                          embeddings_regularizer=self.cfg.kernel_regular,
                                                          input_length=None,
                                                          name='realtime_user_group_embedding')
        self.goods_sparse_embed = layers.Embedding(input_dim=sum(self.cfg.goods_field_dims),
                                                   output_dim=self.cfg.embed_dim,
                                                   embeddings_initializer=self.cfg.embed_init,
                                                   embeddings_regularizer=self.cfg.kernel_regular,
                                                   input_length=None,
                                                   name='goods_sparse_embedding')
        self.bucket_user_box_obj_embed = layers.Embedding(input_dim=sum(self.cfg.bucket_user_dims),
                                                          output_dim=self.cfg.embed_dim,
                                                          embeddings_initializer=self.cfg.embed_init,
                                                          embeddings_regularizer=self.cfg.kernel_regular,
                                                          input_length=None,
                                                          name='bucket_user_embedding')
        self.bucket_goods_box_obj_embed = layers.Embedding(input_dim=sum(self.cfg.bucket_goods_dims),
                                                           output_dim=self.cfg.embed_dim,
                                                           embeddings_initializer=self.cfg.embed_init,
                                                           embeddings_regularizer=self.cfg.kernel_regular,
                                                           input_length=None,
                                                           name='bucket_goods_embedding')
        self.bucket_goods_gross_obj_embed = layers.Embedding(input_dim=sum(self.cfg.bucket_goods_gross_dims),
                                                             output_dim=self.cfg.embed_dim,
                                                             embeddings_initializer=self.cfg.embed_init,
                                                             embeddings_regularizer=self.cfg.kernel_regular,
                                                             input_length=None,
                                                             name='bucket_goods_gross_embedding')
        self.pair_feature_embed = layers.Embedding(input_dim=sum(self.cfg.pair_field_dims),
                                                   output_dim=self.cfg.embed_dim,
                                                   embeddings_initializer=self.cfg.embed_init,
                                                   embeddings_regularizer=self.cfg.kernel_regular,
                                                   input_length=None,
                                                   name='pair_feature_embedding')
        self.bucket_pair_box_obj_embed = layers.Embedding(input_dim=sum(self.cfg.bucket_pair_dims),
                                                          output_dim=self.cfg.embed_dim,
                                                          embeddings_initializer=self.cfg.embed_init,
                                                          embeddings_regularizer=self.cfg.kernel_regular,
                                                          input_length=None,
                                                          name='bucket_pair_embedding')
        self.bucket_user_cspu_obj_embed = layers.Embedding(input_dim=sum(self.cfg.bucket_user_cspu_dims),
                                                           output_dim=self.cfg.embed_dim,
                                                           embeddings_initializer=self.cfg.embed_init,
                                                           embeddings_regularizer=self.cfg.kernel_regular,
                                                           input_length=None,
                                                           name='bucket_user_cspu_embedding')
        self.bucket_ozid_cspu_obj_embed = layers.Embedding(input_dim=sum(self.cfg.bucket_ozid_cspu_dims),
                                                           output_dim=self.cfg.embed_dim,
                                                           embeddings_initializer=self.cfg.embed_init,
                                                           embeddings_regularizer=self.cfg.kernel_regular,
                                                           input_length=None,
                                                           name='bucket_ozid_cspu_embedding')
        self.bucket_user_behavior_obj_embed = layers.Embedding(input_dim=sum(self.cfg.bucket_user_behavior_dims),
                                                               output_dim=self.cfg.embed_dim,
                                                               embeddings_initializer=self.cfg.embed_init,
                                                               embeddings_regularizer=self.cfg.kernel_regular,
                                                               input_length=None,
                                                               name='bucket_user_behavior_embedding')
        self.cspu_embed_lookup = layers.Embedding(input_dim=self.cfg.dims_dict['cspu_idx'],
                                                  output_dim=self.cfg.embed_dim,
                                                  embeddings_initializer=self.cfg.embed_init,
                                                  embeddings_regularizer=self.cfg.kernel_regular,
                                                  input_length=None,
                                                  name='cspu_embedding')
        self.supplier_embed_lookup = layers.Embedding(input_dim=self.cfg.dims_dict['supplier_idx'],
                                                      output_dim=self.cfg.embed_dim,
                                                      embeddings_initializer=self.cfg.embed_init,
                                                      embeddings_regularizer=self.cfg.kernel_regular,
                                                      input_length=None,
                                                      name='supplier_embedding')
        self.level2_embed_lookup = layers.Embedding(input_dim=self.cfg.dims_dict['level2_idx'],
                                                    output_dim=self.cfg.embed_dim,
                                                    embeddings_initializer=self.cfg.embed_init,
                                                    embeddings_regularizer=self.cfg.kernel_regular,
                                                    input_length=None,
                                                    name='level2_embedding')

    def call(self, x):
        context_features, \
        realtime_back_category, realtime_goods, realtime_pair_click, realtime_passtime, realtime_user_group, \
        goods_sparse, \
        bucket_user_box_obj, bucket_goods_box_obj, bucket_goods_gross_obj, \
        pair_feature, \
        bucket_pair_box_obj, bucket_user_cspu_obj, bucket_ozid_cspu_obj, bucket_user_behavior_obj, \
        cspu_idx, supplier_idx, lv2_idx, \
        long_click, long_cart, long_buy, \
        long_buy_level2, long_cart_level2, long_click_level2, \
        short_click, short_cart, short_buy, \
        short_click_level2, short_cart_level2, short_buy_level2 = x

        context_embed = self.context_embed(context_features)
        realtime_back_category_embed = self.realtime_back_category_embed(realtime_back_category)
        realtime_goods_embed = self.realtime_goods_embed(realtime_goods)
        realtime_pair_click_embed = self.realtime_pair_click_embed(realtime_pair_click)
        realtime_pair_click_embed = tf.expand_dims(realtime_pair_click_embed, 1)
        realtime_passtime_embed = self.realtime_passtime_embed(realtime_passtime)
        realtime_user_group_embed = self.realtime_user_group_embed(realtime_user_group)
        goods_sparse_embed = self.goods_sparse_embed(goods_sparse)
        bucket_user_box_obj_embed = self.bucket_user_box_obj_embed(bucket_user_box_obj)
        bucket_goods_box_obj_embed = self.bucket_goods_box_obj_embed(bucket_goods_box_obj)
        bucket_goods_gross_obj_embed = self.bucket_goods_gross_obj_embed(bucket_goods_gross_obj)
        pair_feature_embed = self.pair_feature_embed(pair_feature)
        bucket_pair_box_obj_embed = self.bucket_pair_box_obj_embed(bucket_pair_box_obj)
        bucket_user_cspu_obj_embed = self.bucket_user_cspu_obj_embed(bucket_user_cspu_obj)
        bucket_ozid_cspu_obj_embed = self.bucket_ozid_cspu_obj_embed(bucket_ozid_cspu_obj)
        bucket_user_behavior_obj_embed = self.bucket_user_behavior_obj_embed(bucket_user_behavior_obj)

        cspu_embed = self.cspu_embed_lookup(cspu_idx)
        supplier_embed = self.supplier_embed_lookup(supplier_idx)
        lv2_embed = self.level2_embed_lookup(lv2_idx)

        long_click_embed = self.cspu_embed_lookup(long_click)
        long_cart_embed = self.cspu_embed_lookup(long_cart)
        long_buy_embed = self.cspu_embed_lookup(long_buy)

        long_buy_level2_embed = self.level2_embed_lookup(long_buy_level2)
        long_cart_level2_embed = self.level2_embed_lookup(long_cart_level2)
        long_click_level2_embed = self.level2_embed_lookup(long_click_level2)

        short_click_embed = self.cspu_embed_lookup(short_click)
        short_cart_embed = self.cspu_embed_lookup(short_cart)
        short_buy_embed = self.cspu_embed_lookup(short_buy)

        short_click_level2_embed = self.level2_embed_lookup(short_click_level2)
        short_cart_level2_embed = self.level2_embed_lookup(short_cart_level2)
        short_buy_level2_embed = self.level2_embed_lookup(short_buy_level2)

        return context_embed, \
               realtime_back_category_embed, realtime_goods_embed, realtime_pair_click_embed, \
               realtime_passtime_embed, realtime_user_group_embed, \
               goods_sparse_embed, \
               bucket_user_box_obj_embed, bucket_goods_box_obj_embed, bucket_goods_gross_obj_embed, \
               pair_feature_embed, bucket_pair_box_obj_embed, \
               bucket_user_cspu_obj_embed, bucket_ozid_cspu_obj_embed, bucket_user_behavior_obj_embed, \
               cspu_embed, supplier_embed, lv2_embed, \
               long_click_embed, long_cart_embed, long_buy_embed, \
               long_buy_level2_embed, long_cart_level2_embed, long_click_level2_embed, \
               short_click_embed, short_cart_embed, short_buy_embed, \
               short_click_level2_embed, short_cart_level2_embed, short_buy_level2_embed

    def get_config(self):
        config = super(MakeEmbedding, self).get_config().copy()
        config.update({
            'context_embed': self.context_embed,
            'realtime_back_category_embed': self.realtime_back_category_embed,
            'realtime_goods_embed': self.realtime_goods_embed,
            'realtime_pair_click_embed': self.realtime_pair_click_embed,
            'realtime_passtime_embed': self.realtime_passtime_embed,
            'realtime_user_group_embed': self.realtime_user_group_embed,
            'goods_sparse_embed': self.goods_sparse_embed,
            'bucket_user_box_obj_embed': self.bucket_user_box_obj_embed,
            'bucket_goods_box_obj_embed': self.bucket_goods_box_obj_embed,
            'bucket_goods_gross_obj_embed': self.bucket_goods_gross_obj_embed,
            'pair_feature_embed': self.pair_feature_embed,
            'bucket_pair_box_obj_embed': self.bucket_pair_box_obj_embed,
            'bucket_user_cspu_obj_embed': self.bucket_user_cspu_obj_embed,
            'bucket_ozid_cspu_obj_embed': self.bucket_ozid_cspu_obj_embed,
            'bucket_user_behavior_obj_embed': self.bucket_user_behavior_obj_embed,
            'cspu_embed_lookup': self.cspu_embed_lookup,
            'supplier_embed_lookup': self.supplier_embed_lookup,
            'level2_embed_lookup': self.level2_embed_lookup
        })
        return config


class ProcessPairs(tf.keras.Model):
    def __init__(self, cfg):
        super(ProcessPairs, self).__init__()
        self.cfg = cfg

    def call(self, x):
        pair_feature, bucket_pair_features, bucket_user_cspu_obj, bucket_ozid_cspu_obj, bucket_user_behavior_obj = x
        pair_feature = tf.cast(pair_feature, tf.int32)
        pair_feature = tf.clip_by_value(pair_feature,
                                        clip_value_min=0,
                                        clip_value_max=sum(self.cfg.pair_field_dims) - 1,
                                        name='pair_feature')

        bucket_pair_box_obj = tf.cast(tf.concat([bucket_pair_features[:, :10], bucket_pair_features[:, 20:]], axis=-1),
                                      tf.int32)  # 离散值
        bucket_pair_offsets = tf.expand_dims(tf.cumsum([0] + self.cfg.bucket_pair_dims[:-1], axis=0), axis=0)  # [1, n]
        bucket_pair_box_obj = bucket_pair_box_obj + bucket_pair_offsets
        bucket_pair_box_obj = tf.clip_by_value(bucket_pair_box_obj,
                                               clip_value_min=0,
                                               clip_value_max=sum(self.cfg.bucket_pair_dims) - 1,
                                               name='bucket_pair')

        bucket_user_cspu_offsets = tf.expand_dims(tf.cumsum([0] + self.cfg.bucket_user_cspu_dims[:-1], axis=0), axis=0)  # [1, n]
        bucket_user_cspu_obj = tf.cast(bucket_user_cspu_obj, tf.int32)
        bucket_user_cspu_obj = bucket_user_cspu_obj + bucket_user_cspu_offsets
        bucket_user_cspu_obj = tf.clip_by_value(bucket_user_cspu_obj,
                                                clip_value_min=0,
                                                clip_value_max=sum(self.cfg.bucket_user_cspu_dims) - 1,
                                                name='bucket_user_cspu')

        bucket_ozid_cspu_obj = tf.cast(bucket_ozid_cspu_obj, tf.int32)
        bucket_ozid_cspu_offsets = tf.expand_dims(tf.cumsum([0] + self.cfg.bucket_ozid_cspu_dims[:-1], axis=0),
                                                  axis=0)  # [1, n]
        bucket_ozid_cspu_obj = bucket_ozid_cspu_obj + bucket_ozid_cspu_offsets
        bucket_ozid_cspu_obj = tf.clip_by_value(bucket_ozid_cspu_obj,
                                                clip_value_min=0,
                                                clip_value_max=sum(self.cfg.bucket_ozid_cspu_dims) - 1,
                                                name='bucket_ozid_cspu')

        bucket_user_behavior_obj = tf.cast(bucket_user_behavior_obj, tf.int32)
        bucket_user_behavior_offsets = tf.expand_dims(tf.cumsum([0] + self.cfg.bucket_user_behavior_dims[:-1], axis=0),
                                                      axis=0)  # [1, n]
        bucket_user_behavior_obj = bucket_user_behavior_obj + bucket_user_behavior_offsets
        bucket_user_behavior_obj = tf.clip_by_value(bucket_user_behavior_obj,
                                                    clip_value_min=0,
                                                    clip_value_max=sum(self.cfg.bucket_user_behavior_dims) - 1,
                                                    name='bucket_user_behavior')
        return pair_feature, bucket_pair_box_obj, bucket_user_cspu_obj, bucket_ozid_cspu_obj, bucket_user_behavior_obj


class ProcessGoods(tf.keras.Model):
    def __init__(self, cfg):
        super(ProcessGoods, self).__init__()
        self.cfg = cfg

    def call(self, x):
        bucket_goods_features, bucket_goods_gross_obj = x
        bucket_goods_box_obj = tf.cast(bucket_goods_features[:, 4:53], tf.int32)  # 离散值,去除后端类目，goods_sparse_features已经有了
        bucket_goods_raw = tf.cast(bucket_goods_features[:, 98:], tf.float32)  # 连续值
        bucket_goods_raw = tf.clip_by_value(bucket_goods_raw, clip_value_min=0, clip_value_max=1, name='bucket_goods_raw')
        bucket_goods_offsets = tf.expand_dims(tf.cumsum([0] + self.cfg.bucket_goods_dims[:-1], axis=0),
                                              axis=0,
                                              name='bucket_goods_offsets')  # [1, n]
        bucket_goods_box_obj = bucket_goods_box_obj + bucket_goods_offsets
        bucket_goods_box_obj = tf.clip_by_value(bucket_goods_box_obj,
                                                clip_value_min=0,
                                                clip_value_max=sum(self.cfg.bucket_goods_dims) - 1,
                                                name='bucket_goods')

        bucket_goods_gross_obj = tf.cast(bucket_goods_gross_obj, tf.int32)
        bucket_goods_gross_offsets = tf.expand_dims(tf.cumsum([0] + self.cfg.bucket_goods_gross_dims[:-1], axis=0),
                                                    axis=0)  # [1, n]
        bucket_goods_gross_obj = bucket_goods_gross_obj + bucket_goods_gross_offsets
        bucket_goods_gross_obj = tf.clip_by_value(bucket_goods_gross_obj,
                                                  clip_value_min=0,
                                                  clip_value_max=sum(self.cfg.bucket_goods_gross_dims) - 1,
                                                  name='bucket_goods_gross')
        return bucket_goods_raw, bucket_goods_box_obj, bucket_goods_gross_obj


class ProcessUser(tf.keras.Model):
    def __init__(self, cfg):
        super(ProcessUser, self).__init__()
        self.cfg = cfg

    def call(self, x):
        bucket_user_box_obj = tf.cast(x[:, :66], tf.int32)  # 离散值
        bucket_user_raw = tf.cast(x[:, 120:], tf.float32)  # 连续值
        bucket_user_raw = tf.clip_by_value(bucket_user_raw, clip_value_min=0, clip_value_max=1, name='bucket_user_raw')
        bucket_user_offsets = tf.expand_dims(tf.cumsum([0] + self.cfg.bucket_user_dims[:-1], axis=0), axis=0)  # [1, n]
        bucket_user_box_obj = bucket_user_box_obj + bucket_user_offsets
        bucket_user_box_obj = tf.clip_by_value(bucket_user_box_obj,
                                               clip_value_min=0,
                                               clip_value_max=sum(self.cfg.bucket_user_dims) - 1,
                                               name='bucket_user')
        return bucket_user_raw, bucket_user_box_obj


class ProcessSequence(tf.keras.Model):
    def __init__(self, cfg):
        super(ProcessSequence, self).__init__()
        self.cfg = cfg

    def call(self, x):
        long_click = x[:, :self.cfg.sequence_stride[0]]
        long_cart = x[:, self.cfg.sequence_stride[0]: self.cfg.sequence_stride[1]]
        long_buy = x[:, self.cfg.sequence_stride[1]: self.cfg.sequence_stride[2]]
        # long_buy_supplier = sequence[:, cfg.sequence_stride[2]:cfg.sequence_stride[3]]
        long_buy_level2 = x[:, self.cfg.sequence_stride[3]: self.cfg.sequence_stride[4]]
        long_cart_level2 = x[:, self.cfg.sequence_stride[4]: self.cfg.sequence_stride[5]]
        long_click_level2 = x[:, self.cfg.sequence_stride[5]: self.cfg.sequence_stride[6]]

        short_click = x[:, self.cfg.sequence_stride[6]: self.cfg.sequence_stride[7]]
        short_cart = x[:, self.cfg.sequence_stride[7]: self.cfg.sequence_stride[8]]
        short_buy = x[:, self.cfg.sequence_stride[8]: self.cfg.sequence_stride[9]]
        # short_click_supplier = sequence[:, cfg.sequence_stride[9]:cfg.sequence_stride[10]]
        short_click_level2 = x[:, self.cfg.sequence_stride[10]: self.cfg.sequence_stride[11]]
        short_cart_level2 = x[:, self.cfg.sequence_stride[11]: self.cfg.sequence_stride[12]]
        short_buy_level2 = x[:, self.cfg.sequence_stride[12]:]

        long_click = tf.clip_by_value(long_click,
                                      clip_value_min=0,
                                      clip_value_max=self.cfg.dims_dict['cspu_idx'] - 1,
                                      name='long_click')
        long_cart = tf.clip_by_value(long_cart,
                                     clip_value_min=0,
                                     clip_value_max=self.cfg.dims_dict['cspu_idx'] - 1,
                                     name='long_cart')
        long_buy = tf.clip_by_value(long_buy,
                                    clip_value_min=0,
                                    clip_value_max=self.cfg.dims_dict['cspu_idx'] - 1,
                                    name='long_buy')

        long_buy_level2 = tf.clip_by_value(long_buy_level2,
                                           clip_value_min=0,
                                           clip_value_max=self.cfg.dims_dict['level2_idx'] - 1,
                                           name='long_buy_level2')
        long_cart_level2 = tf.clip_by_value(long_cart_level2,
                                            clip_value_min=0,
                                            clip_value_max=self.cfg.dims_dict['level2_idx'] - 1,
                                            name='long_cart_level2')
        long_click_level2 = tf.clip_by_value(long_click_level2,
                                             clip_value_min=0,
                                             clip_value_max=self.cfg.dims_dict['level2_idx'] - 1,
                                             name='long_click_level2')

        short_click = tf.clip_by_value(short_click,
                                       clip_value_min=0,
                                       clip_value_max=self.cfg.dims_dict['cspu_idx'] - 1,
                                       name='short_click')
        short_cart = tf.clip_by_value(short_cart,
                                      clip_value_min=0,
                                      clip_value_max=self.cfg.dims_dict['cspu_idx'] - 1,
                                      name='short_cart')
        short_buy = tf.clip_by_value(short_buy,
                                     clip_value_min=0,
                                     clip_value_max=self.cfg.dims_dict['cspu_idx'] - 1,
                                     name='short_buy')

        short_click_level2 = tf.clip_by_value(short_click_level2,
                                              clip_value_min=0,
                                              clip_value_max=self.cfg.dims_dict['level2_idx'] - 1,
                                              name='short_click_level2')
        short_cart_level2 = tf.clip_by_value(short_cart_level2,
                                             clip_value_min=0,
                                             clip_value_max=self.cfg.dims_dict['level2_idx'] - 1,
                                             name='short_cart_level2')
        short_buy_level2 = tf.clip_by_value(short_buy_level2,
                                            clip_value_min=0,
                                            clip_value_max=self.cfg.dims_dict['level2_idx'] - 1,
                                            name='short_buy_level2')

        return (long_click, long_cart, long_buy,
                long_buy_level2, long_cart_level2, long_click_level2), \
               (short_click, short_cart, short_buy,
                short_click_level2, short_cart_level2, short_buy_level2)


class ProcessSparseID(tf.keras.Model):
    def __init__(self, cfg):
        super(ProcessSparseID, self).__init__()
        self.cfg = cfg

    def call(self, x):
        lv1_idx = tf.reshape(
            tf.cast(x[:, self.cfg.goods_sparse_id_fmap['back_lv1']], tf.int32), shape=(-1, 1))
        lv2_idx = tf.reshape(
            tf.cast(x[:, self.cfg.goods_sparse_id_fmap['back_lv2']], tf.int32), shape=(-1, 1))
        lv3_idx = tf.reshape(
            tf.cast(x[:, self.cfg.goods_sparse_id_fmap['back_lv3']], tf.int32), shape=(-1, 1))
        lv4_idx = tf.reshape(
            tf.cast(x[:, self.cfg.goods_sparse_id_fmap['back_lv4']], tf.int32), shape=(-1, 1))
        cspu_idx = tf.reshape(tf.cast(x[:, self.cfg.goods_sparse_id_fmap['cspu_id']], tf.int32),
                              shape=(-1, 1))
        supplier_idx = tf.reshape(tf.cast(x[:, self.cfg.goods_sparse_id_fmap['supplier_id']], tf.int32),
                                  shape=(-1, 1))
        goods_unit_idx = tf.reshape(tf.cast(x[:, self.cfg.goods_sparse_id_fmap['goods_unit']], tf.int32),
                                    shape=(-1, 1))
        origin_idx = tf.reshape(tf.cast(x[:, self.cfg.goods_sparse_id_fmap['origin']], tf.int32),
                                shape=(-1, 1))
        selling_points_idx = tf.reshape(tf.cast(x[:, self.cfg.goods_sparse_id_fmap['selling_points']], tf.int32),
                                        shape=(-1, 1))
        brand_idx = tf.reshape(tf.cast(x[:, self.cfg.goods_sparse_id_fmap['brand']], tf.int32),
                               shape=(-1, 1))
        entity_idx = tf.reshape(tf.cast(x[:, self.cfg.goods_sparse_id_fmap['goods_entity']], tf.int32),
                                shape=(-1, 1))

        lv1_idx = tf.clip_by_value(lv1_idx, clip_value_min=0, clip_value_max=self.cfg.dims_dict['level1_idx'] - 1, name='lv1_idx')
        lv2_idx = tf.clip_by_value(lv2_idx, clip_value_min=0, clip_value_max=self.cfg.dims_dict['level2_idx'] - 1, name='lv2_idx')
        lv3_idx = tf.clip_by_value(lv3_idx, clip_value_min=0, clip_value_max=self.cfg.dims_dict['level3_idx'] - 1, name='lv3_idx')
        lv4_idx = tf.clip_by_value(lv4_idx, clip_value_min=0, clip_value_max=self.cfg.dims_dict['level4_idx'] - 1, name='lv4_idx')
        cspu_idx = tf.clip_by_value(cspu_idx, clip_value_min=0, clip_value_max=self.cfg.dims_dict['cspu_idx'] - 1, name='cspu_idx')
        supplier_idx = tf.clip_by_value(supplier_idx,
                                        clip_value_min=0,
                                        clip_value_max=self.cfg.dims_dict['supplier_idx'] - 1,
                                        name='supplier_idx')
        goods_unit_idx = tf.clip_by_value(goods_unit_idx,
                                          clip_value_min=0,
                                          clip_value_max=self.cfg.dims_dict['goods_unit_idx'] - 1,
                                          name='goods_unit_idx')
        origin_idx = tf.clip_by_value(origin_idx,
                                      clip_value_min=0,
                                      clip_value_max=self.cfg.dims_dict['origin_idx'] - 1,
                                      name='origin_idx')
        selling_points_idx = tf.clip_by_value(selling_points_idx,
                                              clip_value_min=0,
                                              clip_value_max=self.cfg.dims_dict['selling_points_idx'] - 1,
                                              name='selling_points_idx')
        brand_idx = tf.clip_by_value(brand_idx,
                                     clip_value_min=0,
                                     clip_value_max=self.cfg.dims_dict['brand_idx'] - 1,
                                     name='brand_idx')
        entity_idx = tf.clip_by_value(entity_idx,
                                      clip_value_min=0,
                                      clip_value_max=self.cfg.dims_dict['entity_idx'] - 1,
                                      name='entity_idx')

        ## concat 商品离散idx
        goods_sparse = tf.concat([lv1_idx, lv3_idx, lv4_idx, goods_unit_idx, origin_idx, selling_points_idx, brand_idx, entity_idx],
                                 axis=1, name="goods_sparse_idx_concat")
        goods_offsets = tf.expand_dims(tf.cumsum([0] + self.cfg.goods_field_dims[:-1], axis=0), axis=0, name='goods_offsets')  ## [1, n]
        goods_sparse = goods_sparse + goods_offsets
        goods_sparse = tf.clip_by_value(goods_sparse,
                                        clip_value_min=0,
                                        clip_value_max=sum(self.cfg.goods_field_dims) - 1,
                                        name='goods_sparse')
        cspu_idx = tf.clip_by_value(cspu_idx,
                                    clip_value_min=0,
                                    clip_value_max=self.cfg.dims_dict['cspu_idx'] - 1,
                                    name='cspu_idx')
        supplier_idx = tf.clip_by_value(supplier_idx,
                                        clip_value_min=0,
                                        clip_value_max=self.cfg.dims_dict['supplier_idx'] - 1,
                                        name='supplier_idx')
        lv2_idx = tf.clip_by_value(lv2_idx,
                                   clip_value_min=0,
                                   clip_value_max=self.cfg.dims_dict['level2_idx'] - 1,
                                   name='lv2_idx')

        return goods_sparse, cspu_idx, supplier_idx, lv2_idx


class ProcessContext(tf.keras.Model):
    def __init__(self, cfg):
        super(ProcessContext, self).__init__()
        self.cfg = cfg

    def call(self, x):
        realtime_features, context_features = x
        # realtime_user = realtime_features[:, :5]
        realtime_pair_click = realtime_features[:, 5]
        # realtime_cross_category_front = features[:,6:6+24]  ## 前端类目pair不要了
        realtime_back_category = realtime_features[:, 30:30 + 4 * 12]  # 分桶后传入fm
        realtime_goods = realtime_features[:, 30 + 4 * 12:30 + 4 * 12 + 15]
        realtime_passtime = realtime_features[:, 30 + 4 * 12 + 15:30 + 4 * 12 + 15 + 21]
        realtime_user_group = realtime_features[:, 30 + 4 * 12 + 15 + 21:30 + 4 * 12 + 15 + 21 + 36]

        # 实时离散特征处理
        # ------------ 实时特征等距分箱处理 ---------
        #   商品特征进行log分桶，用户和pair都进行等距分桶，间隔3为一个桶，最大10个桶。
        # -----------------------------------------
        # realtime_user = tf.clip_by_value(realtime_user, clip_value_min=0, clip_value_max=29)
        realtime_pair_click = tf.cast(tf.clip_by_value(realtime_pair_click, clip_value_min=0, clip_value_max=2),
                                      tf.int32,
                                      name='realtime_pair_click')
        # realtime_cross_category_front = tf.clip_by_value(realtime_cross_category_front, clip_value_min=0, clip_value_max=29)
        realtime_back_category = tf.clip_by_value(realtime_back_category, clip_value_min=0, clip_value_max=29)
        realtime_goods = tf.clip_by_value(realtime_goods, clip_value_min=0, clip_value_max=100000000)
        realtime_passtime = tf.clip_by_value(realtime_passtime, clip_value_min=0, clip_value_max=129600)
        realtime_user_group = tf.clip_by_value(realtime_user_group, clip_value_min=0, clip_value_max=129600)

        # realtime_user = tf.cast(tf.math.floordiv(realtime_user, 5), tf.int32)
        # realtime_cross_category_front = tf.cast(tf.floordiv(realtime_cross_category_front, 5), tf.int32)
        realtime_back_category = tf.cast(tf.math.floordiv(realtime_back_category, 5),
                                         tf.int32,
                                         name='realtime_back_category')
        realtime_goods = tf.cast(tf.math.log(realtime_goods + 1),
                                 tf.int32)
        realtime_goods = tf.clip_by_value(realtime_goods,
                                          clip_value_min=0,
                                          clip_value_max=30,
                                          name='realtime_goods')
        realtime_passtime_hour = tf.math.floordiv(realtime_passtime, 60, name="pst_hour")
        realtime_passtime_day = tf.add(tf.math.floordiv(realtime_passtime, 1440), 23, name="pst_day")
        realtime_passtime = tf.cast(tf.where(realtime_passtime < 1440.0, realtime_passtime_hour, realtime_passtime_day),
                                    tf.int32,
                                    name='realtime_passtime')
        realtime_user_group = tf.cast(tf.math.log(realtime_user_group + 1), tf.int32)
        realtime_user_group = tf.clip_by_value(realtime_user_group,
                                               clip_value_min=0,
                                               clip_value_max=30,
                                               name='realtime_user_group')

        # ------------ 实时特征等距分箱处理 done ---------
        realtime_back_category_offsets = tf.expand_dims(tf.cumsum([0] + self.cfg.realtime_back_category_dims[:-1], axis=0),
                                                        axis=0,
                                                        name='realtime_back_category_offsets')  ## [1, n]
        realtime_back_category = realtime_back_category + realtime_back_category_offsets
        realtime_goods_offsets = tf.expand_dims(tf.cumsum([0] + self.cfg.realtime_goods_dims[:-1], axis=0),
                                                axis=0,
                                                name='realtime_goods_offsets')
        realtime_goods = realtime_goods + realtime_goods_offsets
        realtime_passtime_offsets = tf.expand_dims(tf.cumsum([0] + self.cfg.realtime_passtime_dims[:-1], axis=0),
                                                   axis=0,
                                                   name='realtime_passtime_offsets')
        realtime_passtime = realtime_passtime + realtime_passtime_offsets
        realtime_user_group_offsets = tf.expand_dims(tf.cumsum([0] + self.cfg.realtime_user_group_dims[:-1], axis=0),
                                                     axis=0,
                                                     name='realtime_user_group_offsets')
        realtime_user_group = realtime_user_group + realtime_user_group_offsets

        context_features = context_features[:, 0:2]  # 只取星期和小时；
        context_offsets = tf.expand_dims(tf.cumsum([0] + self.cfg.context_dims[:-1], axis=0),
                                         axis=0,
                                         name='context_offsets')  # [1, n]
        context_features = context_features + context_offsets
        return (realtime_pair_click, realtime_back_category, realtime_goods, realtime_passtime, realtime_user_group), context_features


class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, num_heads, middle_units,
                 max_seq_len, epsilon=1e-6, dropout_rate=0.1, training=False, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.n_layers = n_layers
        self.d_model = d_model
        # self.pos_embedding = PositionalEncoding(sequence_len=max_seq_len, embedding_dim=d_model)

        self.encode_layer = [EncoderLayer(d_model=d_model, num_heads=num_heads,
                                          middle_units=middle_units,
                                          epsilon=epsilon, dropout_rate=dropout_rate,
                                          training=training)
                             for _ in range(n_layers)]

    def call(self, emb, mask=None):
        # emb = self.pos_embedding(emb)
        for i in range(self.n_layers):
            emb = self.encode_layer[i](emb, mask)

        return emb

    def get_config(self):
        config = super(Encoder, self).get_config().copy()
        config.update({
            'n_layers': self.n_layers,
            'd_model': self.d_model,
            'encode_layer': self.encode_layer
        })
        return config


# 编码层
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, middle_units,
                 epsilon=1e-6, dropout_rate=0.1, training=False, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, middle_units)

        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.training = training

    def call(self, inputs, mask, **kwargs):
        # 多头注意力网络
        att_output = self.mha([inputs, inputs, inputs, mask])
        att_output = self.dropout1(att_output, training=self.training)
        out1 = self.layernorm1(inputs + att_output)  # (batch_size, input_seq_len, d_model)

        # 前向网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=self.training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

    def get_config(self):
        config = super(EncoderLayer, self).get_config().copy()
        config.update({
            'mha': self.mha,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
            'training': self.training
        })
        return config


# 层标准化
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        self.eps = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(LayerNormalization, self).get_config().copy()
        config.update({
            'eps': self.eps
        })
        return config


# 前向网络
def point_wise_feed_forward_network(d_model, middle_units):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(middle_units, activation='relu'),
        tf.keras.layers.Dense(d_model, activation='relu')])


# dot attention
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dim_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dim_k)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output


# 构造 multi head attention 层
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        # d_model 必须可以正确分为各个头
        assert d_model % num_heads == 0

        # 分头后的维度
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

        self.dot_attention = scaled_dot_product_attention

    def split_heads(self, x):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (-1, x.shape[1], self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, **kwargs):
        q, k, v, mask = inputs

        # 分头前的前向网络，获取q、k、v语义
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        # 分头
        q = self.split_heads(q)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v)  # (batch_size, num_heads, seq_len_v, depth)

        # 通过缩放点积注意力层
        scaled_attention = self.dot_attention(q, k, v, mask)  # (batch_size, num_heads, seq_len_q, depth)

        # “多头维度” 后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        # 合并 “多头维度”
        concat_attention = tf.reshape(scaled_attention,
                                      (-1, scaled_attention.shape[1], self.d_model))

        # 全连接层
        output = self.dense(concat_attention)

        return output

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'depth': self.depth,
            'wq': self.wq,
            'wk': self.wk,
            'wv': self.wv,
            'dense': self.dense,
            'dot_attention': self.dot_attention
        })
        return config


def padding_mask(seq):
    # 获取为 0的padding项
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 扩充维度用于attention矩阵
    return seq[:, np.newaxis, np.newaxis, :]  # (batch_size, 1, 1, seq_len)


# 位置编码
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_len=None, embedding_dim=None, **kwargs):
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim
        super(PositionalEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        if self.embedding_dim == None:
            self.embedding_dim = int(inputs.shape[-1])

        position_embedding = np.array([
            [pos / np.power(10000, 2. * i / self.embedding_dim) for i in range(self.embedding_dim)]
            for pos in range(self.sequence_len)])

        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])  # dim 2i
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])  # dim 2i+1

        position_embedding = tf.cast(position_embedding, dtype=tf.float32)

        return position_embedding + inputs

    def compute_output_shape(self, input_shape):
        return input_shape


class BuildLinear(tf.keras.Model):
    def __init__(self, cfg):
        super(BuildLinear, self).__init__()
        self.cfg = cfg
        self.linear_raw = layers.Dense(1,
                                       activation=None,
                                       kernel_regularizer=self.cfg.kernel_regular,
                                       use_bias=True,
                                       name='linear_raw')
        self.linear_pair = layers.Embedding(input_dim=sum(self.cfg.pair_field_dims),
                                            output_dim=1,
                                            embeddings_initializer=self.cfg.embed_init,
                                            embeddings_regularizer=self.cfg.kernel_regular,
                                            input_length=None,
                                            name='linear_pair')
        self.linear_realtime_back_category = layers.Embedding(input_dim=sum(self.cfg.realtime_back_category_dims),
                                                              output_dim=1,
                                                              embeddings_initializer=self.cfg.embed_init,
                                                              embeddings_regularizer=self.cfg.kernel_regular,
                                                              input_length=None,
                                                              name='linear_realtime_back_category')
        self.linear_realtime_goods = layers.Embedding(input_dim=sum(self.cfg.realtime_goods_dims),
                                                      output_dim=1,
                                                      embeddings_initializer=self.cfg.embed_init,
                                                      embeddings_regularizer=self.cfg.kernel_regular,
                                                      input_length=None,
                                                      name='linear_realtime_goods')
        self.linear_realtime_pair_click = layers.Embedding(input_dim=3,
                                                           output_dim=1,
                                                           embeddings_initializer=self.cfg.embed_init,
                                                           embeddings_regularizer=self.cfg.kernel_regular,
                                                           input_length=None,
                                                           name='linear_realtime_pair_click')
        self.linear_pair_box_obj = layers.Embedding(input_dim=sum(self.cfg.bucket_pair_dims),
                                                    output_dim=1,
                                                    embeddings_initializer=self.cfg.embed_init,
                                                    embeddings_regularizer=self.cfg.kernel_regular,
                                                    input_length=None,
                                                    name='linear_pair_box_obj')  ## [1, 62, embed_dim]
        self.linear_goods_box_obj = layers.Embedding(input_dim=sum(self.cfg.bucket_goods_dims),
                                                     output_dim=1,
                                                     embeddings_initializer=self.cfg.embed_init,
                                                     embeddings_regularizer=self.cfg.kernel_regular,
                                                     input_length=None,
                                                     name='linear_goods_box_obj')  ## [1, 62, embed_dim]
        self.linear_realtime_passtime = layers.Embedding(input_dim=sum(self.cfg.realtime_passtime_dims),
                                                         output_dim=1,
                                                         embeddings_initializer=self.cfg.embed_init,
                                                         embeddings_regularizer=self.cfg.kernel_regular,
                                                         input_length=None,
                                                         name='linear_realtime_passtime')
        self.linear_user_cspu_obj = layers.Embedding(input_dim=sum(self.cfg.bucket_user_cspu_dims),
                                                     output_dim=1,
                                                     embeddings_initializer=self.cfg.embed_init,
                                                     embeddings_regularizer=self.cfg.kernel_regular,
                                                     input_length=None,
                                                     name='linear_user_cspu_obj')
        self.linear_ozid_cspu_obj = layers.Embedding(input_dim=sum(self.cfg.bucket_ozid_cspu_dims),
                                                     output_dim=1,
                                                     embeddings_initializer=self.cfg.embed_init,
                                                     embeddings_regularizer=self.cfg.kernel_regular,
                                                     input_length=None,
                                                     name='linear_ozid_cspu_obj')
        self.linear_user_behavior_obj = layers.Embedding(input_dim=sum(self.cfg.bucket_user_behavior_dims),
                                                         output_dim=1,
                                                         embeddings_initializer=self.cfg.embed_init,
                                                         embeddings_regularizer=self.cfg.kernel_regular,
                                                         input_length=None,
                                                         name='linear_user_behavior_obj')
        self.linear_goods_gross_obj = layers.Embedding(input_dim=sum(self.cfg.bucket_goods_gross_dims),
                                                       output_dim=1,
                                                       embeddings_initializer=self.cfg.embed_init,
                                                       embeddings_regularizer=self.cfg.kernel_regular,
                                                       input_length=None,
                                                       name='linear_goods_gross_obj')
        self.linear_realtime_user_group = layers.Embedding(input_dim=sum(self.cfg.realtime_user_group_dims),
                                                           output_dim=1,
                                                           embeddings_initializer=self.cfg.embed_init,
                                                           embeddings_regularizer=self.cfg.kernel_regular,
                                                           input_length=None,
                                                           name='linear_realtime_user_group')

    def call(self, x):
        bucket_goods_raw, \
        realtime_back_category, \
        realtime_goods, \
        realtime_pair_click, \
        realtime_passtime, \
        realtime_user_group, \
        pair_feature, \
        bucket_user_cspu_obj, \
        bucket_goods_box_obj, \
        bucket_pair_box_obj, \
        bucket_ozid_cspu_obj, \
        bucket_user_behavior_obj, \
        bucket_goods_gross_obj = x
        linear_raw = self.linear_raw(bucket_goods_raw)
        linear_pair = self.linear_pair(pair_feature)
        linear_realtime_back_category = self.linear_realtime_back_category(realtime_back_category)
        linear_realtime_goods = self.linear_realtime_goods(realtime_goods)
        linear_realtime_pair_click = self.linear_realtime_pair_click(realtime_pair_click)
        linear_pair_box_obj = self.linear_pair_box_obj(bucket_pair_box_obj)  ## [1, 62, embed_dim]
        linear_goods_box_obj = self.linear_goods_box_obj(bucket_goods_box_obj)  ## [1, 62, embed_dim]
        linear_realtime_passtime = self.linear_realtime_passtime(realtime_passtime)
        linear_user_cspu_obj = self.linear_user_cspu_obj(bucket_user_cspu_obj)
        linear_ozid_cspu_obj = self.linear_ozid_cspu_obj(bucket_ozid_cspu_obj)
        linear_user_behavior_obj = self.linear_user_behavior_obj(bucket_user_behavior_obj)
        linear_goods_gross_obj = self.linear_goods_gross_obj(bucket_goods_gross_obj)
        linear_realtime_user_group = self.linear_realtime_user_group(realtime_user_group)

        linear_pair = tf.reduce_sum(tf.squeeze(linear_pair, 2), 1, keepdims=True)
        linear_realtime_back_category = tf.reduce_sum(tf.squeeze(linear_realtime_back_category, 2), 1, keepdims=True)
        linear_realtime_goods = tf.reduce_sum(tf.squeeze(linear_realtime_goods, 2), 1, keepdims=True)
        linear_pair_box_obj = tf.reduce_sum(tf.squeeze(linear_pair_box_obj, 2), 1, keepdims=True)
        linear_goods_box_obj = tf.reduce_sum(tf.squeeze(linear_goods_box_obj, 2), 1, keepdims=True)
        linear_realtime_passtime = tf.reduce_sum(tf.squeeze(linear_realtime_passtime, 2), 1, keepdims=True)
        linear_user_cspu_obj = tf.reduce_sum(tf.squeeze(linear_user_cspu_obj, 2), 1, keepdims=True)
        linear_ozid_cspu_obj = tf.reduce_sum(tf.squeeze(linear_ozid_cspu_obj, 2), 1, keepdims=True)
        linear_user_behavior_obj = tf.reduce_sum(tf.squeeze(linear_user_behavior_obj, 2), 1, keepdims=True)
        linear_goods_gross_obj = tf.reduce_sum(tf.squeeze(linear_goods_gross_obj, 2), 1, keepdims=True)
        linear_realtime_user_group = tf.reduce_sum(tf.squeeze(linear_realtime_user_group, 2), 1, keepdims=True)

        linear = tf.concat(
            [linear_raw, linear_pair,
             linear_realtime_back_category, linear_realtime_goods, linear_realtime_pair_click,
             linear_pair_box_obj, linear_goods_box_obj, linear_realtime_passtime,
             linear_user_cspu_obj, linear_ozid_cspu_obj, linear_user_behavior_obj,
             linear_goods_gross_obj, linear_realtime_user_group],
            axis=1, name="linear_concat")
        return linear

    def get_config(self):
        config = super(BuildLinear, self).get_config().copy()
        config.update({
            'linear_raw': self.linear_raw,
            'linear_pair': self.linear_pair,
            'linear_realtime_back_category': self.linear_realtime_back_category,
            'linear_realtime_goods': self.linear_realtime_goods,
            'linear_realtime_pair_click': self.linear_realtime_pair_click,
            'linear_pair_box_obj': self.linear_pair_box_obj,
            'linear_goods_box_obj': self.linear_goods_box_obj,
            'linear_realtime_passtime': self.linear_realtime_passtime,
            'linear_user_cspu_obj': self.linear_user_cspu_obj,
            'linear_ozid_cspu_obj': self.linear_ozid_cspu_obj,
            'linear_user_behavior_obj': self.linear_user_behavior_obj,
            'linear_goods_gross_obj': self.linear_goods_gross_obj,
            'linear_realtime_user_group': self.linear_realtime_user_group
        })
        return config


class BuildFM(tf.keras.Model):
    def __init__(self, cfg):
        super(BuildFM, self).__init__()
        self.cfg = cfg

    def call(self, x):
        summed_features_emb = tf.reduce_sum(x, 1)
        summed_features_emb_square = tf.square(summed_features_emb)
        # square_sum part
        squared_features_emb = tf.square(x)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)
        # second order
        fm = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)
        return fm


class BuildDeep(tf.keras.Model):
    def __init__(self, cfg):
        super(BuildDeep, self).__init__()
        self.cfg = cfg
        self.dense1 = layers.Dense(128,
                                   activation=None,
                                   kernel_regularizer=self.cfg.kernel_regular,
                                   use_bias=True)
        self.bn1 = layers.BatchNormalization(axis=-1, momentum=0.99)
        self.relu1 = layers.ReLU()
        self.dense2 = layers.Dense(64,
                                   activation=None,
                                   kernel_regularizer=self.cfg.kernel_regular,
                                   use_bias=True)
        self.bn2 = layers.BatchNormalization(axis=-1, momentum=0.99)
        self.relu2 = layers.ReLU()
        self.dense3 = layers.Dense(32,
                                   activation=None,
                                   kernel_regularizer=self.cfg.kernel_regular,
                                   use_bias=True)
        self.bn3 = layers.BatchNormalization(axis=-1, momentum=0.99)
        self.relu3 = layers.ReLU()

    def call(self, x):
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dense3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def get_config(self):
        config = super(BuildDeep, self).get_config().copy()
        config.update({
            'dense1': self.dense1,
            'bn1': self.bn1,
            'relu1': self.relu1,
            'dense2': self.dense2,
            'bn2': self.bn2,
            'relu2': self.relu2,
            'dense3': self.dense3,
            'bn3': self.bn3,
            'relu3': self.relu3,
        })
        return config


if __name__ == "__main__":
    n_layers = 2
    d_model = 512
    num_heads = 8
    middle_units = 1024
    max_seq_len = 60

    samples = 10
    training = False

    seq = np.random.randint(0, 108, size=(samples, max_seq_len))  # [bs,seq_length]

    encode_padding_mask_list = padding_mask(seq)
    input_data = tf.random.uniform((samples, max_seq_len, d_model))

    sample_encoder = Encoder(n_layers, d_model, num_heads, middle_units, max_seq_len, training)
    sample_encoder_output = sample_encoder([input_data, encode_padding_mask_list])

    print(sample_encoder_output.shape)
