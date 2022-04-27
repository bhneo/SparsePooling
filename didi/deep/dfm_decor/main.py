#!/usr/bin/python
#-- coding:utf8 --

"""
    deepFM:
    样本：首页+去除无行为用户+去leader、仓店+负样本20%随机采样（正负样本1：30）；
    特征：93维实时特征+passtime+user_group+context+sequence+goods_sparse_v3.1+user分桶+goods分桶+pair分桶+cspu特征+用户偏好特征+商品毛利特征；
    deep：128+64+32，隐向量=10，去除L2和dropout；
    batch_size=10000
    利用transformer对用户行为序列进行建模
"""

import argparse
import os
import subprocess
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# coding: utf-8
import data_inputs
import modules
import decorrelation
import utils

path_append = os.path.join(os.path.abspath(os.path.dirname(__file__)), "./")
sys.path.append(path_append)

print("run: ", os.path.abspath(__file__))
print("tensorflow version: ", tf.__version__)
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    # 设置内存增长方式 自增长
    tf.config.experimental.set_memory_growth(gpu, True)

print(tf.config.list_physical_devices('GPU'))


def build_model(cfg):
    # inputs
    inputs = tf.keras.Input(shape=(cfg.all_len,), dtype=tf.float32, name="inputs")
    contexts, ids, seqs, users, goods, pairs = modules.ProcessInputs(cfg)(inputs)
    (realtime_features, context_features) = contexts
    (realtime_pair_click, realtime_back_category, realtime_goods, realtime_passtime, realtime_user_group) = realtime_features
    (goods_sparse, cspu_idx, supplier_idx, lv2_idx) = ids
    (long_seq, short_seq) = seqs
    (long_click, long_cart, long_buy,
     long_buy_level2, long_cart_level2, long_click_level2) = long_seq
    (short_click, short_cart, short_buy,
     short_click_level2, short_cart_level2, short_buy_level2) = short_seq
    (bucket_user_raw, bucket_user_box_obj) = users
    (bucket_goods_raw, bucket_goods_box_obj, bucket_goods_gross_obj) = goods
    (pair_feature, bucket_pair_box_obj, bucket_user_cspu_obj, bucket_ozid_cspu_obj, bucket_user_behavior_obj) = pairs

    # embedding
    context_embed, \
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
    short_click_level2_embed, short_cart_level2_embed, short_buy_level2_embed = modules.MakeEmbedding(cfg)((
                                                                                    context_features,
                                                                                    realtime_back_category, realtime_goods, realtime_pair_click, realtime_passtime, realtime_user_group,
                                                                                    goods_sparse,
                                                                                    bucket_user_box_obj, bucket_goods_box_obj, bucket_goods_gross_obj,
                                                                                    pair_feature,
                                                                                    bucket_pair_box_obj, bucket_user_cspu_obj, bucket_ozid_cspu_obj, bucket_user_behavior_obj,
                                                                                    cspu_idx, supplier_idx, lv2_idx,
                                                                                    long_click, long_cart, long_buy,
                                                                                    long_buy_level2, long_cart_level2, long_click_level2,
                                                                                    short_click, short_cart, short_buy,
                                                                                    short_click_level2, short_cart_level2, short_buy_level2))

    pooling_long_click_din, pooling_long_cart_din, pooling_long_buy_din, \
    pooling_short_click_din, pooling_short_cart_din, pooling_short_buy_din = modules.BuildSequence(cfg)((
                                                                                    long_click, long_click_embed,
                                                                                    long_cart, long_cart_embed,
                                                                                    long_buy, long_buy_embed,
                                                                                    long_buy_level2, long_buy_level2_embed,
                                                                                    long_cart_level2, long_cart_level2_embed,
                                                                                    long_click_level2, long_click_level2_embed,
                                                                                    short_click, short_click_embed,
                                                                                    short_cart, short_cart_embed,
                                                                                    short_buy, short_buy_embed,
                                                                                    short_click_level2, short_click_level2_embed,
                                                                                    short_cart_level2, short_cart_level2_embed,
                                                                                    short_buy_level2, short_buy_level2_embed,
                                                                                    cspu_embed, lv2_embed))

    linear = modules.BuildLinear(cfg)((bucket_goods_raw,
                                       realtime_back_category,
                                       realtime_goods,
                                       realtime_pair_click,
                                       realtime_passtime,
                                       realtime_user_group,
                                       pair_feature,
                                       bucket_user_cspu_obj,
                                       bucket_goods_box_obj,
                                       bucket_pair_box_obj,
                                       bucket_ozid_cspu_obj,
                                       bucket_user_behavior_obj,
                                       bucket_goods_gross_obj))

    embeddings = tf.keras.layers.Concatenate(axis=1, name="embedding_table")([
                                           realtime_passtime_embed, context_embed, realtime_back_category_embed, realtime_goods_embed,
                                           realtime_pair_click_embed, realtime_user_group_embed,
                                           pair_feature_embed,
                                           goods_sparse_embed,
                                           bucket_user_box_obj_embed, bucket_goods_box_obj_embed, bucket_pair_box_obj_embed,
                                           cspu_embed, supplier_embed, lv2_embed,
                                           bucket_user_cspu_obj_embed, bucket_ozid_cspu_obj_embed, bucket_user_behavior_obj_embed,
                                           bucket_goods_gross_obj_embed])

    if cfg.whiten != '':
        center_axis, group_axis, m, decomposition, iter_num, affine = cfg.whiten.split('-')
        embeddings = decorrelation.DecorelationNormalization(center_axis=int(center_axis),
                                                             group_axis=int(group_axis),
                                                             m_per_group=int(m),
                                                             decomposition=decomposition,
                                                             iter_num=int(iter_num),
                                                             affine=bool(int(affine)))(embeddings)

    fm = modules.BuildFM(cfg)(embeddings)

    deep_fc = tf.keras.layers.Reshape((embeddings.shape[1] * embeddings.shape[2], ))(embeddings)
    deep_fc = tf.keras.layers.Concatenate(axis=-1, name='deep_inputs')([
        deep_fc,
        bucket_user_raw, bucket_goods_raw,
        pooling_long_click_din, pooling_long_cart_din, pooling_long_buy_din,
        pooling_short_click_din, pooling_short_cart_din, pooling_short_buy_din])

    deep_fc = modules.BuildDeep(cfg)(deep_fc)

    concat_all = tf.keras.layers.Concatenate(axis=1, name='concat')([linear, fm, deep_fc])
    output = layers.Dense(1,
                          kernel_regularizer=cfg.kernel_regular,
                          activation='sigmoid',
                          use_bias=True,
                          name="output")(concat_all)
    return tf.keras.Model(inputs=[inputs], outputs=[output])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    args, cfg = data_inputs.parse(parser)
    mode = args.mode.split(",")

    if cfg.whiten == '':
        name = args.name
    else:
        name = args.name + '_{}'.format(cfg.whiten)
    log_dir = os.path.join(cfg.workspace_dir, 'tensorboard/{}'.format(name))
    checkpoint_dir = os.path.join(cfg.workspace_dir, 'checkpoint')
    utils.mkdirs(log_dir)
    utils.mkdirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, 'model_{}.h5'.format(name))

    tfrecords = args.tfrecords
    train_data_end_time = args.train_time
    test_data_end_time = args.test_time
    train_path, train_path_others = data_inputs.get_hdfs_path_list(tfrecords, train_data_end_time, days=16)
    test_path, val_path = data_inputs.get_hdfs_path_list_test(tfrecords, test_data_end_time, days=0)

    train_dataset = data_inputs.dataset_pipeline(cfg, hdfs_path_list=train_path + train_path_others, epochs=cfg.epochs)
    val_dataset = data_inputs.dataset_pipeline(cfg, hdfs_path_list=test_path + val_path, epochs=cfg.epochs * 10)
    test_dataset = data_inputs.dataset_pipeline(cfg, hdfs_path_list=test_path + val_path, epochs=1)

    # for feature, label in test_dataset.take(3):
    #     print(feature)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=6),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           save_weights_only=True,
                                           monitor='val_auc',
                                           mode='max',
                                           save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.2,
                                             patience=2, min_lr=1e-5, mode='max')
    ]

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = build_model(cfg)
        ## load model
        # if os.path.exists(checkpoint_path):
        #     model.load_weights(checkpoint_path, by_name=True)
        #     print("load model: {}".format(checkpoint_path))

        model.compile(optimizer=tf.keras.optimizers.Adam(cfg.init_lr),
                      loss={
                          "output": tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      },
                      metrics={
                          "output": [
                              tf.keras.metrics.AUC(),
                          ],
                      },
                      loss_weights={
                          "output": 1.0,
                      },
                      )
        model.summary()

        if "train" in mode:
            model.fit(train_dataset,
                      validation_data=val_dataset,
                      epochs=cfg.epochs,
                      # epochs=1,
                      steps_per_epoch=cfg.steps_per_epoch,
                      callbacks=my_callbacks,
                      validation_steps=cfg.validation_steps,
                      verbose=1)

            # save pd格式
            model.load_weights(checkpoint_path, by_name=True)
            pb_save_path = os.path.join(checkpoint_dir, "{}/{}".format(name, args.train_time), "1")
            tf.keras.models.save_model(model, pb_save_path)

        # model.load_weights(checkpoint_path, by_name=True) ## 加载存储的模型

        if "predict" in mode:
            test_dataset = data_inputs.dataset_pipeline(cfg, hdfs_path_list=test_path + val_path, epochs=1)
            # model = tf.keras.models.load_model(args.model_path)
            model.load_weights(args.model_path)

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
                label = labels['output'].numpy()
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
            utils.print_process_info(rst)
            rst = subprocess.run(
                ["hadoop", "fs", "-put", "-f", parquet_local_path, "hdfs://difed{}".format(args.predict_hdfs_dir)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            utils.print_process_info(rst)
            rst = subprocess.run(["rm", predict_local_path],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            utils.print_process_info(rst)
            rst = subprocess.run(["rm", parquet_local_path],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            utils.print_process_info(rst)

        if "pb" in mode:
            pb_save_path = os.path.join(checkpoint_dir, "{}_{}".format(args.train_time, name), "1")
            tf.keras.models.save_model(model, pb_save_path)

        if "evaluate" in mode:
            test_dataset = data_inputs.dataset_pipeline(cfg, hdfs_path_list=test_path + val_path, epochs=1)

            results = model.evaluate(test_dataset,
                                     verbose=1,
                                     steps=1000)
            print("test: ", results)
