import datetime
import os
import subprocess

from utils import read_json, get_list_by_feature_list
from tensorflow.keras import initializers

import numpy as np
import tensorflow as tf
import random
import re


class Config(object):
    ## ---train compile----
    workspace_dir = None

    epochs = 100000
    batch_size = 1024
    prefetch_buffer_size = 10000
    shuffle_buffer_size = 20000
    steps_per_epoch = 5000
    validation_steps = 500
    init_lr = 1e-3
    embed_dim = 8
    whiten = ''

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


def parse(parser):
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--embed_dim', type=int, help='隐向量维度')
    parser.add_argument('--train_time', type=str, help='训练数据时间')
    parser.add_argument('--test_time', type=str, help='测试数据时间')
    parser.add_argument('--name', type=str, help='模型备注')
    parser.add_argument('--whiten', type=str, help='白化方案', default='')
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

    cfg = Config()

    cfg.batch_size = args.batch_size
    cfg.init_lr = args.lr
    cfg.embed_dim = args.embed_dim
    cfg.whiten = args.whiten
    cfg.workspace_dir = args.workspace
    cfg.steps_per_epoch = args.steps_per_epoch
    cfg.validation_steps = args.validation_steps
    return args, cfg


def date_range(start, end, step=1, format="%Y-%m-%d"):
    # date_range("2017-01-01", "2017-01-03")
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days
    return [strftime(strptime(start, format) + datetime.timedelta(i), format) for i in range(0, days + 1, step)]


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


def get_hdfs_path_list(path, train_data_end_time, days):
    train_data_start_time = datetime.datetime.strptime(train_data_end_time,
                                                       '%Y-%m-%d') - datetime.timedelta(days)  ## 7天数据训练
    train_data_start_time = train_data_start_time.strftime('%Y-%m-%d')
    date_list = date_range(train_data_start_time, train_data_end_time)
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
    date_list = date_range(train_data_start_time, train_data_end_time)
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


def dataset_pipeline(cfg, hdfs_path_list, epochs):
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
                    "output": label,
                })
        return data

    dataset = tf.data.TFRecordDataset(hdfs_path_list)
    dataset = dataset.map(parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=cfg.shuffle_buffer_size)
    dataset = dataset.repeat(epochs)  ## epoch
    dataset = dataset.prefetch(buffer_size=cfg.prefetch_buffer_size)
    dataset = dataset.batch(batch_size=cfg.batch_size)
    return dataset
