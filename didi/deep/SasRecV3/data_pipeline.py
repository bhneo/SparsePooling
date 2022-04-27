import subprocess
import tensorflow as tf
import re

def list_hdfs_tfrecords_file(path):
    cat = subprocess.Popen(["hdfs", "dfs", "-ls", "{}".format(path)], stdout=subprocess.PIPE)
    parquet_list = []
    pattern = re.compile(r"/user/.+part-r-\d+")
    for line in cat.stdout:
        if re.search(pattern, str(line)) is not None:
            parquet_list.append(re.search(pattern, str(line)).group(0))

    hdfs_path = ["hdfs://difed{}".format(s) for s in parquet_list]  # commit
    return hdfs_path

def dataset_pipeline(hdfs_path_list, batch_size, epochs=1):
    features = {
        'user_feat': tf.io.FixedLenFeature([20], tf.float32),
        'user_seq': tf.io.FixedLenFeature([150], tf.int64),
        'sample_cspuidx': tf.io.FixedLenFeature([1], tf.int64),
        'item_feat': tf.io.FixedLenFeature([35], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.int64)
    }

    def parse_record(record):
        parsed = tf.io.parse_single_example(record, features=features)
        user_feat_inputs = tf.cast(parsed['user_feat'], tf.float32)
        user_seq_inputs = tf.cast(parsed['user_seq'], tf.float32)
        sample_cspuidx_inputs = tf.cast(parsed['sample_cspuidx'], tf.float32)
        item_feat_inputs = tf.cast(parsed['item_feat'], tf.float32)
        label_inputs = tf.cast(parsed['label'], tf.int32)

        return [user_feat_inputs, user_seq_inputs, sample_cspuidx_inputs, item_feat_inputs, label_inputs]

    dataset = tf.data.TFRecordDataset(hdfs_path_list)
    dataset = dataset.map(parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(epochs)
    dataset = dataset.shuffle(buffer_size=batch_size*50)
    dataset = dataset.prefetch(buffer_size=batch_size*50)
    dataset = dataset.batch(batch_size=batch_size)

    return dataset.skip(500).take(30000), dataset.take(500)

def item_feat_pipeline(hdfs_path_list, batch_size, epochs=1):
    features = {
        'cspu_idx': tf.io.FixedLenFeature([1], tf.int64),
        'feat': tf.io.FixedLenFeature([35], tf.float32)
    }

    def parse_record(record):
        parsed = tf.io.parse_single_example(record, features=features)
        cspu_idx_inputs = tf.cast(parsed['cspu_idx'], tf.float32)
        feat_inputs = tf.cast(parsed['feat'], tf.float32)

        return [cspu_idx_inputs, feat_inputs]

    dataset = tf.data.TFRecordDataset(hdfs_path_list)
    dataset = dataset.map(parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(epochs)
    # dataset = dataset.shuffle(buffer_size=batch_size * 50)
    # dataset = dataset.prefetch(buffer_size=batch_size * 50)
    dataset = dataset.batch(batch_size=batch_size)

    return dataset
