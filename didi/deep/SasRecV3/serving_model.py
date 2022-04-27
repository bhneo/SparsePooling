import os
# import tensorflow as tf
from model import *
# from data_pipeline import *
# from tensorflow import keras
from tensorflow.keras.models import Model
import argparse

class Config(object):
    embed_dim = 50
    item_fea_col = {'feat_num' : 300001, 'embed_dim': embed_dim}
    num_blocks = 2
    num_heads = 1
    hidden_units = 64
    dropout_rate = 0.2
    maxlen = 150
    l2_emb = 1e-6

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default="2021-10-23", type=str)
    args = parser.parse_args()
    if not os.path.isdir('serving_models'): os.makedirs('serving_models')
    serving_model_path = "serving_models/model_{}".format(args.date)

    cfg = Config()
    model = get_sasrec(cfg.maxlen, cfg.item_fea_col, cfg.embed_dim, cfg.l2_emb, cfg.dropout_rate,
                       cfg.num_heads, cfg.num_blocks, cfg.hidden_units, use_norm=True, causality=True)
    weight_path = "train_models/my_weights_{}.h5".format(args.date)
    model.load_weights(weight_path)

    serving_model = Model(inputs=model.user_inputs,outputs=model.user_outputs)
    serving_model.save(serving_model_path)

    ## test serving model output
    # data_path = "/user/prod_recommend/dm_recommend/mind_model/tfrecords/guhaoqi/sasrecV3/dataset_2021-10-23"
    # parquet_list = list_hdfs_tfrecords_file(data_path)
    # _, test = dataset_pipeline(parquet_list, 100, epochs=1)
    # test = test.take(1)
    #
    # for step, (user_feat_inputs, user_seq_inputs, sample_cspuidx_inputs, item_feat_inputs, label_inputs) in enumerate(test):
    #     user_inputs = tf.concat([user_feat_inputs, user_seq_inputs],1)
    #     out = serving_model(user_inputs)
    #     print(out)

    print("serving model: success")

if __name__ == '__main__':
    main()