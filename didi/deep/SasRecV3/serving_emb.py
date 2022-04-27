import os
import tensorflow as tf
from model import *
from data_pipeline import *
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
    parser.add_argument('--date', default="2021-10-25", type=str)
    args = parser.parse_args()
    if not os.path.isdir('serving_emb'): os.makedirs('serving_emb')
    item_embedding_path = "serving_emb/emb_{}.txt".format(args.date)

    cfg = Config()
    model = get_sasrec(cfg.maxlen, cfg.item_fea_col, cfg.embed_dim, cfg.l2_emb, cfg.dropout_rate,
                       cfg.num_heads, cfg.num_blocks, cfg.hidden_units, use_norm=True, causality=True)
    weight_path = "train_models/my_weights.h5"   # "train_models/my_weights_{}.h5".format(args.date)
    model.load_weights(weight_path)

    item_tower = Model(inputs=model.item_inputs,outputs=model.item_outputs)
    data_path = "/user/prod_recommend/dm_recommend/mind_model/tfrecords/guhaoqi/sasrecV3/item_feat/dataset_{}".format(args.date)
    parquet_list = list_hdfs_tfrecords_file(data_path)
    item_inp = item_feat_pipeline(parquet_list, 1, epochs=1)

    for step, (cspu_idx_inputs, feat_inputs) in enumerate(item_inp):
        inp = tf.concat([cspu_idx_inputs, feat_inputs], 1)
        cspu_idx = cspu_idx_inputs.numpy()
        item_vec = item_tower(inp).numpy()
        with open(item_embedding_path, "at") as f:
            cspu_idx_tmp = int(cspu_idx[0][0])
            item_vec_tmp = ','.join('%s' % ele for ele in item_vec[0])
            f.write("{}\t{}\n".format(cspu_idx_tmp, item_vec_tmp))

    print("serving emb: success")

if __name__ == '__main__':
    main()