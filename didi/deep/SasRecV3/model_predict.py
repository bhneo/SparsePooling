from model import *
from evaluate import *
from data_pipeline import *

class Config(object):
    embed_dim = 50
    item_fea_col = {'feat_num' : 300001, 'embed_dim': embed_dim}
    num_blocks = 2
    num_heads = 1
    hidden_units = 64
    dropout_rate = 0.2
    maxlen = 150
    l2_emb = 1e-6

cfg = Config()

# test the saved weights
data_path = "/user/prod_recommend/dm_recommend/mind_model/tfrecords/guhaoqi/sasrecV3/dataset_2021-10-23"
parquet_list = list_hdfs_tfrecords_file(data_path)
_, test = dataset_pipeline(parquet_list, 2048, epochs=1)

model = get_sasrec(cfg.maxlen, cfg.item_fea_col, cfg.embed_dim, cfg.l2_emb, cfg.dropout_rate,
                   cfg.num_heads, cfg.num_blocks, cfg.hidden_units, use_norm=True, causality=True)

weight_path = "train_models/my_weights.h5"
model.load_weights(weight_path)

auc = evaluate_model(model, test)
print("auc: ", auc)













