#!/bin/bash
source ~/.bashrc

emb_path="serving_emb/emb_$1.txt"
model_path="serving_models/model_$1"

hdfs_emb_path="/user/prod_recommend/dm_recommend/hive/dm_recommend/cxyx_user_sasrec_v3_cspuidx_embedding/dt=$1"
hdfs_model_path="/user/prod_recommend/dm_recommend/guhaoqi/sasrec_model_v3/"

python3 main.py --date $1
python3 serving_model.py --date $1
python3 serving_emb.py --date $1

hdfs dfs -mkdir $hdfs_emb_path
hdfs dfs -put $emb_path $hdfs_emb_path
hdfs dfs -put $model_path $hdfs_model_path

# del_emb_path="serving_emb/emb_$2.txt"
# del_model_path="serving_models/model_$2"
# del_tf_path="/user/prod_recommend/dm_recommend/mind_model/tfrecords/guhaoqi/sasrecV3/dataset_$2"
# rm -rf $del_emb_path $del_model_path
# hdfs dfs -rm -r $del_tf_path

spark-sql -e "MSCK REPAIR TABLE dm_recommend.cxyx_user_sasrec_v3_cspuidx_embedding;" --queue root.recommend_prod