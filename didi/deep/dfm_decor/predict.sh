#创建日志目录
log_path="./log/"
if [ -d  ${log_path} ]; then
echo "${log_path} is exists !!!!"
else
mkdir -p ${log_path}
fi

train_time=$1
test_time=$2

#batch_size=$3
batch_size=1000
lr=0.001
embed_dim=$4
base='baseline'
version='v0'
name="${base}_d${embed_dim}_${version}"
steps_per_epoch=500
validation_steps=200

tfrecords="/user/prod_recommend/dm_recommend/zhaolei/base/features/merge/tfrecords/TAG=deep_feature_v8_1"
workspace="log"

# ---------------------------- 从hdfs获取上一次的 best model(best模型地址第一次需要先手动创建)
model_path="./log/checkpoint/${base}_d${embed_dim}_v0.h5"
predict_local_path="./log/predict"
predict_hdfs_dir="/user/prod_recommend/dm_recommend/zhaolei/base/model/${name}/predict/${test_time}"

python  ./main.py \
--batch_size ${batch_size} \
--lr ${lr} \
--embed_dim ${embed_dim} \
--train_time ${train_time} \
--test_time ${test_time} \
--name ${name} \
--tfrecords ${tfrecords} \
--workspace ${workspace} \
--predict_hdfs_dir ${predict_hdfs_dir} \
--mode 'predict' \
--steps_per_epoch ${steps_per_epoch} \
--validation_steps ${validation_steps} \
--model_path ${model_path} \
--predict_local_path ${predict_local_path}



