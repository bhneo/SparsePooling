
#创建日志目录
log_path="./log/"
if [ -d  ${log_path} ]; then
echo "${log_path} is exists !!!!"
else
mkdir -p ${log_path}
fi

train_time=$1
test_time=$2

batch_size=$3
lr=0.001
embed_dim=$4
base='baseline'
version='v0'
name="${base}_d${embed_dim}_${version}"
mode="train"
steps_per_epoch=500
validation_steps=200

tfrecords="/user/prod_recommend/dm_recommend/zhaolei/base/features/merge/tfrecords/TAG=deep_feature_v8_1"
workspace="log"

python  ./main.py \
--batch_size ${batch_size} \
--lr ${lr} \
--embed_dim ${embed_dim} \
--train_time ${train_time} \
--test_time ${test_time} \
--name ${name} \
--tfrecords ${tfrecords} \
--workspace ${workspace} \
--mode ${mode} \
--steps_per_epoch ${steps_per_epoch} \
--validation_steps ${validation_steps}

##预测
#predict_batch_size=1000
#predict_mode="predict"
#
#hdfs_workpace="/user/prod_recommend/dm_recommend/ranking/model/model_cvr/deep/TAG=${tag}"
#
## ---------------------------- 从hdfs获取上一次的 best model(best模型地址第一次需要先手动创建)
#echo "[operation]:get old best model from : ${hdfs_workpace}/model/best/1"
#hadoop fs -get "${hdfs_workpace}/model/best/1"
#model_path="./best_old"
#echo "[operation]:mv local old best model to: ."
#mv ./1 ${model_path}
#predict_hdfs_dir="${hdfs_workpace}/predict_pk/${test_time}"
#predict_local_path="./predict_res_old"
#
#python  ./main.py \
#--batch_size ${predict_batch_size} \
#--lr ${lr} \
#--embed_dim ${embed_dim} \
#--train_time ${train_time} \
#--test_time ${test_time} \
#--name ${name} \
#--tfrecords ${tfrecords} \
#--workspace ${workspace} \
#--predict_hdfs_dir ${predict_hdfs_dir} \
#--mode ${predict_mode} \
#--steps_per_epoch ${steps_per_epoch} \
#--validation_steps ${validation_steps} \
#--model_path ${model_path} \
#--predict_local_path ${predict_local_path}
#
#
## 清理数据
#rm -r ${model_path}
#
#
###---------------------------- 今天新训练的最优模型
#echo "[operation]:get new best model from : ${hdfs_workpace}/model/daily/${train_time}/1"
#hadoop fs -get "${hdfs_workpace}/model/daily/${train_time}/1"
#model_path="./best_new"
#echo "[operation]:mv local new best model to: ."
#mv ./1 ${model_path}
#predict_hdfs_dir="${hdfs_workpace}/predict_pk/${test_time}"
#predict_local_path="./predict_res_new"
#
#python  ./main.py \
#--batch_size ${predict_batch_size} \
#--lr ${lr} \
#--embed_dim ${embed_dim} \
#--train_time ${train_time} \
#--test_time ${test_time} \
#--name ${name} \
#--tfrecords ${tfrecords} \
#--workspace ${workspace} \
#--predict_hdfs_dir ${predict_hdfs_dir} \
#--mode ${predict_mode} \
#--steps_per_epoch ${steps_per_epoch} \
#--validation_steps ${validation_steps} \
#--model_path ${model_path} \
#--predict_local_path ${predict_local_path}
#
#
## 清理数据
#rm -r ${model_path}
#
#echo ">>>>>>>>>> done !"


