import argparse
import os
import tensorflow as tf
from model import *
from data_pipeline import *
from tensorflow import keras
from evaluate import *

# train file
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default="2021-10-19", type=str)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=1e-5, type=float)
    parser.add_argument('--embed_dim', default=50, type=int)
    parser.add_argument('--item_num', default=300001, type=int)
    parser.add_argument('--maxlen', default=150, type=int) # change before getting data

    args = parser.parse_args()
    if not os.path.isdir('train_models'):
        os.makedirs('train_models')
    if not os.path.isdir('emb'):
        os.makedirs('emb')

    data_path = "/user/prod_recommend/dm_recommend/mind_model/tfrecords/guhaoqi/sasrecV3/dataset_{}".format(args.date)

    if not os.path.isdir('train_args'):
        os.makedirs('train_args')
    with open(os.path.join('train_args', 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    parquet_list = list_hdfs_tfrecords_file(data_path)
    train, test = dataset_pipeline(parquet_list, args.batch_size, epochs=1)

    print("finish loading dataset..")
    item_fea_col = {'feat_num': args.item_num, 'embed_dim': args.embed_dim}

    # build model
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = get_sasrec(args.maxlen, item_fea_col, args.embed_dim, args.l2_emb, args.dropout_rate,
                           args.num_heads, args.num_blocks, args.hidden_units, use_norm = True, causality = True)
        model.summary()
        # results = []
        print("start training...")
        optimizer = keras.optimizers.Adam(args.lr)

        # distributed training
        dist_train = mirrored_strategy.experimental_distribute_dataset(train)
        def compute_loss(logits, labels):
            # BCE
            losses = tf.keras.losses.binary_crossentropy(labels, logits, from_logits=False)
            loss = tf.reduce_mean(losses)
            return loss

        def train_step(inputs):
            with tf.GradientTape() as tape:
                user_inp, item_inp, labels = inputs
                logits = model([user_inp, item_inp])
                # auc = evaluate_model(model, test)
                loss_value = compute_loss(logits, labels)
            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss_value

        def distributed_train_step(dist_inputs):
            per_replica_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
            return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                            axis=None)

        for epoch in range(1, args.num_epochs + 1):
            print("epoch:", epoch)

            for step, (user_feat_inputs, user_seq_inputs, sample_cspuidx_inputs, item_feat_inputs, label_inputs) in enumerate(dist_train):
                user_inputs = tf.concat([user_feat_inputs, user_seq_inputs], 1)   # 20+150+1+35-> (none,306)
                item_inputs = tf.concat([sample_cspuidx_inputs, item_feat_inputs], 1)
                cur_loss = distributed_train_step([user_inputs, item_inputs, label_inputs])
                if step % 500 == 0:
                    cur_auc = evaluate_model(model, test)
                    print("Epoch: %d, Step: %d, ;loss: %.4f ;auc: %.4f" % (epoch, step, cur_loss, cur_auc))


    print("Finish training!")

    # compute model's auc on the test data
    auc = evaluate_model(model, test)
    print('The trained model has an auc of %.4f on the test dataset!'%auc)

    # save weights and model
    tmp_path = "train_models/my_weights_{}.h5".format(args.date)
    model.save_weights(tmp_path)


if __name__ == '__main__':
    main()