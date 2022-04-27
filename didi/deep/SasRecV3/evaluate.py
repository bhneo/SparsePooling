import tensorflow as tf

def evaluate_model(model, test):
    """
    evaluate model
    :param model: model
    :param test: test set
    :return: auc
    """
    sum_auc = 0
    num_batch = 500
    for step, (user_feat_inputs, user_seq_inputs, sample_cspuidx_inputs, item_feat_inputs, label_inputs) in enumerate(test):
        user_inputs = tf.concat([user_feat_inputs, user_seq_inputs], 1)
        item_inputs = tf.concat([sample_cspuidx_inputs, item_feat_inputs], 1)
        logits = model([user_inputs, item_inputs])
        # compute auc
        logits = logits.numpy().tolist()
        logits = [ele[0] for ele in logits]
        label_inputs = label_inputs.numpy().tolist()
        label_inputs = [ele[0] for ele in label_inputs]
        auc = compute_auc(logits, label_inputs)
        sum_auc += auc
    return sum_auc/num_batch

def compute_auc(scores, labels):
    f = list(zip(scores,labels))
    rank = [v2 for v1, v2 in sorted(f, key=lambda x:x[0])]
    rank_list = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    pos_num, neg_num = 0, 0
    for i in range(len(labels)):
        if labels[i] == 1:
            pos_num += 1
        else:
            neg_num += 1
    auc = (sum(rank_list) - (pos_num * (pos_num + 1) / 2))/ (pos_num * neg_num)

    return auc
