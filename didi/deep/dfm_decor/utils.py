import json
import os

import tensorflow as tf


def get_list_by_feature_list(feature_list, dims_dict):
    rst = []
    for k in feature_list:
        rst.append(dims_dict[k])
    return rst


def read_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def print_process_info(rst):
    if rst.returncode == 0:
        print("success:", rst)
    else:
        print("error:", rst)


def mkdirs(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)
        print("mkdirs: {} ,ok!".format(pth))


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

    return focal_loss_fixed


def dice_loss(smooth=1e-6):
    def dice_loss_fixed(y_true, y_pred):
        # flatten label and prediction tensors
        intersection = K.sum(y_true * y_pred)
        dice = (2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)

        return 1 - dice

    return dice_loss_fixed
