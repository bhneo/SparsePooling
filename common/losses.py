import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils import losses_utils


class MarginLoss(keras.losses.Loss):
    def __init__(self,
                 sparse=True,
                 upper_margin=0.9,
                 bottom_margin=0.1,
                 down_weight=0.5,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name=None):
        super(MarginLoss, self).__init__(reduction=reduction, name=name)
        self.sparse = sparse
        self.upper_margin = upper_margin
        self.bottom_margin = bottom_margin
        self.down_weight = down_weight

    def call(self, y_true, y_pred):
        num_out = y_pred.get_shape().as_list()[-1]
        if self.sparse:
            y_true = tf.reshape(y_true, [-1])
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_out)
        return margin_loss(y_true, y_pred, self.upper_margin, self.bottom_margin, self.down_weight)

    def get_config(self):
        config = {'upper_margin': self.upper_margin, 'bottom_margin': self.bottom_margin,
                  'down_weight': self.down_weight}
        base_config = super(MarginLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SpreadLoss(keras.losses.Loss):
    def __init__(self,
                 sparse=True,
                 upper_margin=0.9,
                 bottom_margin=0.1,
                 down_weight=0.5,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name=None):
        super(SpreadLoss, self).__init__(reduction=reduction, name=name)
        self.sparse = sparse
        self.upper_margin = upper_margin
        self.bottom_margin = bottom_margin
        self.down_weight = down_weight

    def call(self, y_true, y_pred):
        num_out = y_pred.get_shape().as_list()[-1]
        if self.sparse:
            y_true = tf.reshape(y_true, [-1])
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_out)
        return margin_loss(y_true, y_pred, self.upper_margin, self.bottom_margin, self.down_weight)

    def get_config(self):
        config = {'upper_margin': self.upper_margin, 'bottom_margin': self.bottom_margin,
                  'down_weight': self.down_weight}
        base_config = super(SpreadLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def spread_loss(labels, predictions, margin):
    a_target = tf.reduce_sum(labels * predictions, axis=1, keepdims=True)
    dist = (1 - labels) * margin - (a_target - predictions)
    loss = tf.pow(tf.maximum(0., dist), 2)
    return loss


def margin_loss(labels,
                predictions,
                upper_margin=0.9,
                bottom_margin=0.1,
                down_weight=0.5):
    labels = tf.cast(labels, tf.float32)
    positive_selector = tf.cast(tf.less(predictions, upper_margin), tf.float32)
    positive_cost = positive_selector * labels * tf.pow(predictions - upper_margin, 2)

    negative_selector = tf.cast(tf.greater(predictions, bottom_margin), tf.float32)
    negative_cost = negative_selector * (1 - labels) * tf.pow(predictions - bottom_margin, 2)
    loss = positive_cost + down_weight * negative_cost
    return loss


def spread_loss(scores, y):
    """Spread loss.

    "In order to make the training less sensitive to the initialization and
    hyper-parameters of the model, we use “spread loss” to directly maximize the
    gap between the activation of the target class (a_t) and the activation of the
    other classes. If the activation of a wrong class, a_i, is closer than the
    margin, m, to at then it is penalized by the squared distance to the margin."

    See Hinton et al. "Matrix Capsules with EM Routing" equation (3).

    Author:
      Ashley Gritzman 19/10/2018
    Credit:
      Adapted from Suofei Zhang's implementation on GitHub, "Matrix-Capsules-
      EM-Tensorflow"
      https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow
    Args:
      scores:
        scores for each class
        (batch_size, num_class)
      y:
        index of true class
        (batch_size, 1)
    Returns:
      loss:
        mean loss for entire batch
        (scalar)
    """

    with tf.variable_scope('spread_loss') as scope:
        batch_size = int(scores.get_shape()[0])

        # AG 17/09/2018: modified margin schedule based on response of authors to
        # questions on OpenReview.net:
        # https://openreview.net/forum?id=HJWLfGWRb
        # "The margin that we set is:
        # margin = 0.2 + .79 * tf.sigmoid(tf.minimum(10.0, step / 50000.0 - 4))
        # where step is the training step. We trained with batch size of 64."
        global_step = tf.to_float(tf.train.get_global_step())
        m_min = 0.2
        m_delta = 0.79
        m = (m_min
             + m_delta * tf.sigmoid(tf.minimum(10.0, global_step / 50000.0 - 4)))

        num_class = int(scores.get_shape()[-1])

        y = tf.one_hot(y, num_class, dtype=tf.float32)

        # Get the score of the target class
        # (64, 1, 5)
        scores = tf.reshape(scores, shape=[batch_size, 1, num_class])
        # (64, 5, 1)
        y = tf.expand_dims(y, axis=2)
        # (64, 1, 5)*(64, 5, 1) = (64, 1, 1)
        at = tf.matmul(scores, y)

        # Compute spread loss, paper eq (3)
        loss = tf.square(tf.maximum(0., m - (at - scores)))

        # Sum losses for all classes
        # (64, 1, 5)*(64, 5, 1) = (64, 1, 1)
        # e.g loss*[1 0 1 1 1]
        loss = tf.matmul(loss, 1. - y)

        # Compute mean
        loss = tf.reduce_mean(loss)

    return loss

