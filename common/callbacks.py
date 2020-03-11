import tensorflow as tf
import numpy as np


class LearningRateBatchScheduler(tf.keras.callbacks.Callback):
    """Callback to update learning rate on every batch (not epoch boundaries).

    N.B. Only support Keras optimizers, not TF optimizers.

    Attributes:
        schedule: a function that takes an epoch index and a batch index as input
            (both integer, indexed from 0) and returns a new learning rate as
            output (float).
    """

    def __init__(self, schedule, batch_size, num_images):
        super(LearningRateBatchScheduler, self).__init__()
        self.schedule = schedule
        self.batches_per_epoch = num_images / batch_size
        self.batch_size = batch_size
        self.epochs = -1
        self.prev_lr = -1

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'learning_rate'):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        self.epochs += 1
        print('\nEpoch %05d: LearningRateScheduler reducing learning '
              'rate to %s.' % (epoch+1, self.model.optimizer.learning_rate))

    def on_batch_begin(self, batch, logs=None):
        """Executes before step begins."""
        lr = self.schedule(self.epochs,
                           batch,
                           self.batches_per_epoch,
                           self.batch_size)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function should be float.')
        if lr != self.prev_lr and batch % 100 == 0:
            self.model.optimizer.learning_rate = lr  # lr should be a float here
            self.prev_lr = lr
            print('  Epoch %05d Batch %05d: LearningRateBatchScheduler '
                  'change learning rate to %s.' % (self.epochs+1, batch, lr))
