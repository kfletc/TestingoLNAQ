# utils.py
# contains useful functions for preprocessing data, calculating accuracy and loss,
# and computing inner products of total model variables

import tensorflow as tf
import numpy as np

def batch_data(input_x, labels):
    # need to convert data from uint8 to datatypes tensorflow expects
    input_x = input_x.astype(np.float32)
    labels = labels.astype(np.int32)
    input_layer_x = tf.reshape(input_x, [-1, 28, 28, 1])
    dataset = tf.data.Dataset.from_tensor_slices((input_layer_x, labels))
    shuffled_dataset = dataset.shuffle(60000)
    batched_dataset = shuffled_dataset.batch(128, drop_remainder=True)
    return batched_dataset

def cross_entropy_loss(y_pred, y):
    # Compute cross entropy loss
    sparse_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
    return tf.reduce_mean(sparse_ce)

def accuracy(y_pred, y):
    # Compute accuracy after extracting class predictions
    class_preds = tf.argmax(tf.nn.softmax(y_pred), axis=1, output_type=tf.dtypes.int32)
    is_equal = tf.equal(y, class_preds)
    return tf.reduce_mean(tf.cast(is_equal, tf.float32))

def inner_product(tuple1, tuple2):
    total = 0
    for tensor1, tensor2 in zip(tuple1, tuple2):
        flattened1 = tf.reshape(tensor1, [-1])
        flattened2 = tf.reshape(tensor2, [-1])
        dot_product = tf.tensordot(flattened1, flattened2, axes=1)
        total += float(dot_product)
    return total
