
import tensorflow as tf

def batch_data(input_x, labels):
    input_layer_x = tf.reshape(input_x, [-1, 28, 28, 1])
    dataset = tf.data.Dataset.from_tensor_slices((input_layer_x, labels))
    shuffled_dataset = dataset.shuffle(60000)
    batched_dataset = shuffled_dataset.batch(64, drop_remainder=True)
    return batched_dataset

def cross_entropy_loss(y_pred, y):
    # Compute cross entropy loss
    sparse_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
    return tf.reduce_mean(sparse_ce)

def accuracy(y_pred, y):
    # Compute accuracy after extracting class predictions
    class_preds = tf.argmax(tf.nn.softmax(y_pred), axis=1)
    is_equal = tf.equal(y, class_preds)
    return tf.reduce_mean(tf.cast(is_equal, tf.float32))
