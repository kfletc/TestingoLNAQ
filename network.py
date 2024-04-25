# network.py
# contains functions for initializing weights, classes that represent types of neural network layers
# and a CNN class that defines the structure and forward pass of the convolutional neural network

import tensorflow as tf

def xavier_init_lin(shape):
    # Performs xavier initialization for weight matrices in fully connected layer
    in_dim, out_dim = shape
    xavier_val = tf.sqrt(6.0)/tf.sqrt(float(in_dim + out_dim))
    weight_vals = tf.random.uniform(shape=(in_dim, out_dim), minval=-xavier_val, maxval=xavier_val)
    return weight_vals

def xavier_init_conv(shape, filter_size):
    # Performs xavier initialization for kernel tensors in convolutional layers
    in_dim, out_dim = shape
    xavier_val = tf.sqrt(6.0)/tf.sqrt(float(in_dim + out_dim))
    kernel_vals = tf.random.uniform(shape=(filter_size, filter_size, in_dim, out_dim), minval=-xavier_val, maxval=xavier_val)
    return kernel_vals

class FullyConnectedLayer(tf.Module):
    def __init__(self, out_dim, weight_init=xavier_init_lin, activation=tf.identity, name=None):
        super().__init__(name=name)
        # initialize dimension, weight initialization, and activation function
        self.out_dim = out_dim
        self.weight_init = weight_init
        self.activation = activation
        self.built = False

    def __call__(self, x):
        # layer computation on call
        # initialize layer if not initialized
        if not self.built:
            self.in_dim = x.shape[1]
            self.w = tf.Variable(self.weight_init((self.in_dim, self.out_dim)))
            # biases can be initialized to 0
            self.b = tf.Variable(tf.zeros(shape=(self.out_dim,)))
            self.built = True
        # compute forward pass
        affine_output = tf.add(tf.matmul(x, self.w), self.b)
        return self.activation(affine_output)

class ConvLayer(tf.Module):
    def __init__(self, out_dim, kernel_size, strides=1, padding='VALID', weight_init=xavier_init_conv, activation=tf.identity, name=None):
        super().__init__(name=name)
        # Initialize dimension, kernel size, weight initialization, and activation
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.weight_init = weight_init
        self.activation = activation
        self.built = False

    def __call__(self, x):
        # layer computation on call
        # initialize layer if not initialized
        if not self.built:
            self.in_dim = x.shape[3]
            self.kernel = tf.Variable(self.weight_init((self.in_dim, self.out_dim), self.kernel_size))
            # biases can be initialized to 0
            self.b = tf.Variable(tf.zeros(shape=(self.out_dim,)))
            self.built = True
        # compute forward pass
        conv_output = tf.add(tf.nn.conv2d(x, self.kernel, self.strides, self.padding), self.b)
        return self.activation(conv_output)

class CNN(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        # specify layers in model
        self.conv_layer_1 = ConvLayer(16, 3, activation=tf.nn.relu)
        self.conv_layer_2 = ConvLayer(32, 3, activation=tf.nn.relu)
        self.connected_layer_1 = FullyConnectedLayer(100, activation=tf.nn.relu)
        self.connected_layer_2 = FullyConnectedLayer(10)

    @tf.function
    def __call__(self, x, preds=False):
        # forward pass through layers
        x_output = self.conv_layer_1(x)
        x_output = self.conv_layer_2(x_output)
        x_output = tf.nn.max_pool2d(x_output, 2, 2, 'VALID')
        x_output = tf.nn.dropout(x_output, 0.25)
        x_output = tf.reshape(x_output, [-1, 4608])
        x_output = self.connected_layer_1(x_output)
        x_output = self.connected_layer_2(x_output)
        return x_output
