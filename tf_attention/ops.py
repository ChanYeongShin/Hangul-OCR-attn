import tensorflow as tf


def weight_var(shape, mean=0.0, stddev=0.02, name='weights'):
    init_w = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(init_w, name=name)


def bias_var(shape, value=0.0, name='bias'):
    init_b = tf.constant(value=value, shape=shape)
    return tf.Variable(init_b, name=name)


def batch_norm(input, is_training):
    return tf.contrib.layers.batch_norm(input, is_training=is_training, scale=True, decay=0.99)


def conv2d(input, filter, strides=(1, 1, 1, 1), padding='SAME', name=None):
    return tf.nn.conv2d(input, filter, strides=strides, padding=padding, name=name )


def max_2x2pool(input, name='2x2pooling'):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')


def max_2x1pool(input, name='2x1pooling'):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input, ksize=(1, 2, 1, 1), strides=(1, 2, 1, 1), padding='SAME')


def dropout(input, keep_prob):
    return tf.contrib.layers.dropout(input, keep_prob=keep_prob)


def relu(input, name='relu'):
    return tf.nn.relu(input, name=name)


