__author__ = 'solivr'
__license__ = "GPL"

from .ops import *
import tensorflow as tf

from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import LSTMStateTuple


def deep_cnn(input_imgs: tf.Tensor, input_channels: int, is_training: bool, summaries: bool=True) -> tf.Tensor:
    """
    CNN part of the CRNN network.
    :param input_imgs: input images [B, H, W, C]
    :param input_channels: input channels, 1 for greyscale images, 3 for RGB color images
    :param is_training: flag to indicate training or not
    :param summaries: flag to enable bias and weight histograms to be visualized in Tensorboard
    :return: tensor of shape [batch, final_width, final_height x final_features]
    """
    assert (input_channels in [1, 3])

    input_tensor = input_imgs

    with tf.variable_scope('deep_cnn'):
        # - conv1 - maxPool2x2
        with tf.variable_scope('layer1'):
            W = weight_var([3, 3, input_channels, 64])
            b = bias_var([64])
            conv = conv2d(input_tensor, W, name='conv')
            out = tf.nn.bias_add(conv, b)
            conv1 = relu(out)
            pool1 = max_2x2pool(conv1)

            if summaries:
                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer1/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer1/bias:0'][0]
                tf.summary.histogram('bias', bias)

        # - conv2 - maxPool 2x2
        with tf.variable_scope('layer2'):
            W = weight_var([3, 3, 64, 128])
            b = bias_var([128])
            conv = conv2d(pool1, W)
            out = tf.nn.bias_add(conv, b)
            conv2 = relu(out)
            pool2 = max_2x2pool(conv2)

            if summaries:
                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer2/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer2/bias:0'][0]
                tf.summary.histogram('bias', bias)

        # - conv3 - w/batch-norm
        with tf.variable_scope('layer3'):
            W = weight_var([3, 3, 128, 256])
            b = bias_var([256])
            conv = conv2d(pool2, W)
            out = tf.nn.bias_add(conv, b)
            b_norm = batch_norm(out, is_training=is_training)
            conv3 = relu(b_norm, name='ReLU')

            if summaries:
                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer3/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer3/bias:0'][0]
                tf.summary.histogram('bias', bias)

        # - conv4 - maxPool 2x1
        with tf.variable_scope('layer4'):
            W = weight_var([3, 3, 256, 256])
            b = bias_var([256])
            conv = conv2d(conv3, W)
            out = tf.nn.bias_add(conv, b)
            conv4 = relu(out)
            pool4 = max_2x1pool(conv4)

            if summaries:
                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer4/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer4/bias:0'][0]
                tf.summary.histogram('bias', bias)

        # - conv5 - w/batch-norm
        with tf.variable_scope('layer5'):
            W = weight_var([3, 3, 256, 512])
            b = bias_var([512])
            conv = conv2d(pool4, W)
            out = tf.nn.bias_add(conv, b)
            b_norm = batch_norm(out, is_training=is_training)
            conv5 = relu(b_norm)

            if summaries:
                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer5/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer5/bias:0'][0]
                tf.summary.histogram('bias', bias)

        # - conv6 - maxPool 2x1
        with tf.variable_scope('layer6'):
            W = weight_var([3, 3, 512, 512])
            b = bias_var([512])
            conv = conv2d(conv5, W)
            out = tf.nn.bias_add(conv, b)
            conv6 = relu(out)
            pool6 = max_2x1pool(conv6)

            if summaries:
                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer6/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer6/bias:0'][0]
                tf.summary.histogram('bias', bias)

        # - conv 7 - w/batch-norm
        with tf.variable_scope('layer7'):
            W = weight_var([2, 2, 512, 512])
            b = bias_var([512])
            conv = conv2d(pool6, W, padding='VALID')
            out = tf.nn.bias_add(conv, b)
            b_norm = batch_norm(out, is_training=is_training)
            conv7 = relu(b_norm)

            if summaries:
                weights = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer7/weights:0'][0]
                tf.summary.histogram('weights', weights)
                bias = [var for var in tf.global_variables() if var.name == 'deep_cnn/layer7/bias:0'][0]
                tf.summary.histogram('bias', bias)

        cnn_net = conv7

        with tf.variable_scope('Reshaping_cnn'):
            shape = tf.shape(cnn_net)  # [batch, height, width, features]
            transposed = tf.transpose(cnn_net, perm=[0, 2, 1, 3],
                                      name='transposed')  # [batch, width, height, features]
            conv_reshaped = tf.reshape(transposed, [shape[0], shape[2], shape[1] * shape[3]],
                                       name='reshaped')  # [batch, width, height x features]
            # Setting shape
            shape_list = cnn_net.get_shape().as_list()
            conv_reshaped.set_shape([None, shape_list[2], shape_list[1] * shape_list[3]])

    return conv_reshaped

def deep_bidirectional_lstm(inputs: tf.Tensor) -> tf.Tensor:
    """
    Recurrent part of the CRNN encoder network.
    Uses a biderectional LSTM.
    :param inputs: output of ``deep_cnn``
    :return : [batch, width(time), num_hidden], encoder_last_state
    """

    # list_n_hidden = [256, 256] # if depth = 2
    list_n_hidden = [256]

    with tf.name_scope('deep_bidirectional_lstm'):
        # Forward direction cells
        fw_cell_list = [LSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]
        # Backward direction cells
        bw_cell_list = [LSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]

        lstm_net, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell_list,
                                                                        bw_cell_list,
                                                                        inputs,
                                                                        dtype=tf.float32)
        encoder_last_state = []

        for i in range(len(output_state_fw)):
            # c = tf.concat((output_state_fw[i].c ,output_state_bw[i].c), 1) # if depth = 2
            c = output_state_fw[i].c
            # h = tf.concat((output_state_fw[i].h, output_state_bw[i].h), 1) # if depth = 2
            h = output_state_fw[i].h

            encoder_last_state.append(LSTMStateTuple(c=c, h=h))

        # encoder_last_state = tf.concat(axis=1, values=[output_state_fw, output_state_bw])
        encoder_last_state = tuple(encoder_last_state)

        # Dropout layer
        # lstm_out = dropout(lstm_net, keep_prob=params.keep_prob_dropout) # [batch, width, 2*n_hidden]
        lstm_out = lstm_net # [batch, width, depth * n_hidden]

        # with tf.variable_scope('Reshaping_rnn'):
        #     # shape = lstm_net.get_shape().as_list()  # [batch, width, 2*n_hidden]
        #     shape = tf.shape(lstm_net)
        #     rnn_reshaped = tf.reshape(lstm_net, [shape[0]*shape[1], shape[2]])  # [batch x width, 2*n_hidden]
        #
        #
        # lstm_out = tf.reshape(rnn_reshaped, [shape[0], shape[1], shape[2]]) # [batch, width(time), layer_output]

        # raw_pred = tf.argmax(tf.nn.softmax(lstm_out), axis=2, name='raw_prediction')

        # Swap batch and time axis
        # lstm_out = tf.transpose(lstm_out, [1, 0, 2], name='transpose_time_major')  # [width(time), batch, n_classes]

        return lstm_out, encoder_last_state
