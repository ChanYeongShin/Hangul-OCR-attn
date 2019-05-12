__author__ = 'solivr'
__license__ = "GPL"

_GO = 750
_EOS = 751

from .encoder import deep_cnn, deep_bidirectional_lstm
from .decoder import *

import tensorflow as tf
from .config import Params, TrainingParams


def crnn_attention_fn(features, labels, mode, params):
    """
    CRNN model definition for ``tf.Estimator``.
    Combines ``deep_cnn`` and ``deep_bidirectional_lstm`` ``embedding_attention_decoder`` to define the model and adds seq2seq loss computation
    :param features: dictionary with keys : '`images`', '`images_widths`', '`image_url`'
    :param labels: string containing the transcriptions.
        Flattend (1D) array with encoded label (one code per character)
    :param mode: TRAIN, EVAL, PREDICT
    :param params: dictionary with keys: '`Params`', '`TrainingParams`'
    :return:
    """

    parameters = params.get('Params')
    training_params = params.get('TrainingParams')
    assert isinstance(parameters, Params)
    assert isinstance(training_params, TrainingParams)

    if mode == tf.estimator.ModeKeys.TRAIN:
        parameters.keep_prob_dropout = 0.7
    else:
        parameters.keep_prob_dropout = 1.0

    conv = deep_cnn(features['images'], input_channels=parameters.input_channels,
                    is_training=(mode == tf.estimator.ModeKeys.TRAIN), summaries=True)

    net_output, encoder_last_state = deep_bidirectional_lstm(conv)
    batch_size = tf.shape(net_output)[0]

    # # Compute seq_len from image width
    # n_pools = CONST.DIMENSION_REDUCTION_W_POOLING  # 2x2 pooling in dimension W on layer 1 and 2
    # seq_len_inputs = tf.cast(tf.divide(features['images_widths'], n_pools, name='seq_len_input_op') - 1, tf.int32)

    # net_output padding until length = 20
    # max_len_input = seq_len_inputs[tf.argmax(seq_len_inputs)]
    # pad_len = 20 - max_len_input
    # paddings = [[0,0],[0,pad_len],[0,0]]
    # net_output = tf.pad(net_output, paddings=paddings)
    # net_output = tf.reshape(net_output, [-1, 20, 512])

    decoder_start_token = tf.ones(shape=[batch_size, 1], dtype=tf.int32) * parameters.hangul.codes[750]
    decoder_end_token = tf.ones(shape=[batch_size, 1], dtype=tf.int32) * parameters.hangul.codes[751]

    if not mode == tf.estimator.ModeKeys.PREDICT:

        # Hangul and codes
        keys_hangul_units = parameters.hangul.hangul_units
        values_hangul_codes = parameters.hangul.codes

        # Convert string label to code label
        with tf.name_scope('str2code_conversion'):
            table_str2int = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(keys_hangul_units, values_hangul_codes), -1)
            labels_splited = tf.string_split(labels, delimiter=parameters.string_split_delimiter)
            codes = table_str2int.lookup(labels_splited.values)
            sparse_code_target = tf.sparse.SparseTensor(labels_splited.indices, codes, labels_splited.dense_shape)

            # convert sparse tensor to dense tensor to produce decoder_input
            decoder_input = tf.sparse.to_dense(sparse_code_target, default_value=parameters.hangul.codes[751])

            # find length from input (sparse tensor)
            seq_len_labels = tf.bincount(tf.cast(sparse_code_target.indices[:, 0], tf.int32))
            # seq_len_labels = tf.reshape(seq_len_labels, [-1,1])

        try:
            predictions_dict['filename'] = features['filename']
        except KeyError:
            pass

        decoder_cell, decoder_initial_state = build_decoder_cell(net_output, encoder_last_state, seq_len_labels,
                                                                 batch_size, parameters)

        outputs = build_decoder(decoder_input, decoder_cell, decoder_initial_state, parameters, seq_len_labels,
                                True, batch_size)

        masks = tf.sequence_mask(lengths=seq_len_labels+1,
                                 maxlen=tf.reduce_max(seq_len_labels+1),
                                 dtype=tf.float32, name='masks')
        # Loss
        # >>> Cannot have longer labels than predictions -> error

        predictions_dict = {'probs': tf.nn.softmax(outputs)}
        decoder_input_target = tf.concat([decoder_input, decoder_end_token], axis=1)
        loss_seq2seq = seq2seq.sequence_loss(logits=outputs,
                                             targets=decoder_input_target,
                                             weights=masks,
                                             average_across_timesteps=True,
                                             average_across_batch=True, )
        # Training summary for the current batch_loss
        tf.summary.scalar('loss', loss_seq2seq)

            # loss_seq2seq = tf.Print(loss_seq2seq, [loss_seq2seq], message='* Loss : ')

        global_step = tf.train.get_or_create_global_step()
        # # Create an ExponentialMovingAverage object
        ema = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=global_step, zero_debias=True)
        # Create the shadow variables, and add op to maintain moving averages
        maintain_averages_op = ema.apply([loss_seq2seq])

        # Train op
        # --------
        learning_rate = tf.train.exponential_decay(training_params.learning_rate, global_step,
                                                   training_params.learning_decay_steps,
                                                   training_params.learning_decay_rate,
                                                   staircase=True)

        if training_params.optimizer == 'ada':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        elif training_params.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
        elif training_params.optimizer == 'rms':
            optimizer = tf.train.RMSPropOptimizer(learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        opt_op = optimizer.minimize(loss_seq2seq, global_step=global_step)
        with tf.control_dependencies(update_ops + [opt_op]):
            train_op = tf.group(maintain_averages_op)

        # Summaries
        # ---------
        tf.summary.scalar('learning_rate', learning_rate)
        # tf.summary.scalar('losses/ctc_loss', loss_ctc)
        tf.summary.scalar('losses/seq2seq_loss', loss_seq2seq)
    else:
        loss_seq2seq, train_op = None, None


    if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.TRAIN]:
        with tf.name_scope('code2str_conversion'):
            keys_hangul_codes = tf.cast(parameters.hangul.codes, tf.int64)
            values_hangul_units = [c for c in parameters.hangul.hangul_units]
            table_int2str = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(keys_hangul_codes, values_hangul_units), '?')

            # Output is 2 list of length NUM_BEAM_PATHS with tensors of shape [Batch, ...]
            # sparse_code_pred, log_probability_seq2seq = tf.nn.ctc_beam_search_decoder(
            #     net_output,
            #     sequence_length=tf.cast(seq_len_inputs, tf.int32),
            #     merge_repeated=False,
            #     beam_width=100,
            #     top_paths=parameters.num_beam_paths)

            # sequence_lengths_pred = tf.bincount(tf.cast(outputs, tf.int32),
            #                                     minlength=tf.shape(net_output)[1])
            #
            # pred_chars = table_int2str.lookup(outputs)
            # predictions_dict['words'] = get_words_from_chars(pred_chars.values, sequence_lengths=sequence_lengths_pred)
            # predictions_dict['codes'] = outputs

#        with tf.name_scope('predictions'):
#
#            decoder_cell, decoder_initial_state = build_decoder_cell(net_output, encoder_last_state, None, batch_size, parameters)
#
#            outputs = build_decoder(decoder_start_token, decoder_cell, decoder_initial_state, parameters,
#                                    None, False, batch_size)
#            sequence_lengths_pred = tf.bincount(tf.cast(outputs, tf.int32))
#
#            pred_chars = table_int2str.lookup(tf.cast(outputs, tf.int64))
#
#            predictions_dict = {}
#            predictions_dict['words'] = get_words_from_chars(pred_chars, sequence_lengths=sequence_lengths_pred)
#            predictions_dict['codes'] = outputs
#            tf.summary.text('predicted_words', predictions_dict['words'][:10])

    # Compute these values only when predicting, they're not useful during training/evaluation
    if mode == tf.estimator.ModeKeys.PREDICT:
        with tf.name_scope('predictions'):

            decoder_cell, decoder_initial_state = build_decoder_cell(net_output, encoder_last_state, None, batch_size, parameters)

            outputs = build_decoder(decoder_start_token, decoder_cell, decoder_initial_state, parameters,
                                    None, False, batch_size)
            # sequence_lengths_pred = [tf.bincount(tf.cast(sp.indices[:, 0], tf.int32),
            #                                      minlength=tf.shape(net_output)[1])
            #                          for sp in outputs]
            #
            # pred_chars = [table_int2str.lookup(sp) for sp in outputs]
            #
            # predictions_dict['best_transcriptions'] = tf.stack(
            #     [get_words_from_chars(char.values, sequence_lengths=length)
            #      for char, length in zip(pred_chars, sequence_lengths_pred)]
            # )
            sequence_lengths_pred = tf.bincount(tf.cast(outputs, tf.int32))
#                                                minlength=tf.shape(net_output)[1])

            pred_chars = table_int2str.lookup(tf.cast(outputs, tf.int64))

            predictions_dict = {}
            predictions_dict['words'] = pred_chars
            #predictions_dict['words'] = get_words_from_chars(pred_chars, sequence_lengths=sequence_lengths_pred)
            predictions_dict['codes'] = outputs
            # #Score
            # predictions_dict['score'] = tf.subtract(loss_seq2seq[:, 0], loss_seq2seq[:, 1],
            #                                         name='score_computation')
            #
            # # Logprobs seq2seq decoding :
            # predictions_dict['logprob_seq2seq'] = loss_seq2seq
            tf.summary.text('predicted_words', predictions_dict['words'][:10])

    # Evaluation ops
    # --------------
    if mode == tf.estimator.ModeKeys.EVAL:
        with tf.name_scope('evaluation'):
#            CER = tf.metrics.mean(tf.edit_distance(outputs, tf.cast(codes, dtype=tf.int64)), name='CER')

            # Convert label codes to decoding hangul to compare predicted and groundtrouth words
            target_chars = table_int2str.lookup(tf.cast(codes, tf.int64))
            target_words = get_words_from_chars(target_chars, seq_len_labels)

            sequence_lengths_pred = tf.bincount(tf.cast(outputs, tf.int32))

            pred_chars = table_int2str.lookup(tf.cast(outputs, tf.int64))

            predcitions_dict = {}
            predictions_dict['words'] = get_words_from_chars(pred_chars, seq_len_labels)
            accuracy = tf.metrics.accuracy(target_words, predictions_dict['words'], name='accuracy')

            eval_metric_ops = {
                'eval/accuracy': accuracy,
#                'eval/CER': CER,
            }
#            CER = tf.Print(CER, [CER], message='-- CER : ')
#            accuracy = tf.Print(accuracy, [accuracy], message='-- Accuracy : ')

    else:
        eval_metric_ops = None
    export_outputs = {'predictions': tf.estimator.export.PredictOutput(predictions_dict)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_dict,
        loss=loss_seq2seq,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=export_outputs,
        scaffold=tf.train.Scaffold()
    )
