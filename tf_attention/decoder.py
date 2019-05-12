import math
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import MultiRNNCell

from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

from typing import List



def get_words_from_chars(characters_list: List[str], sequence_lengths: List[int], name='chars_conversion') -> tf.Tensor:
    """
    Joins separated characters to form words.
    :param characters_list: List of hangul symbols (characters) to join to form words
    :param sequence_lengths: list containing the length of each string sequence
    :param name: op name to add to tf graph
    :return:
    """
    with tf.name_scope(name=name):
        def join_charcaters_fn(coords):
            return tf.reduce_join(characters_list[coords[0]:coords[1]])

        def coords_several_sequences():
            end_coords = tf.cumsum(sequence_lengths)
            start_coords = tf.concat([[0], end_coords[:-1]], axis=0)
            coords = tf.stack([start_coords, end_coords], axis=1)
            coords = tf.cast(coords, dtype=tf.int32)
            return tf.map_fn(join_charcaters_fn, coords, dtype=tf.string)

        def coords_single_sequence():
            return tf.reduce_join(characters_list, keep_dims=True)

        words = tf.cond(tf.shape(sequence_lengths)[0] > 1,
                        true_fn=lambda: coords_several_sequences(),
                        false_fn=lambda: coords_single_sequence())

        return words



# Building decoder cell and attention. Also returns decoder_initial_state
def build_decoder_cell(encoder_output: tf.Tensor, encoder_last_state, seq_length_input, batch_size, params):

    decoder_input_length = seq_length_input # shape : [batch_size]

    # To use BeamSearchDecoder, encoder_outputs, encoder_last_state, encoder_inputs_length
    # needs to be tiled so that: [batch_size, .., ..] -> [batch_size x beam_width, .., ..]

    if params.use_beamsearch_decode:
        encoder_output = seq2seq.tile_batch(encoder_output, multiplier=params.beam_width)
        encoder_last_state = nest.map_structure(lambda s: seq2seq.tile_batch(s, params.beam_width), encoder_last_state)
        decoder_input_length = seq2seq.tile_batch(decoder_input_length, multiplier=params.beam_width)

    # Building attention mechanism: Default Bahdanau
    # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
#    encoder_output = tf.Print(encoder_output, [encoder_output])
    attention_mechanism = seq2seq.BahdanauAttention(
    num_units=params.hidden_units, memory=encoder_output)
#    memory_sequence_length=decoder_input_length)

    # 'Luong' style attention: https://arxiv.org/abs/1508.04025
    if params.attention_type == 'luong':
        attention_mechanism = seq2seq.LuongAttention(
            num_units=params.hidden_units, memory=encoder_output,
            memory_sequence_length=decoder_input_length)

    # Building decoder_cell
    num_units = [256,256] # depth = 2
    num_units = [256]  # depth = 1
    decoder_cells = [LSTMCell(num_units=n) for n in num_units]

    def attn_decoder_input_fn(inputs, attention):
        if not params.attn_input_feeding:
            return inputs

        # Essential when use_residual=True
        _input_layer = Dense(params.hidden_units, dtype=tf.float32,
                             name='attn_input_feeding')
        return _input_layer(tf.concat([inputs, attention], -1))

    # AttentionWrapper wraps RNNCell with the attention_mechanism
    # Note: We implement Attention mechanism only on the top decoder layer
    decoder_cells[-1] = seq2seq.AttentionWrapper(
        cell=decoder_cells[-1],
        attention_mechanism=attention_mechanism,
        attention_layer_size=params.hidden_units,
        cell_input_fn=attn_decoder_input_fn,
        initial_cell_state=encoder_last_state[-1],
        alignment_history=False,
        name='Attention_Wrapper')

    # To be compatible with AttentionWrapper, the encoder last state
    # of the top layer should be converted into the AttentionWrapperState form
    # We can easily do this by calling AttentionWrapper.zero_state

    # Also if beamsearch decoding is used, the batch_size argument in .zero_state
    # should be ${decoder_beam_width} times to the origianl batch_size
    if not params.use_beamsearch_decode:
        batch_size = batch_size
    else:
        batch_size = batch_size * params.beam_width

    initial_state = [state for state in encoder_last_state]
    initial_state[-1] = decoder_cells[-1].zero_state(
        batch_size = batch_size, dtype=tf.float32)
    decoder_initial_state2 = tuple(initial_state)

    return MultiRNNCell(decoder_cells), decoder_initial_state2


def build_decoder(decoder_input, decoder_cell, decoder_initial_state, params, seq_length_input, is_training: bool, batch_size):
    with tf.variable_scope('decoder'):
        decoder_start_token = tf.ones(shape=[batch_size, 1], dtype=tf.int32) * params.hangul.codes[750]
#        print(decoder_start_token)
        decoder_end_token = tf.ones(shape=[batch_size, 1], dtype=tf.int32) * params.hangul.codes[751]
        # Initialize decoder embeddings to have variance=1.
        sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=tf.float32)

        decoder_embeddings = tf.get_variable(name='embedding',
                                                  shape=[params.hangul.n_classes, params.embedding_size],
                                                  initializer=initializer, dtype=tf.float32)

        # Input projection layer to feed embedded inputs to the cell
        # ** Essential when use_residual=True to match input/output dims
        input_layer = Dense(params.hidden_units, dtype=tf.float32, name='input_projection')

        # Output projection layer to convert cell_outputs to logits
        output_layer = Dense(params.hangul.n_classes, name='output_projection')

        if is_training:
            decoder_input_train = tf.concat([decoder_start_token,
                                                  decoder_input], axis=1)

            # encoder_output_embedded: [batch_size, max_time_step + 1, embedding_size]
            decoder_input_embedded = tf.nn.embedding_lookup(
                params=decoder_embeddings, ids=decoder_input_train)

#            print("decoder_input_embedded", decoder_input_embedded.shape)

            # Embedded inputs having gone through input projection layer
            decoder_input_embedded = input_layer(decoder_input_embedded)
#            print("decoder_input_embedded2", decoder_input_embedded.shape)
            decoder_inputs_length_train = seq_length_input + 1

#            print("seq_len_input", seq_length_input)
            # Helper to feed inputs for training: read inputs from dense ground truth vectors

#            decoder_input_embedded = tf.Print(decoder_input_embedded,
#                    [decoder_input_embedded])
#            decoder_inputs_length_train = tf.Print(decoder_inputs_length_train,
#                   [decoder_inputs_length_train])
            # decoder_cell = tf.Print(decoder_cell, [decoder_cell])

            training_helper = seq2seq.TrainingHelper(inputs=decoder_input_embedded,
                                                     sequence_length=decoder_inputs_length_train,
                                                     time_major=False,
                                                     name='training_helper')
#            print(training_helper)

            training_decoder = seq2seq.BasicDecoder(cell=decoder_cell,
                                                    helper=training_helper,
                                                    initial_state=decoder_initial_state,
                                                    output_layer=output_layer)
#            print(training_decoder)

            # Maximum decoder time_steps in current batch
            max_decoder_length = tf.reduce_max(decoder_inputs_length_train)
            
#            max_decoder_length = tf.Print(max_decoder_length,
#                    [max_decoder_length])

#            print("max_decoder_len:", max_decoder_length)

            # decoder_outputs_train: BasicDecoderOutput
            #                        namedtuple(rnn_outputs, sample_id)
            # decoder_outputs_train.rnn_output: [batch_size, max_time_step + 1, num_decoder_symbols] if output_time_major=False
            #                                   [max_time_step + 1, batch_size, num_decoder_symbols] if output_time_major=True
            # decoder_outputs_train.sample_id: [batch_size], tf.int32

 #            seq2seq.dynamic_decode(
 #                decoder=training_decoder,
 #                output_time_major=False,
 #                impute_finished=True,
 #                maximum_iterations=max_decoder_length)


            (decoder_outputs_train, decoder_last_state_train,
             decoder_outputs_length_train) = (seq2seq.dynamic_decode(
                decoder=training_decoder,
                output_time_major=False,
                # impute_finished=True,
                impute_finished=False,
                maximum_iterations=max_decoder_length))

            # More efficient to do the projection on the batch-time-concatenated tensor
            # logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
            # self.decoder_logits_train = output_layer(self.decoder_outputs_train.rnn_output)
            outputs = tf.identity(decoder_outputs_train.rnn_output)
            # Use argmax to extract decoder symbols to emit


        else:


            def embed_and_input_proj(inputs):
                return input_layer(tf.nn.embedding_lookup(decoder_embeddings, inputs))

            if not params.use_beamsearch_decode:
                # Helper to feed inputs for greedy decoding: uses the argmax of the output
                decoding_helper = seq2seq.GreedyEmbeddingHelper(
                        start_tokens=tf.reshape(decoder_start_token, shape=[-1]),
                        end_token=params.hangul.codes[751],
                        embedding=embed_and_input_proj)
                # Basic decoder performs greedy decoding at each time step
                print("building greedy decoder..")
                inference_decoder = seq2seq.BasicDecoder(cell=decoder_cell,
                                                         helper=decoding_helper,
                                                         initial_state=decoder_initial_state,
                                                         output_layer=output_layer)
            else:
                # Beamsearch is used to approximately find the most likely translation
                print("building beamsearch decoder..")
                inference_decoder = seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                                          embedding=embed_and_input_proj,
                                                                          start_tokens=decoder_start_token,
                                                                          end_token=decoder_end_token,
                                                                          initial_state=decoder_initial_state,
                                                                          beam_width=params.beam_width,
                                                                          output_layer=output_layer, )
            # For GreedyDecoder, return
            # decoder_outputs_decode: BasicDecoderOutput instance
            #                         namedtuple(rnn_outputs, sample_id)
            # decoder_outputs_decode.rnn_output: [batch_size, max_time_step, num_decoder_symbols] 	if output_time_major=False
            #                                    [max_time_step, batch_size, num_decoder_symbols] 	if output_time_major=True
            # decoder_outputs_decode.sample_id: [batch_size, max_time_step], tf.int32		if output_time_major=False
            #                                   [max_time_step, batch_size], tf.int32               if output_time_major=True

            # For BeamSearchDecoder, return
            # decoder_outputs_decode: FinalBeamSearchDecoderOutput instance
            #                         namedtuple(predicted_ids, beam_search_decoder_output)
            # decoder_outputs_decode.predicted_ids: [batch_size, max_time_step, beam_width] if output_time_major=False
            #                                       [max_time_step, batch_size, beam_width] if output_time_major=True
            # decoder_outputs_decode.beam_search_decoder_output: BeamSearchDecoderOutput instance
            #                                                    namedtuple(scores, predicted_ids, parent_ids)

            (decoder_outputs_decode, decoder_last_state_decode,
             decoder_outputs_length_decode) = (seq2seq.dynamic_decode(
                decoder=inference_decoder,
                output_time_major=False,
                # impute_finished=True,	# error occurs
                maximum_iterations=params.max_decode_step))

            if not params.use_beamsearch_decode:
                # decoder_outputs_decode.sample_id: [batch_size, max_time_step]
                # Or use argmax to find decoder symbols to emit:
                # self.decoder_pred_decode = tf.argmax(self.decoder_outputs_decode.rnn_output,
                #                                      axis=-1, name='decoder_pred_decode')

                # Here, we use expand_dims to be compatible with the result of the beamsearch decoder
                # decoder_pred_decode: [batch_size, max_time_step, 1] (output_major=False)
                outputs = tf.expand_dims(decoder_outputs_decode.sample_id, -1)

            else:
                # Use beam search to approximately find the most likely translation
                # decoder_pred_decode: [batch_size, max_time_step, beam_width] (output_major=False)
                outputs = decoder_outputs_decode.predicted_ids

    return outputs

