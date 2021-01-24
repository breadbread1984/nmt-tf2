#!/usr/bin/python3

from math import floor;
import tensorflow as tf;
import tensorflow_addons as tfa;

def Cell(unit_type = 'lstm', units = None, drop_rate = 0, forget_bias = False, residual = False):

  assert unit_type in ['lstm','gru','layer_norm_lstm','nas'];
  if unit_type == 'lstm':
    cell = tf.keras.layers.LSTMCell(units, unit_forget_bias = forget_bias);
  elif unit_type == 'gru':
    cell = tf.keras.layers.GRUCell(units);
  elif unit_type == 'layer_norm_lstm':
    cell = tfa.rnn.LayerNormLSTMCell(units, unit_forget_bias = forget_bias);
  elif unit_type == 'nas':
    cell = tfa.rnn.NASCell(units);
  else:
    raise 'unknown cell type!';
  if drop_rate > 0:
    cell = tf.nn.RNNCellDropoutWrapper(cell, input_keep_prob = 1. - drop_rate);
  if residual == True:
    cell = tf.nn.RNNCellResidualWrapper(cell);
  return cell;  

def NMT(src_vocab_size, tgt_vocab_size, input_dims, enc_type = 'bi', unit_type = 'lstm', units = None, drop_rate = 0, forget_bias = False, residual_layer_num = None, layer_num = None):

  assert type(units) is int;
  assert type(residual_layer_num) is int;
  assert type(layer_num) is int;
  if enc_type == 'uni':
    cells = list();
    for i in range(layer_num):
      # NOTE: cells over certain layer use residual structure to prevent gradient vanishing
      cells.append(Cell(unit_type, units, drop_rate, forget_bias, i >= layer_num - residual_layer_num));
    encoder = tf.keras.layers.RNN(cells, return_sequences = False, return_state = True); # results = (last output, last output, last state)
  elif enc_type == 'bi':
    layer_num = floor(layer_num / 2);
    residual_layer_num = floor(residual_layer_num / 2);
    forward_cells = list();
    backward_cells = list();
    for i in range(layer_num):
      # NOTE: cells over certain layer use residual structure to prevent gradient vanishing
      forward_cells.append(Cell(unit_type, units, drop_rate, forget_bias, i >= layer_num - residual_layer_num));
      backward_cells.append(Cell(unit_type, units, drop_rate, forget_bias, i >= layer_num - residual_layer_num));
    encoder = tf.keras.layers.Bidirectional(
      layer = tf.keras.layers.RNN(forward_cells, return_sequences = False, return_state = True),
      backward_layer = tf.keras.layers.RNN(backward_cells, return_sequences = False, return_state = True, go_backwards = True),
      merge_mode = 'concat'); # results = (last output, last output of forward, last state of forward, first output of backward, first state of backward)
  else:
    raise 'unknown encoder type!';
  sampler = tfa.seq2seq.TrainingSampler();
  output_layer = tf.keras.layers.Dense(tgt_vocab_size);
  decoder = tfa.seq2seq.BasicDecoder(encoder, sampler, output_layer);

  inputs = tf.keras.Input((None, 1)); # inputs.shape = (batch, length, 1)
  input_tensors = tf.keras.layers.Embeddings(src_vocab_size, input_dims)(inputs); # results.shape = (batch, length, input_dims)
  input_lengths = tf.keras.layers.Lambda(lambda x: tf.ones((tf.shape(x)[0],), dtype = tf.int64) * tf.shape(x)[1])(inputs); # input_lengths.shape = (batch)
  initial_state = decoder_cell.get_initial_state(input_lengths); # initial_state = (last_output, state)
  output, state, lengths = decoder(input_tensors, sequence_length = input_lengths, initial_state = initial_state);
  return tf.keras.Model(inputs = inputs, outputs = output.rnn_output);

if __name__ == "__main__":
  
  assert tf.executing_eagerly();
