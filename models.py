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

def Encoder(channels, enc_type, unit_type = 'lstm', units = None, drop_rate = 0, forget_bias = False, residual_layer_num = None, layer_num = None):

  assert channels is int;
  assert enc_type in ['uni', 'bi'];
  assert type(units) is int;
  assert type(residual_layer_num) is int;
  assert type(layer_num) is int;
  inputs = tf.keras.Input((None, channels)); # inputs.shape = (batch, length, channels)
  if enc_type == 'uni':
    cells = list();
    for i in range(layer_num):
      # NOTE: cells over certain layer use residual structure to prevent gradient vanishing
      cells.append(Cell(unit_type, units, drop_rate, forget_bias, i >= layer_num - residual_layer_num));
    results = tf.keras.layers.RNN(cells, return_sequences = False, return_state = True)(inputs); # results = (last output, last output, last state)
  elif enc_type == 'bi':
    layer_num = floor(layer_num / 2);
    residual_layer_num = floor(residual_layer_num / 2);
    forward_cells = list();
    backward_cells = list();
    for i in range(layer_num):
      # NOTE: cells over certain layer use residual structure to prevent gradient vanishing
      forward_cells.append(Cell(unit_type, units, drop_rate, forget_bias, i >= layer_num - residual_layer_num));
      backward_cells.append(Cell(unit_type, units, drop_rate, forget_bias, i >= layer_num - residual_layer_num));
    results = tf.keras.layers.Bidirectional(
      layer = tf.keras.layers.RNN(forward_cells, return_sequences = False, return_state = True),
      backward_layer = tf.keras.layers.RNN(backward_cells, return_sequences = False, return_state = True, go_backwards = True),
      merge_mode = 'concat')(inputs); # results = (last output, last output of forward, last state of forward, first output of backward, first state of backward)
  else:
    raise 'unknown encoder type!';
  # NOTE: return only the state part of RNN output
  return tf.keras.Model(inputs = inputs, outputs = results[1:]);

def Decoder(channels, src_vocab_size, tgt_vocab_size, max_length, unit_type = 'lstm', units = None, drop_rate = 0, forget_bias = False, residual_layer_num = None, layer_num = None):

  assert type(channels) is int;
  assert type(src_vocab_size) is int;
  assert type(tgt_vocab_size) is int;
  assert type(max_length) is int;
  assert type(units) is int;
  assert type(residual_layer_num) is int;
  assert type(layer_num) is int;
  inputs = tf.keras.Input((None, channels)); # inputs.shape = (batch, length, channels)
  cells = list();
  for i in range(layer_num):
    # NOTE: cells over certain layer use residual structure to prevent gradient vanishing
    cells.append(Cell(unit_type, units, drop_rate, forget_bias, i >= layer_num - residual_layer_num));
  decoder_cell = tf.keras.layers.RNN(cells, return_sequences = True);
  sampler = tfa.seq2seq.TrainingSampler();
  output_layer = tf.keras.layers.Dense(tgt_vocab_size);
  decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer);
  input_ids = tf.keras.layers.Lambda(lambda x, d, l: tf.random.uniform((tf.shape(x)[0], l), maxval = d, dtype = tf.int64), arguments = {'d': tgt_vocab_size, 'l': max_length})(inputs);
  input_tensors = tf.keras.layers.Embeddings(src_vocab_size, channels)(input_ids);
  input_lengths = tf.keras.layers.Lambda(lambda x, l: tf.ones((tf.shape(x)[0],), dtype = tf.int64) * l, arguments = {'l': max_length})(inputs);
  initial_state = decoder_cell.get_initial_state(input_lengths);
  output, state, lengths = decoder(input_tensors, sequence_length = input_lengths, initial_state = initial_state);
  return tf.keras.Model(inputs = inputs, outputs = output.rnn_output);

def NMT(src_vocab_size, tgt_vocab_size, input_dims, output_dims, max_len_infer = None, enc_type = 'bi', unit_type = 'lstm', units = None, drop_rate = 0, forget_bias = False, residual_layer_num = None, layer_num = None):

  assert type(units) is int;
  assert type(residual_layer_num) is int;
  assert type(layer_num) is int;
  inputs = tf.keras.Input((None, 1)); # inputs.shape = (batch, length, 1)
  results = tf.keras.layers.Embeddings(src_vocab_size, input_dims)(inputs); # results.shape = (batch, length, input_dims)
  results = Encoder(input_dims, enc_type, unit_type, units, drop_rate, forget_bias, residual_layer_num, layer_num)(results); # results.shape = (batch, length, units)
  if max_len_infer
  max_iteration = 
