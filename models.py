#!/usr/bin/python3

from math import floor;
import tensorflow as tf;
import tensorflow_addons as tfa;

def Cell(unit_type = 'lstm', units = None, drop_rate = 0, forget_bias = False, residual = False):

  assert unit_type in ['lstm','gru','layer_norm_lstm','nas'];
  if unit_type == 'lstm':
    cell = tf.keras.layers.LSTMCell(units, unit_forget_bias = forget_bias, dropout = drop_rate, recurrent_dropout = drop_rate);
  elif unit_type == 'gru':
    cell = tf.keras.layers.GRUCell(units, dropout = drop_rate, recurrent_dropout = drop_rate);
  elif unit_type == 'layer_norm_lstm':
    cell = tfa.rnn.LayerNormLSTMCell(units, unit_forget_bias = forget_bias, dropout = drop_rate, recurrent_dropout = drop_rate);
  elif unit_type == 'nas':
    cell = tfa.rnn.NASCell(units);
  else:
    raise 'unknown cell type!';
  if residual == True:
    cell = tf.nn.RNNCellResidualWrapper(cell);
  return cell;  

def NMT(src_vocab_size, tgt_vocab_size, input_dims, is_train = False, 
        encoder_params = {'enc_type': 'uni', 'unit_type': 'lstm', 'units': 32, 'drop_rate': 0.2, 'forget_bias': 1.0, 'use_residual': True, 'residual_layer_num': 1, 'layer_num': 2},
        decoder_params = {'unit_type': 'lstm', 'units': 32, 'drop_rate': 0.2, 'forget_bias': 1.0, 'use_residual': True, 'residual_layer_num': 1, 'layer_num': 2},
        infer_params = {'infer_mode': 'beam_search', 'max_infer_len': None, 'beam_width': 0, 'start_token': 1, 'end_token': 2, 'length_penalty_weight': 0., 'coverage_penalty_weight': 0., 'softmax_temperature': 0.}):

  assert infer_params['infer_mode'] in ['beam_search', 'sample', 'greedy'];
  if encoder_params['use_residual'] and encoder_params['layer_num'] > 1: encoder_params['residual_layer_num'] = encoder_params['layer_num'] - 1;
  if decoder_params['use_residual'] and decoder_params['layer_num'] > 1: decoder_params['residual_layer_num'] = decoder_params['layer_num'] - 1;

  inputs = tf.keras.Input((None, 1), ragged = True); # inputs.shape = (batch, ragged length, 1)
  inputs = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis = -1))(inputs); # inputs.shape = (batch, ragged length)
  input_tensors = tf.keras.layers.Embedding(src_vocab_size, input_dims)(inputs); # input_tensors.shape = (batch, ragged length, input_dims)
  if is_train == True:
    targets = tf.keras.Input((None, 1)); # targets.shape = (batch, ragged length, 1)
    targets = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis = -1))(targets); # targets.shape = (batch, ragged length)
    target_tensors = tf.keras.layers.Embedding(target_vocab_size, input_dims)(targets); # target_tensors.shape = (batch, ragged length, input_dims)
  batch = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.map_fn(lambda x: 1, x, fn_output_signature = tf.TensorSpec((), dtype = tf.int32))))(inputs); # batch.shape = ()
  # 1) encoder
  if encoder_params['enc_type'] == 'uni':
    cells = list();
    for i in range(encoder_params['layer_num']):
      # NOTE: cells over certain layer use residual structure to prevent gradient vanishing
      cells.append(Cell(encoder_params['unit_type'], encoder_params['units'], encoder_params['drop_rate'], encoder_params['forget_bias'], i >= encoder_params['layer_num'] - encoder_params['residual_layer_num']));
    dummy, hidden, cell = tf.keras.layers.RNN(cells, return_state = True)(input_tensors); # hidden.shape = (batch, units) cell.shape = (batch, units)
  elif encoder_params['enc_type'] == 'bi':
    layer_num = floor(layer_num / 2);
    residual_layer_num = floor(residual_layer_num / 2);
    forward_cells = list();
    backward_cells = list();
    for i in range(encoder_params['layer_num']):
      # NOTE: cells over certain layer use residual structure to prevent gradient vanishing
      forward_cells.append(Cell(encoder_params['unit_type'], encoder_params['units'], encoder_params['drop_rate'], encoder_params['forget_bias'], i >= encoder_params['layer_num'] - encoder_params['residual_layer_num']));
      backward_cells.append(Cell(encoder_params['unit_type'], encoder_params['units'], encoder_params['drop_rate'], encoder_params['forget_bias'], i >= encoder_params['layer_num'] - encoder_params['residual_layer_num']));
    concated_hidden, forward_hidden, forward_cell, backward_hidden, backward_cell = tf.keras.layers.Bidirectional(
      layer = tf.keras.layers.RNN(forward_cells, return_state = True),
      backward_layer = tf.keras.layers.RNN(backward_cells, return_state = True, go_backwards = True),
      merge_mode = 'concat')(input_tensors); # results = (concated hidden, forward hidden, forward cell, backward hidden, backward cell)
    hidden = tf.keras.layers.Concatenate(axis = -1)([forward_hidden, backward_hidden]); # hidden.shape = (batch, units * 2)
    cell = tf.keras.layers.Concatenate(axis = -1)([forward_cell, backward_cell]); # cell.shape = (batch, units * 2)
  else:
    raise 'unknown encoder type!';
  # 2) decoder
  cells = list();
  for i in range(decoder_params['layer_num']):
    # NOTE: cells over certain layer use residual structure to prevent gradient vanishing
    cells.append(Cell(decoder_params['unit_type'], decoder_params['units'], decoder_params['drop_rate'], decoder_params['forget_bias'], i >= decoder_params['layer_num'] - decoder_params['residual_layer_num']));
  # NOTE: decoder can't use bidirectional RNN, because the output length is unknown
  decoder_cell = tf.keras.layers.StackedRNNCells(cells);
  # NOTE: decoder RNN -> dense -> softmax -> output ids
  output_layer = tf.keras.layers.Dense(tgt_vocab_size, use_bias = False);
  if is_train == True:
    sampler = tfa.seq2seq.TrainingSampler();
    decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer);
  else:
    start_tokens = tf.keras.layers.Lambda(lambda x, s: tf.ones((x,), dtype = tf.int32) * s, arguments = {'s': infer_params['start_token']})(batch);
    if infer_params['infer_mode'] == 'beam_search':
      decoder = tfa.seq2seq.BeamSearchDecoder(decoder_cell, beam_width = infer_params['beam_width'], start_tokens = start_tokens, end_tokens = infer_params['end_token'], length_penalty_weight = infer_params['length_penalty_weight'], coverage_penalty_weight = infer_params['coverage_penalty_weight']);
    else:
      if infer_params['infer_mode'] == 'sample':
        sampler = tfa.seq2seq.SampleEmbeddingSampler(softmax_temperature = infer_params['softmax_temperature']);
      elif infer_params['infer_mode'] == 'greedy':
        sampler = tfa.seq2seq.GreedyEmbeddingSampler();
      else:
        raise 'unknown infer mode!';
      # get maximum_iterations
      if infer_params['max_infer_len']:
        maximum_iterations = tf.keras.layers.Lambda(lambda x, l: tf.ones((x,), dtype = tf.int32) * l, arguments = {'l': infer_params['max_infer_len']})(batch); # max_infer_length = (batch)
      else:
        maximum_iterations = tf.keras.layers.Lambda(lambda x: tf.ones((x[0],), dtype = tf.int32) * 2 * tf.math.reduce_max(tf.map_fn(lambda x: tf.shape(x)[0], x[1])))([batch, inputs]); # max_infer_length = (batch)
      decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer, maximum_iterations = maximum_iterations);
  if is_train == True:
    # NOTE: has target_tensors for supervision at training mode
    target_length = tf.keras.layers.Lambda(lambda x: tf.map_fn(lambda x: tf.shape(x)[0], x))(targets);
    output, state, lengths = decoder(target_tensors, sequence_length = target_length, initial_state = (hidden, cell));
  else:
    # NOTE: no target_tensors for supervision at inference mode
    output, state, lengths = decoder(None, start_tokens = start_tokens, end_tokens = infer_params['end_token'], initial_state = (hidden, cell));
  # NOTE: rnn_output.shape = (batch, ragged length, tgt_vocab_size) predicted_ids.shape = (batch, beam_width)
  return tf.keras.Model(inputs = (inputs, targets) if is_train == True else inputs, outputs = output.rnn_output if is_train == True or infer_params['infer_mode'] != 'beam_search' else output.predicted_ids);

if __name__ == "__main__":
  
  assert tf.executing_eagerly();
  nmt = NMT(100,200,64);
  nmt.save('nmt.h5');
