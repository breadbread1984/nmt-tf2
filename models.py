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

def Encoder(src_vocab_size, input_dims, encoder_params = {'enc_type': 'uni', 'unit_type': 'lstm', 'units': 32, 'drop_rate': 0.2, 'forget_bias': 1.0, 'use_residual': True, 'residual_layer_num': 1, 'layer_num': 2}):

  # NOTE: this is the commonly used encoder cross the nmt models
  assert encoder_params['enc_type'] in ['uni', 'bi'];
  assert encoder_params['unit_type'] in ['lstm','gru','layer_norm_lstm','nas'];
  if encoder_params['use_residual'] and encoder_params['layer_num'] > 1: encoder_params['residual_layer_num'] = encoder_params['layer_num'] - 1;
  inputs = tf.keras.Input((None, 1), ragged = True); # inputs.shape = (batch, ragged length, 1)
  squeezed_inputs = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis = -1))(inputs); # inputs.shape = (batch, ragged length)
  input_tensors = tf.keras.layers.Embedding(src_vocab_size, input_dims)(squeezed_inputs); # input_tensors.shape = (batch, ragged length, input_dims)
  # NOTE: because the tf.shape can't be used with ragged tensor, the batch is calculated this way
  batch = tf.keras.layers.Lambda(lambda x: x.nrows())(inputs); # batch.shape = ()
  # 1) encoder
  if encoder_params['enc_type'] == 'uni':
    cells = list();
    for i in range(encoder_params['layer_num']):
      # NOTE: cells over certain layer use residual structure to prevent gradient vanishing
      cells.append(Cell(encoder_params['unit_type'], encoder_params['units'], encoder_params['drop_rate'], encoder_params['forget_bias'], i >= encoder_params['layer_num'] - encoder_params['residual_layer_num']));
    dummy, hidden, cell = tf.keras.layers.RNN(cells, return_state = True)(input_tensors); # hidden.shape = (batch, units) cell.shape = (batch, units)
  elif encoder_params['enc_type'] == 'bi':
    layer_num = floor(encoder_params['layer_num'] / 2);
    residual_layer_num = floor(encoder_params['residual_layer_num'] / 2);
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
  return tf.keras.Model(inputs = inputs, outputs = (hidden, cell));

def DecoderCell(decoder_params = {'unit_type': 'lstm', 'units': 32, 'drop_rate': 0.2, 'forget_bias': 1.0, 'use_residual': True, 'residual_layer_num': 1, 'layer_num': 2}):
  
  cells = list();
  for i in range(decoder_params['layer_num']):
    # NOTE: cells over certain layer use residual structure to prevent gradient vanishing
    cells.append(Cell(decoder_params['unit_type'], decoder_params['units'], decoder_params['drop_rate'], decoder_params['forget_bias'], i >= decoder_params['layer_num'] - decoder_params['residual_layer_num']));
  # NOTE: decoder can't use bidirectional RNN, because the output length is unknown
  decoder_cell = tf.keras.layers.StackedRNNCells(cells);
  return decoder_cell;

def NMT(src_vocab_size, tgt_vocab_size, input_dims, is_train = False, 
        encoder_params = {'enc_type': 'uni', 'unit_type': 'lstm', 'units': 32, 'drop_rate': 0.2, 'forget_bias': 1.0, 'use_residual': True, 'residual_layer_num': 1, 'layer_num': 2},
        decoder_params = {'unit_type': 'lstm', 'units': 32, 'drop_rate': 0.2, 'forget_bias': 1.0, 'use_residual': True, 'residual_layer_num': 1, 'layer_num': 2},
        infer_params = {'infer_mode': 'beam_search', 'start_token': 1, 'end_token': 2, 'max_infer_len': None, 'beam_width': 2, 'length_penalty_weight': 0., 'coverage_penalty_weight': 0., 'softmax_temperature': 0.}):

  assert decoder_params['unit_type'] in ['lstm','gru','layer_norm_lstm','nas'];
  assert infer_params['infer_mode'] in ['beam_search', 'sample', 'greedy'];
  if decoder_params['use_residual'] and decoder_params['layer_num'] > 1: decoder_params['residual_layer_num'] = decoder_params['layer_num'] - 1;

  inputs = tf.keras.Input((None, 1), ragged = True); # inputs.shape = (batch, ragged length, 1)  
  # NOTE: target_embedding is used by decoder, it transform the predicted id from the hidden_{t-1} to an embedding vector
  # and feed the embedding as the input_t to decoder.
  target_embedding = tf.keras.layers.Embedding(tgt_vocab_size, input_dims);
  if is_train == True:
    targets = tf.keras.Input((None, 1)); # targets.shape = (batch, ragged length, 1)
    squeezed_targets = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis = -1))(targets); # targets.shape = (batch, ragged length)
    target_tensors = target_embedding(squeezed_targets); # target_tensors.shape = (batch, ragged length, input_dims)
  # NOTE: because the tf.shape can't be used with ragged tensor, the batch is calculated this way
  batch = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.map_fn(lambda x: 1, x, fn_output_signature = tf.TensorSpec((), dtype = tf.int32))))(inputs); # batch.shape = ()
  # 1) encoder
  hidden, cell = Encoder(src_vocab_size, input_dims, encoder_params)(inputs);
  # 2) decoder
  cells = list();
  for i in range(decoder_params['layer_num']):
    # NOTE: cells over certain layer use residual structure to prevent gradient vanishing
    cells.append(Cell(decoder_params['unit_type'], decoder_params['units'], decoder_params['drop_rate'], decoder_params['forget_bias'], i >= decoder_params['layer_num'] - decoder_params['residual_layer_num']));
  # NOTE: decoder can't use bidirectional RNN, because the output length is unknown
  decoder_cell = tf.keras.layers.StackedRNNCells(cells);
  output_layer = tf.keras.layers.Dense(tgt_vocab_size, use_bias = False);
  if is_train == True:
    sampler = tfa.seq2seq.TrainingSampler();
    decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer);
  else:
    if infer_params['infer_mode'] == 'beam_search':
      decoder = tfa.seq2seq.BeamSearchDecoder(decoder_cell, embedding_fn = target_embedding, beam_width = infer_params['beam_width'], length_penalty_weight = infer_params['length_penalty_weight'], coverage_penalty_weight = infer_params['coverage_penalty_weight']);
    else:
      if infer_params['infer_mode'] == 'sample':
        sampler = tfa.seq2seq.SampleEmbeddingSampler(embedding_fn = target_embedding, softmax_temperature = infer_params['softmax_temperature']);
      elif infer_params['infer_mode'] == 'greedy':
        sampler = tfa.seq2seq.GreedyEmbeddingSampler(embedding_fn = target_embedding);
      else:
        raise 'unknown infer mode!';
      # get maximum_iterations
      if infer_params['max_infer_len']:
        maximum_iterations = tf.keras.layers.Lambda(lambda x, l: l, arguments = {'l': infer_params['max_infer_len']})(inputs); # max_infer_length = (batch)
      else:
        maximum_iterations = tf.keras.layers.Lambda(lambda x: 2 * tf.math.reduce_max(tf.map_fn(lambda x: tf.shape(x)[0], x, fn_output_signature = tf.TensorSpec((), dtype = tf.int32))))(inputs); # max_infer_length = (batch)
      decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer, maximum_iterations = maximum_iterations);
  if is_train == True:
    # NOTE: has target_tensors for supervision at training mode
    target_length = tf.keras.layers.Lambda(lambda x: tf.map_fn(lambda x: tf.shape(x)[0], x, fn_output_signature = tf.TensorSpec((), dtype = tf.int32)))(targets); # target_length.shape = (batch)
    output, state, lengths = decoder(target_tensors, sequence_length = target_length, initial_state = (hidden, cell));
  else:
    start_tokens = tf.keras.layers.Lambda(lambda x, s: tf.ones((x,), dtype = tf.int32) * s, arguments = {'s': infer_params['start_token']})(batch);
    # NOTE: no target_tensors for supervision at inference mode
    output, state, lengths = decoder(None, start_tokens = start_tokens, end_token = infer_params['end_token'], initial_state = (hidden, cell));
  # NOTE: rnn_output.shape = (batch, ragged length, tgt_vocab_size) predicted_ids.shape = (batch, beam_width)
  # NOTE: rnn_output is the output of output_layer but output of RNN, this is specified in the document(https://tensorflow.google.cn/addons/api_docs/python/tfa/seq2seq/BasicDecoderOutput)
  return tf.keras.Model(inputs = (inputs, targets) if is_train == True else inputs, outputs = output.rnn_output if is_train == True or infer_params['infer_mode'] != 'beam_search' else output.predicted_ids);

def AttentionModel(src_vocab_size, tgt_vocab_size, input_dims, is_train = False,
                   encoder_params = {'enc_type': 'uni', 'unit_type': 'lstm', 'units': 32, 'drop_rate': 0.2, 'forget_bias': 1.0, 'use_residual': True, 'residual_layer_num': 1, 'layer_num': 2},
                   decoder_params = {'unit_type': 'lstm', 'units': 32, 'drop_rate': 0.2, 'forget_bias': 1.0, 'use_residual': True, 'residual_layer_num': 1, 'layer_num': 2},
                   attention_params = {'attention_architecture': 'standard', }):
  
  assert decoder_params['unit_type'] in ['lstm','gru','layer_norm_lstm','nas'];
  assert infer_params['infer_mode'] in ['beam_search', 'sample', 'greedy'];
  if decoder_params['use_residual'] and decoder_params['layer_num'] > 1: decoder_params['residual_layer_num'] = decoder_params['layer_num'] - 1;

  inputs = tf.keras.Input((None, 1), ragged = True); # inputs.shape = (batch, ragged length, 1)
  # NOTE: target_embedding is used by decoder, it transform the predicted id from the hidden_{t-1} to an embedding vector
  # and feed the embedding as the input_t to decoder.
  target_embedding = tf.keras.layers.Embedding(tgt_vocab_size, input_dims);
  if is_train == True:
    targets = tf.keras.Input((None, 1)); # targets.shape = (batch, ragged length, 1)
    squeezed_targets = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis = -1))(targets); # targets.shape = (batch, ragged length)
    target_tensors = target_embedding(squeezed_targets); # target_tensors.shape = (batch, ragged length, input_dims)
  # NOTE: because the tf.shape can't be used with ragged tensor, the batch is calculated this way
  batch = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.map_fn(lambda x: 1, x, fn_output_signature = tf.TensorSpec((), dtype = tf.int32))))(inputs); # batch.shape = ()
  # 1) encoder
  hidden, cell = Encoder(src_vocab_size, input_dims, encoder_params)(inputs);
  # 2) decoder
  

def GNMT(src_vocab_size, tgt_vocab_size, input_dims, is_train = False,
         encoder_params = {'enc_type': 'uni', 'unit_type': 'lstm', 'units': 32, 'drop_rate': 0.2, 'forget_bias': 1.0, 'use_residual': True, 'residual_layer_num': 1, 'layer_num': 2},
         decoder_params = {'unit_type': 'lstm', 'units': 32, 'drop_rate': 0.2, 'forget_bias': 1.0, 'use_residual': True, 'residual_layer_num': 1, 'layer_num': 2}):
  
  assert decoder_params['unit_type'] in ['lstm','gru','layer_norm_lstm','nas'];
  assert infer_params['infer_mode'] in ['beam_search', 'sample', 'greedy'];
  if decoder_params['use_residual'] and decoder_params['layer_num'] > 1: decoder_params['residual_layer_num'] = decoder_params['layer_num'] - 1;

  inputs = tf.keras.Input((None, 1), ragged = True); # inputs.shape = (batch, ragged length, 1)
  # NOTE: target_embedding is used by decoder, it transform the predicted id from the hidden_{t-1} to an embedding vector
  # and feed the embedding as the input_t to decoder.
  target_embedding = tf.keras.layers.Embedding(tgt_vocab_size, input_dims);
  if is_train == True:
    targets = tf.keras.Input((None, 1)); # targets.shape = (batch, ragged length, 1)
    squeezed_targets = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis = -1))(targets); # targets.shape = (batch, ragged length)
    target_tensors = target_embedding(squeezed_targets); # target_tensors.shape = (batch, ragged length, input_dims)
  # NOTE: because the tf.shape can't be used with ragged tensor, the batch is calculated this way
  batch = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.map_fn(lambda x: 1, x, fn_output_signature = tf.TensorSpec((), dtype = tf.int32))))(inputs); # batch.shape = ()
  # 1) encoder
  hidden, cell = Encoder(src_vocab_size, input_dims, encoder_params)(inputs);
  # 2) decoder

if __name__ == "__main__":
  
  assert tf.executing_eagerly();
  nmt = NMT(100, 200, 64, is_train = True);
  nmt.save('nmt_train.h5');
  infer_params = {'infer_mode': 'beam_search', 'start_token': 1, 'end_token': 2, 'max_infer_len': None, 'beam_width': 2, 'length_penalty_weight': 0., 'coverage_penalty_weight': 0.};
  nmt = NMT(100, 200, 64, infer_params = infer_params);
  nmt.save('nmt_infer_beamsearch.h5');
  infer_params = {'infer_mode': 'sample', 'start_token': 1, 'end_token': 2, 'max_infer_len': None, 'softmax_temperature': 0.};
  nmt = NMT(100, 200, 64, infer_params = infer_params);
  nmt.save('nmt_infer_sample.h5');
  infer_params = {'infer_mode': 'greedy', 'start_token': 1, 'end_token': 2, 'max_infer_len': None};
  nmt = NMT(100, 200, 64, infer_params = infer_params);
  nmt.save('nmt_infer_greedy.h5');
