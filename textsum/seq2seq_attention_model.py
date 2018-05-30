# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-Sequence with attention model for text summarization.
"""
from collections import namedtuple

import numpy as np
import seq2seq_lib
from six.moves import xrange
import tensorflow as tf

HParams = namedtuple('HParams',
                     'mode, min_lr, lr, batch_size, '
                     'enc_layers, enc_timesteps, dec_timesteps, '
                     'min_input_len, num_hidden, emb_dim, max_grad_norm, '
                     'num_softmax_samples')

##embedding:[vsize,emb_dim],output_projection[0]:[num_hidden,vsize],output_projection[1]:[vsize]
def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  def loop_function(prev, _):
    """function that feed previous model output rather than ground truth."""
    if output_projection is not None:
      prev = tf.nn.xw_plus_b(
          prev, output_projection[0], output_projection[1])
    prev_symbol = tf.argmax(prev, 1) #找出对应的词的位置
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = tf.stop_gradient(emb_prev) #停止梯度计算，当在graph中执行时，这个op输出和输入的tensor是一样的，加入这个op，这些emb_prev对与gradients的计算不会有任何的帮助
    return emb_prev
  return loop_function


class Seq2SeqAttentionModel(object):
  """Wrapper for Tensorflow model graph for text sum vectors."""

  def __init__(self, hps, vocab, num_gpus=0):
    self._hps = hps
    self._vocab = vocab
    self._num_gpus = num_gpus
    self._cur_gpu = 0

  def run_train_step(self, sess, article_batch, abstract_batch, targets,
                     article_lens, abstract_lens, loss_weights):
    to_return = [self._train_op, self._summaries, self._loss, self.global_step]
    return sess.run(to_return,
                    feed_dict={self._articles: article_batch,
                               self._abstracts: abstract_batch,
                               self._targets: targets,
                               self._article_lens: article_lens,
                               self._abstract_lens: abstract_lens,
                               self._loss_weights: loss_weights})

  def run_eval_step(self, sess, article_batch, abstract_batch, targets,
                    article_lens, abstract_lens, loss_weights):
    to_return = [self._summaries, self._loss, self.global_step]
    return sess.run(to_return,
                    feed_dict={self._articles: article_batch,
                               self._abstracts: abstract_batch,
                               self._targets: targets,
                               self._article_lens: article_lens,
                               self._abstract_lens: abstract_lens,
                               self._loss_weights: loss_weights})

  def run_decode_step(self, sess, article_batch, abstract_batch, targets,
                      article_lens, abstract_lens, loss_weights):
    to_return = [self._outputs, self.global_step]
    return sess.run(to_return,
                    feed_dict={self._articles: article_batch,
                               self._abstracts: abstract_batch,
                               self._targets: targets,
                               self._article_lens: article_lens,
                               self._abstract_lens: abstract_lens,
                               self._loss_weights: loss_weights})

  def _next_device(self):
    """Round robin the gpu device. (Reserve last gpu for expensive op)."""
    if self._num_gpus == 0:
      return ''
    dev = '/gpu:%d' % self._cur_gpu
    if self._num_gpus > 1:
      self._cur_gpu = (self._cur_gpu + 1) % (self._num_gpus-1)
    return dev

  def _get_gpu(self, gpu_id):
    if self._num_gpus <= 0 or gpu_id >= self._num_gpus:
      return ''
    return '/gpu:%d' % gpu_id

  def _add_placeholders(self):
    """Inputs to be fed to the graph."""
    hps = self._hps
    self._articles = tf.placeholder(tf.int32,
                                    [hps.batch_size, hps.enc_timesteps],
                                    name='articles')
    self._abstracts = tf.placeholder(tf.int32,
                                     [hps.batch_size, hps.dec_timesteps],
                                     name='abstracts')
    self._targets = tf.placeholder(tf.int32,
                                   [hps.batch_size, hps.dec_timesteps],
                                   name='targets')
    self._article_lens = tf.placeholder(tf.int32, [hps.batch_size],
                                        name='article_lens')
    self._abstract_lens = tf.placeholder(tf.int32, [hps.batch_size],
                                         name='abstract_lens')
    self._loss_weights = tf.placeholder(tf.float32,
                                        [hps.batch_size, hps.dec_timesteps],
                                        name='loss_weights')

  def _add_seq2seq(self):
    hps = self._hps
    vsize = self._vocab.NumIds()

    with tf.variable_scope('seq2seq'):
      #tf.unstack(value, num=None, axis=0, name='unstack')
      #将self._articles转置，然后再按照维度0解开，返回的是list包含所有的解开的值
      # 这里axis=0是dec_timesteps的维度，unstack的化，list的顺序就是每一个timestep的顺序，长度是batch_size的长度[[batch_size],[batch_size]]，也就是[timestep,batch_size]
      encoder_inputs = tf.unstack(tf.transpose(self._articles))
      decoder_inputs = tf.unstack(tf.transpose(self._abstracts))
      targets = tf.unstack(tf.transpose(self._targets))
      loss_weights = tf.unstack(tf.transpose(self._loss_weights))
      article_lens = self._article_lens

      # Embedding shared by the input and outputs.
      with tf.variable_scope('embedding'), tf.device('/cpu:0'):
        embedding = tf.get_variable(
            'embedding', [vsize, hps.emb_dim], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
        emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, x)
                              for x in encoder_inputs] # list的大小是[timesteps,batch_size,emb_dim]
        emb_decoder_inputs = [tf.nn.embedding_lookup(embedding, x)
                              for x in decoder_inputs]

      for layer_i in xrange(hps.enc_layers): #hps.enc_layers:这里设置了4层,所以RNN的架构中是用4层双向的RNN
      # 多层双向的RNN的实现
        with tf.variable_scope('encoder%d'%layer_i), tf.device(
            self._next_device()):
          # cell_fw:是RNNCell实例，用来前向传播
          cell_fw = tf.contrib.rnn.LSTMCell(
              hps.num_hidden,
              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
              state_is_tuple=False) # state_is_tuple:True，返回（c_state,m_state),False:这两个沿着column axis拼接
          # cell_bw:是RNNcell示例，用来后向传播的方向
          cell_bw = tf.contrib.rnn.LSTMCell(
              hps.num_hidden,
              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
              state_is_tuple=False)
          # tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs, initial_state_fw=None, initial_state_bw=None, dtype=None, sequence_length=None, scope=None)
          # inputs: 长度为T的list，每一个tenosr的shape是[batch_size, input_size]
          # sequence_length,大小是[batch_size],包含每个sequence的真实的长度，所以这里是每一个article的真实的长度
          # 这个函数用来创建双向的递归神经网络
          # 函数返回为（outputs,output_state_fw,output_state_bw), outputs是T个长度的list，每一个output针对每一个input的输出，将前向和后向的输出depth_concatenated，output_state_fw: forward rnn的final state，同理output_state_bw也是最后一个state，每一个tensor是[c_state,m_state]
          (emb_encoder_inputs, fw_state, _) = tf.contrib.rnn.static_bidirectional_rnn(
              cell_fw, cell_bw, emb_encoder_inputs, dtype=tf.float32,
              sequence_length=article_lens)
      encoder_outputs = emb_encoder_inputs

      with tf.variable_scope('output_projection'):
        w = tf.get_variable(
            'w', [hps.num_hidden, vsize], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
        w_t = tf.transpose(w)
        v = tf.get_variable(
            'v', [vsize], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))

      with tf.variable_scope('decoder'), tf.device(self._next_device()):
        # When decoding, use model output from the previous step
        # for the next step.
        loop_function = None
        if hps.mode == 'decode':
          loop_function = _extract_argmax_and_embed(#embedding:[vsize,emb_dim],w:[num_hidden,vsize],v:[vsize]
              embedding, (w, v), update_embedding=False)

        cell = tf.contrib.rnn.LSTMCell(
            hps.num_hidden,
            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
            state_is_tuple=False)

        encoder_outputs = [tf.reshape(x, [hps.batch_size, 1, 2*hps.num_hidden])
                           for x in encoder_outputs] #[timesteps,batchsize,1,2*num_hiddens]
        self._enc_top_states = tf.concat(axis=1, values=encoder_outputs) # 自身concat维度需要大于3,tf.concat和np.concatenate()使用方法基本一致。_enc_top_states最后输出的为[batchsize,timesteps,2*num_hiddens] 
        
        self._dec_in_state = fw_state #这边输出的结构是，每一个cell输出是两个状态[batch_size, 2*num_hiddens]
        # During decoding, follow up _dec_in_state are fed from beam_search.
        # dec_out_state are stored by beam_search for next step feeding.
        initial_state_attention = (hps.mode == 'decode') #这里在decode的过程中从inital state和attention states初始化attentions，否则初始化为0
        
        # API里面说这个被弃用了。。。但是没有说使用什么来代替，蜜汁尴尬
        # 用来创建sequence-to-sequence models的
        # tf.contib.legacy_seq2seq.attention_decoder()是decoder with attention
        # 参数 decoder_inputs:list of [batch_size,input_size],所以tensor的大小是  [timesteps,batch_size,emb_dim]
        # initial_state:[batch_size,cell.state_size]
        # attention_states:[batch_size, attn_length,attn,size]
        # cell:tf.nn.rnn_cell.RNNCell
        # output_size:[output_size](可以为None)
        # num_heads:number of attention_size
        # loop_function:这个function是为了利用ith output产生i+1-th input，并且decoder_inputs会被忽略,除了第一个元素。可以被用来decoding，也可以用来模拟 loop_function(pre,i)=next, pre:[batch_size,output_size],i:integer,next:[batch_size,input_size]
        # 所以在这里，第ith个step的输入是pre，然后loop_finction的输出作为下一个step的输入，这里实际上就是预测的过程。
        decoder_outputs, self._dec_out_state = tf.contrib.legacy_seq2seq.attention_decoder(
            emb_decoder_inputs, self._dec_in_state, self._enc_top_states,
            cell, num_heads=1, loop_function=loop_function,
            initial_state_attention=initial_state_attention)
        # 上面的输出是（outputs,state)
        # outputs是一个和decoder_inputs长度一样的list，每个元素是一个2Dtensor[batch_size,output_size]，这里每一次都有一个attention的计算
        # state: 最后一个time_step的每一个decoder的state,[batch_size,cell.state_size]     

      with tf.variable_scope('output'), tf.device(self._next_device()):
        model_outputs = []
        for i in xrange(len(decoder_outputs)):
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          model_outputs.append(
              tf.nn.xw_plus_b(decoder_outputs[i], w, v)) # 输出的列都是vsize的，对应vocab中的词，所以model_output的list的每个维度是[timesteps,batch_size,v_size]

      if hps.mode == 'decode':
        with tf.variable_scope('decode_output'), tf.device('/cpu:0'):
          best_outputs = [tf.argmax(x, 1) for x in model_outputs]
          tf.logging.info('best_outputs%s', best_outputs[0].get_shape())
          self._outputs = tf.concat(
              axis=1, values=[tf.reshape(x, [hps.batch_size, 1]) for x in best_outputs]) #将[time_steps,batch_size,1]按照axis=1合并为[batch_size,time_steps]

	      #tf.nn.top_k(input,k=1):在输入的最后一维找到k个最大的entries的值和indices，返回values和indices
          self._topk_log_probs, self._topk_ids = tf.nn.top_k(
              tf.log(tf.nn.softmax(model_outputs[-1])), hps.batch_size*2) # model_outputs[-1]最后一个timestep的输出[batch_size,v_size]

      with tf.variable_scope('loss'), tf.device(self._next_device()):
        def sampled_loss_func(inputs, labels):
          with tf.device('/cpu:0'):  # Try gpu.
            labels = tf.reshape(labels, [-1, 1])
            # 通过采样快速估计loss，计算的交叉熵的loss
            return tf.nn.sampled_softmax_loss(
                weights=w_t, biases=v, labels=labels, inputs=inputs,
                num_sampled=hps.num_softmax_samples, num_classes=vsize)

        # 这里可以采用采样的办法来估计loss，在train的时候快速训练
        # 采用的loss是交叉熵，通过权重的均值做估计
        if hps.num_softmax_samples != 0 and hps.mode == 'train':
          self._loss = seq2seq_lib.sampled_sequence_loss(
              decoder_outputs, targets, loss_weights, sampled_loss_func)
        else:
          # weighted_cross_entropy with logits
          self._loss = tf.contrib.legacy_seq2seq.sequence_loss(
              model_outputs, targets, loss_weights)
        # 存储loss的值，最大值只存到12，？
        tf.summary.scalar('loss', tf.minimum(12.0, self._loss))

  def _add_train_op(self):
    """Sets self._train_op, op to run for training."""
    hps = self._hps

    self._lr_rate = tf.maximum(
        hps.min_lr,  # min_lr_rate.
        tf.train.exponential_decay(hps.lr, self.global_step, 30000, 0.98))

    tvars = tf.trainable_variables()
    with tf.device(self._get_gpu(self._num_gpus-1)):
      grads, global_norm = tf.clip_by_global_norm(
          tf.gradients(self._loss, tvars), hps.max_grad_norm)
    tf.summary.scalar('global_norm', global_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr_rate)
    tf.summary.scalar('learning rate', self._lr_rate)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=self.global_step, name='train_step')

  def encode_top_state(self, sess, enc_inputs, enc_len):
    """Return the top states from encoder for decoder.

    Args:
      sess: tensorflow session.
      enc_inputs: encoder inputs of shape [batch_size, enc_timesteps].
      enc_len: encoder input length of shape [batch_size]
    Returns:
      enc_top_states: The top level encoder states.
      dec_in_state: The decoder layer initial state.
    """
    results = sess.run([self._enc_top_states, self._dec_in_state],
                       feed_dict={self._articles: enc_inputs,
                                  self._article_lens: enc_len})
    # print(results[0])
    # print(results[1])
    # print(results[0].shape) # (8,120,512)
    # print(results[1][0].shape) # (8,512)
    return results[0], results[1][0]

  def decode_topk(self, sess, latest_tokens, enc_top_states, dec_init_states):
    """Return the topK results and new decoder states."""
    feed = {
        self._enc_top_states: enc_top_states,
        self._dec_in_state:
            np.squeeze(np.array(dec_init_states)),
        self._abstracts:
            np.transpose(np.array([latest_tokens])),
        self._abstract_lens: np.ones([len(dec_init_states)], np.int32)}

    results = sess.run(
        [self._topk_ids, self._topk_log_probs, self._dec_out_state],
        feed_dict=feed)

    ids, probs, states = results[0], results[1], results[2]
    new_states = [s for s in states]
    return ids, probs, new_states

  def build_graph(self):
    self._add_placeholders()
    self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
