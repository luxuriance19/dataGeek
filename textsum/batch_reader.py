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

"""Batch reader to seq2seq attention model, with bucketing support."""

from collections import namedtuple
from random import shuffle
from threading import Thread #虽然python每个时间只会有一个线程执行，但是因为多线程在一个线程进行I/O读写时会执行另外一个线程的GPU执行，所以，如果做网络交互，涉及I/O，那么适合多线程运算
import time

import numpy as np
import six
from six.moves import queue as Queue
from six.moves import xrange
import tensorflow as tf

import data

ModelInput = namedtuple('ModelInput',
                        'enc_input dec_input target enc_len dec_len '
                        'origin_article origin_abstract')

BUCKET_CACHE_BATCH = 100
QUEUE_NUM_BATCH = 100


class Batcher(object):
  """Batch reader with shuffling and bucketing support."""

  def __init__(self, data_path, vocab, hps,
               article_key, abstract_key, max_article_sentences,
               max_abstract_sentences, bucketing=True, truncate_input=False):
    """Batcher constructor.

    Args:
      data_path: tf.Example filepattern.
      vocab: Vocabulary.
      hps: Seq2SeqAttention model hyperparameters.
      article_key: article feature key in tf.Example.
      abstract_key: abstract feature key in tf.Example.
      max_article_sentences: Max number of sentences used from article.
      max_abstract_sentences: Max number of sentences used from abstract.
      bucketing: Whether bucket articles of similar length into the same batch.
      truncate_input: Whether to truncate input that is too long. Alternative is
        to discard such examples.
    """
    self._data_path = data_path
    self._vocab = vocab
    self._hps = hps
    self._article_key = article_key
    self._abstract_key = abstract_key
    self._max_article_sentences = max_article_sentences
    self._max_abstract_sentences = max_abstract_sentences
    self._bucketing = bucketing #决定是否bucket相似长度的articles在一个batch当中，执行的时候，就是将输入按照en_len进行排序，这样进入batch的就是在这个排序部分的数据
    self._truncate_input = truncate_input
    self._input_queue = Queue.Queue(QUEUE_NUM_BATCH * self._hps.batch_size) #Queue(maxsize=)
    self._bucket_input_queue = Queue.Queue(QUEUE_NUM_BATCH)
    self._input_threads = []
    # 以下部分开启的线程都是守护线程，在主线程执行完毕之后才会结束。
    for _ in xrange(16):
      self._input_threads.append(Thread(target=self._FillInputQueue))
      self._input_threads[-1].daemon = True #守护线程，必须要在start前开启
      self._input_threads[-1].start()
    self._bucketing_threads = []
    for _ in xrange(4):
      self._bucketing_threads.append(Thread(target=self._FillBucketInputQueue))
      self._bucketing_threads[-1].daemon = True
      self._bucketing_threads[-1].start()
    # 只有一个看守线程,这个看守线程查看input_thread和bucket_thread是否在正常运行，如果thread dead那么重新启动
    self._watch_thread = Thread(target=self._WatchThreads)
    self._watch_thread.daemon = True
    self._watch_thread.start()

  def NextBatch(self):
    """Returns a batch of inputs for seq2seq attention model.

    Returns:
      enc_batch: A batch of encoder inputs [batch_size, hps.enc_timestamps].
      dec_batch: A batch of decoder inputs [batch_size, hps.dec_timestamps].
      target_batch: A batch of targets [batch_size, hps.dec_timestamps].
      enc_input_len: encoder input lengths of the batch.
      dec_input_len: decoder input lengths of the batch.
      loss_weights: weights for loss function, 1 if not padded, 0 if padded.
      origin_articles: original article words.
      origin_abstracts: original abstract words.
    """
    enc_batch = np.zeros(
        (self._hps.batch_size, self._hps.enc_timesteps), dtype=np.int32)
    enc_input_lens = np.zeros(
        (self._hps.batch_size), dtype=np.int32)
    dec_batch = np.zeros(
        (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.int32)
    dec_output_lens = np.zeros(
        (self._hps.batch_size), dtype=np.int32)
    target_batch = np.zeros(
        (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.int32)
    loss_weights = np.zeros(
        (self._hps.batch_size, self._hps.dec_timesteps), dtype=np.float32)
    origin_articles = ['None'] * self._hps.batch_size # article由字符串组成
    origin_abstracts = ['None'] * self._hps.batch_size

    buckets = self._bucket_input_queue.get()
    for i in xrange(self._hps.batch_size):
      (enc_inputs, dec_inputs, targets, enc_input_len, dec_output_len,
       article, abstract) = buckets[i]

      origin_articles[i] = article
      origin_abstracts[i] = abstract
      enc_input_lens[i] = enc_input_len
      dec_output_lens[i] = dec_output_len
      enc_batch[i, :] = enc_inputs[:]
      dec_batch[i, :] = dec_inputs[:]
      target_batch[i, :] = targets[:]
      for j in xrange(dec_output_len):
        loss_weights[i][j] = 1
    return (enc_batch, dec_batch, target_batch, enc_input_lens, dec_output_lens,
            loss_weights, origin_articles, origin_abstracts)

  def _FillInputQueue(self):
    """Fill input queue with ModelInput."""
    start_id = self._vocab.WordToId(data.SENTENCE_START)
    end_id = self._vocab.WordToId(data.SENTENCE_END)
    pad_id = self._vocab.WordToId(data.PAD_TOKEN)
    # input_gen 返回的是一个(article_text,abstract_text)的生成器
    input_gen = self._TextGenerator(data.ExampleGen(self._data_path)) #这里调用了tensorflow.core.example.example_pb2
    while True:
      (article, abstract) = six.next(input_gen)
      article_sentences = [sent.strip() for sent in
                           data.ToSentences(article, include_token=False)] #这里将文章一句一句分开
      abstract_sentences = [sent.strip() for sent in
                            data.ToSentences(abstract, include_token=False)]
                            
      #print('article',article,'article_sent',article_sentences)
      #print('abstract',abstract,'abstract_sent',abstract_sentences)

      enc_inputs = []
      # Use the <s> as the <GO> symbol for decoder inputs.
      dec_inputs = [start_id]

      # Convert first N sentences to word IDs, stripping existing <s> and </s>.
      # 这里enc_inputs和dec_inputs对应的都是word的id编号
      for i in xrange(min(self._max_article_sentences,
                          len(article_sentences))):
        enc_inputs += data.GetWordIds(article_sentences[i], self._vocab)
      for i in xrange(min(self._max_abstract_sentences,
                          len(abstract_sentences))):
        dec_inputs += data.GetWordIds(abstract_sentences[i], self._vocab)
      
      #print('enc',enc_inputs)
      #print('dec',dec_inputs)

      # Filter out too-short input
      if (len(enc_inputs) < self._hps.min_input_len or
          len(dec_inputs) < self._hps.min_input_len):
        tf.logging.warning('Drop an example - too short.\nenc:%d\ndec:%d',
                           len(enc_inputs), len(dec_inputs))
        continue

      # If we're not truncating input, throw out too-long input
      if not self._truncate_input:
        if (len(enc_inputs) > self._hps.enc_timesteps or
            len(dec_inputs) > self._hps.dec_timesteps):
          tf.logging.warning('Drop an example - too long.\nenc:%d\ndec:%d',
                             len(enc_inputs), len(dec_inputs))
          continue
      # If we are truncating input, do so if necessary
      else:
        if len(enc_inputs) > self._hps.enc_timesteps:
          enc_inputs = enc_inputs[:self._hps.enc_timesteps]
        if len(dec_inputs) > self._hps.dec_timesteps:
          dec_inputs = dec_inputs[:self._hps.dec_timesteps]

      # targets is dec_inputs without <s> at beginning, plus </s> at end
      # targets和dec_inputs的不同点仅仅在与dec_inputs有开头标记没有结尾标记，targets有结尾标记没有开头标记
      targets = dec_inputs[1:]
      targets.append(end_id)

      # Now len(enc_inputs) should be <= enc_timesteps, and
      # len(targets) = len(dec_inputs) should be <= dec_timesteps

      enc_input_len = len(enc_inputs)
      dec_output_len = len(targets)
      
      # print('article',article,'article_sent',article_sentences,'enc',enc_inputs,"article_lensent",len(article_sentences),"enc_len",enc_input_len,"\n",'abstract',abstract,'abstract_sent',abstract_sentences,'dec',targets,"dec_len",dec_output_len)
      #print('abstract',abstract,'abstract_sent',abstract_sentences,'dec',targets,"dec_len",dec_output_len)

      # Pad if necessary
      # 这里输入和输出的结尾的padding值是不一样的
      while len(enc_inputs) < self._hps.enc_timesteps:
        enc_inputs.append(pad_id)
      while len(dec_inputs) < self._hps.dec_timesteps:
        dec_inputs.append(end_id)
      while len(targets) < self._hps.dec_timesteps:
        targets.append(end_id)

      # 这里继承了nametuple类
      element = ModelInput(enc_inputs, dec_inputs, targets, enc_input_len,
                           dec_output_len, ' '.join(article_sentences),
                           ' '.join(abstract_sentences))
      self._input_queue.put(element)

  def _FillBucketInputQueue(self):
    """Fill bucketed batches into the bucket_input_queue."""
    while True:
      inputs = []
      for _ in xrange(self._hps.batch_size * BUCKET_CACHE_BATCH):
        inputs.append(self._input_queue.get())
      if self._bucketing:
        inputs = sorted(inputs, key=lambda inp: inp.enc_len)

      batches = []
      for i in xrange(0, len(inputs), self._hps.batch_size):# xrange(start, stop, step)
        batches.append(inputs[i:i+self._hps.batch_size])
      shuffle(batches)
      for b in batches:
        self._bucket_input_queue.put(b)

  def _WatchThreads(self):
    """Watch the daemon input threads and restart if dead."""
    while True:
      time.sleep(60)
      input_threads = []
      for t in self._input_threads:
        if t.is_alive():
          input_threads.append(t)
        else:
          tf.logging.error('Found input thread dead.')
          new_t = Thread(target=self._FillInputQueue)
          input_threads.append(new_t)
          input_threads[-1].daemon = True
          input_threads[-1].start()
      self._input_threads = input_threads

      bucketing_threads = []
      for t in self._bucketing_threads:
        if t.is_alive():
          bucketing_threads.append(t)
        else:
          tf.logging.error('Found bucketing thread dead.')
          new_t = Thread(target=self._FillBucketInputQueue)
          bucketing_threads.append(new_t)
          bucketing_threads[-1].daemon = True
          bucketing_threads[-1].start()
      self._bucketing_threads = bucketing_threads

  def _TextGenerator(self, example_gen):
    """Generates article and abstract text from tf.Example."""
    while True:
      e = six.next(example_gen)
      try:
        article_text = self._GetExFeatureText(e, self._article_key)
        abstract_text = self._GetExFeatureText(e, self._abstract_key)
      except ValueError:
        tf.logging.error('Failed to get article or abstract from example')
        continue

      yield (article_text, abstract_text)

  def _GetExFeatureText(self, ex, key):
    """Extract text for a feature from td.Example.

    Args:
      ex: tf.Example.
      key: key of the feature to be extracted.
    Returns:
      feature: a feature text extracted.
    """
    # return ex.features.feature[key].bytes_list.value[0]
    return ex.features.feature[key].bytes_list.value[0].decode('utf-8')
