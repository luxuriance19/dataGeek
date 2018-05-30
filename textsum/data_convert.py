"""Example of Converting TextSum model data.
Usage:
python data_convert_example.py --command binary_to_text --in_file data/data --out_file data/text_data
python data_convert_example.py --command text_to_binary --in_file data/text_data --out_file data/binary_data
python data_convert_example.py --command binary_to_text --in_file data/binary_data --out_file data/text_data2
diff data/text_data2 data/text_data
"""

import struct
import sys
import argparse

import tensorflow as tf
from tensorflow.core.example import example_pb2

parser = argparse.ArgumentParser(description="data_convert")
parser.add_argument("--command", default="", help="Either binary_to_text or text_to_binary.")
parser.add_argument("--in_file",default="",help="path to in file")
parser.add_argument("--out_file",default="",help="path to out file")
args = parser.parse_args()

def _binary_to_text():
  reader = open(args.in_file, 'rb')
  writer = open(args.out_file, 'w')
  while True:
    len_bytes = reader.read(8)
    if not len_bytes:
      sys.stderr.write('Done reading\n')
      return
    str_len = struct.unpack('q', len_bytes)[0]
    tf_example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
    tf_example = example_pb2.Example.FromString(tf_example_str)
    examples = []
    for key in tf_example.features.feature:
      examples.append('%s=%s' % (key, tf_example.features.feature[key].bytes_list.value[0]))
    writer.write('%s\n' % '\t'.join(examples))
  reader.close()
  writer.close()


def _text_to_binary():
  inputs = open(args.in_file, 'r',encoding='utf-8').readlines()
  writer = open(args.out_file, 'wb')
  for inp in inputs:
    tf_example = example_pb2.Example()
    for feature in inp.strip().split('\t'):
      # print(inp)
      (k, v) = feature.split('=')
      k = str.encode(k)
      v = str.encode(v)
      # print(k)
      # print(v)
      tf_example.features.feature[k].bytes_list.value.extend([v])
    tf_example_str = tf_example.SerializeToString()
    # print(tf_example_str)
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))
  writer.close()


def main():
  assert args.command and args.in_file and args.out_file
  if args.command == 'binary_to_text':
    _binary_to_text()
  elif args.command == 'text_to_binary':
    _text_to_binary()


if __name__ == '__main__':
  main()
