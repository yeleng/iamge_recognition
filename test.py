#测试
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import argparse
import os
import numpy as np
import tensorflow as tf
#获取模型
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "input"
  output_layer = "InceptionV3/Predictions/Reshape_1"
  #对变量进行基本定义
  parser = argparse.ArgumentParser()
  parser.add_argument("--image",default='images/')
  parser.add_argument("--graph",default='./tmp/output_graph.pb')#add
  parser.add_argument("--labels", default='./tmp/output_labels.txt')#add
  parser.add_argument("--input_height", type=int,help="input height")
  parser.add_argument("--input_width", type=int)
  parser.add_argument("--input_mean", type=int)
  parser.add_argument("--input_std", type=int)
  parser.add_argument("--input_layer",default='Placeholder')
  parser.add_argument("--output_layer",default='final_result')
  args = parser.parse_args()
  if args.graph:
    model_file = args.graph
  if args.image:
    file_dir = args.image
  for root, dirs, files in os.walk(file_dir):#获取文件名
    file_name=files
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  with tf.Session(graph=graph) as sess:
    fp = open('result.txt', 'w+')
    for files in file_name:

      t = read_tensor_from_image_file(
      file_dir+files,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)
      results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
      })
      results = np.squeeze(results)

  #这里考虑输出前几分辨情况的数据
      top_k = results.argsort()[-1:][::-1]

      labels = load_labels(label_file)
      # print('对于图片： ',files)
      # for i in top_k:
      #     print(labels[i], results[i])
      #写数据有几种不同的模式，最常用的是w’, ‘a’, 分别表示擦除原有数据再写入和将数据写到原数据之后：
      for i in top_k:
        fp.write(files+' result : '+labels[i]+'准确率为: '+str(results[i])+'\n')
    fp.close()
