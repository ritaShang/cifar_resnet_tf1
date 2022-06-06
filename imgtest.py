#coding = utf-8
# cd E:\360Downloads\Github\cifar_resnet_tf1
# conda activate tf1-cpu

import tensorflow as tf
import os

filenames = ["./cifar-100-binary/test_batch.bin"]
print("***** filenames:", filenames)
filename_queue = tf.train.string_input_producer(filenames)

print("+++++++++", type(filename_queue))

label_bytes = 2
record_bytes = 2 + 3072
  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
label = tf.cast(
      tf.slice(record_bytes, [0], [label_bytes]), tf.int32)


#with tf.Session() as sess:
#    print(sess.run(label))