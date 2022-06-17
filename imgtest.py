#coding = utf-8

import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  shuffle = False
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3*512,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3*512)

  return images, tf.reshape(label_batch, [batch_size])


##############################################################

sys.path.append(".")
sys.path.append("..")
filenames = ["./cifar-100-binary/test.bin"]
filename_queue = tf.train.string_input_producer(filenames, shuffle=True)

label_bytes = 2
record_bytes = 3074
reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
key, value = reader.read(filename_queue)

# Convert from a string to a vector of uint8 that is record_bytes long.
record_bytes = tf.decode_raw(value, tf.uint8)

# The first bytes represent the label, which we convert from uint8->int32.
coarse_label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)
fine_label = tf.cast(tf.slice(record_bytes, [1], [1]), tf.int32)

image = tf.reshape(tf.slice(record_bytes, [2], [3072]), [3, 32, 32])
image = tf.transpose(image, [1, 2, 0])
output = [coarse_label, fine_label, image]

reshaped_image = tf.cast(image, tf.float32)

distorted_image = tf.random_crop(reshaped_image, [32, 32, 3])
distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)
float_image = tf.image.per_image_standardization(distorted_image)

min_fraction_of_examples_in_queue = 0.4
min_queue_examples = int(50000 * 0.4)
print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

image_batch, label_batch = _generate_image_and_label_batch(float_image, fine_label, min_queue_examples, 4, shuffle=True)

sess = tf.Session()
# 在运算图中运行队列操作
tf.train.start_queue_runners(sess=sess)

res = {}
for i in range(5):
    output1, float_image1, image_batch1, label_batch1 = sess.run([output, float_image, image_batch, label_batch])

    coarse = output1[0]
    fine = output1[1]
    im = output1[2]
    print("coarse, fine, float_image:", coarse, fine)

    plt.imshow(np.array(image_batch1[3]))
    print("label_batch[4]", label_batch1[3])
    plt.show()