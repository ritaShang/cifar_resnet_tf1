#coding = utf-8
#cd E:\360Downloads\Github\cifar_resnet_tf1
#conda activate tf1-cpu

import tensorflow as tf
import cifar10

images, labels = cifar10.distorted_inputs()
inputs = tf.reshape(images, [-1, 32, 32, 3])
resized = tf.image.resize_images(inputs, (224, 224), method=0)
#padded = tf.image.resize_image_with_crop_or_pad(images, 700, 900)

with tf.Session() as sess:
    #print("****** tf.shape(images):",sess.run(tf.shape(images)))  #[256,32,32,3]
    #print("****** inputs:",sess.run(tf.shape(inputs)))   #[256,32,32,3]
    print("****** tf.shape(resized)", sess.run(tf.shape(resized)) ) #[256,224,224,3]
    #resized = tf.cast(resized, tf.int32)
    #print("****** resized.eval() ",resized.eval())