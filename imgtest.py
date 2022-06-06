import tensorflow as tf
import matplotlib.pyplot as plt
import cifar10


images, labels = cifar10.distorted_inputs()
 
with tf.Session() as sess:
    print("≥ı ºimage£∫",tf.shape(images))
    resized = tf.image.resize_images(images, (200, 200), method=0)
    print(sess.run(tf.shape(resized)))
    resized = tf.cast(resized, tf.int32)
    plt.imshow(resized.eval())
    plt.show()