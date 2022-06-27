import tensorflow as tf

x = tf.Variable(1.0, tf.float32)
op = tf.add(x, 1)

update = tf.assign(x, op)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3):
        img = sess.run(update)
        print(img)