#coding = utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf
import resnet_model

import cifar10
from tensorflow.python.client import timeline

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './model_cifar10_resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
# added by faye
tf.app.flags.DEFINE_integer('max_steps', 500, """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('resnet_size', 32, """The size of the ResNet model to use.""")
tf.app.flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow "
    "server.")
tf.logging.set_verbosity(tf.logging.INFO)

INITIAL_LEARNING_RATE = 0.32       # Initial learning rate.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

updated_batch_size_num = 28
_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5
_WEIGHT_DECAY = 2e-4

def train():
    enter_time = time.time()
    global updated_batch_size_num
    worker_hosts = FLAGS.worker_hosts.split(',')
    ps_hosts = FLAGS.ps_hosts.split(',')
    print ('PS hosts are: %s' % ps_hosts)
    print ('Worker hosts are: %s' % worker_hosts)
    issync = FLAGS.issync
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == "worker":
        time.sleep(5)
        is_chief = (FLAGS.task_index == 0)
        
        if not(tf.gfile.Exists(FLAGS.train_dir)):
            tf.gfile.MakeDirs(FLAGS.train_dir)
        file = FLAGS.train_dir +"/"+ FLAGS.job_name + str(FLAGS.task_index) + "_step_b"+str(FLAGS.batch_size) + "_s"+ str(FLAGS.max_steps) + ".txt"
        loss_file = open(file, "w")
        
        
        worker_device="/job:worker/task:%d" % FLAGS.task_index
        if FLAGS.num_gpus > 0:
            gpu = (FLAGS.task_index % FLAGS.num_gpus)
            worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)            
     
        with tf.device(tf.train.replica_device_setter(
            worker_device=worker_device,
            cluster=cluster
            )):

            global_step = tf.get_variable(
                    'global_step', [],
                    initializer=tf.constant_initializer(0), trainable=False)

            decay_steps = 50000*350.0/FLAGS.batch_size
            batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')
            images, labels = cifar10.distorted_inputs()
            #re = tf.shape(images)[0]
            network = resnet_model.cifar10_resnet_v2_generator(FLAGS.resnet_size, _NUM_CLASSES)
            inputs = tf.reshape(images, [-1, _HEIGHT, _WIDTH, _DEPTH])
            #inputs = images
#            labels = tf.reshape(labels, [-1, _NUM_CLASSES])
            labels = tf.one_hot(labels, 10, 1, 0)
            logits = network(inputs, True)
            #print("logits shape:", logits.get_shape())
            cross_entropy = tf.losses.softmax_cross_entropy(
                logits=logits, 
                onehot_labels=labels)
            loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                            global_step,
                                            decay_steps,
                                            LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True)
            opt = tf.train.GradientDescentOptimizer(lr)

            # Track the moving averages of all trainable variables.
            exp_moving_averager = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
            variables_averages_op = exp_moving_averager.apply(tf.trainable_variables())

            # added by faye
            #grads = opt.compute_gradients(loss)
            grads0 = opt.compute_gradients(loss) 
            grads = [(tf.scalar_mul(tf.cast(batch_size/FLAGS.batch_size, tf.float32), grad), var) for grad, var in grads0]

            if issync:
                opt = tf.train.SyncReplicasOptimizer(
                    opt,
                    replicas_to_aggregate=len(worker_hosts),
                    total_num_replicas=len(worker_hosts),
                    variable_averages=exp_moving_averager,
                    variables_to_average=variables_to_average)
                if is_chief:
                    #chief_queue_runners = [opt.get_chief_queue_runner()]
                    chief_queue_runners = opt.get_chief_queue_runner()
                    init_tokens_op = opt.get_init_tokens_op()

            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
                
            train_op = tf.group(apply_gradient_op, variables_averages_op)
            
            sv = tf.train.Supervisor(is_chief=is_chief,
                                     logdir=FLAGS.train_dir,
                                     init_op=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()),
                                     global_step=global_step,
                                     recovery_wait_secs=1)
                                     #save_model_secs=60)


            sess_config = tf.ConfigProto(
                allow_soft_placement=True, 
                log_device_placement=FLAGS.log_device_placement)

            if is_chief:
                print("Worker %d: Initializing session..." % FLAGS.task_index)
            else:
                print("Worker %d: Waiting for session to be initialized..." % FLAGS.task_index)
            
            # Get a session.
            
            
            sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

            print("Worker %d: Session initialization complete." % FLAGS.task_index)

            # Start the queue runners.
            if is_chief and issync:
                sess.run(init_tokens_op)
                sv.start_queue_runners(sess, [chief_queue_runners])
            #else:
            #    sv.start_queue_runners(sess=sess)

            """Train CIFAR-10 for a number of steps."""

            step = 0
            g_step = 0
            train_begin = time.time()
            InitialTime = train_begin - enter_time
            print("Initial time is @ %f" % InitialTime)
            print("Training begins @ %f" % train_begin)
            tag = 1
            batch_size_num = FLAGS.batch_size
            while g_step <= FLAGS.max_steps:
                start_time = time.time()
                #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #run_metadata = tf.RunMetadata()
                #if step <= 5:
                #batch_size_num = FLAGS.batch_size
                #num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size_num
                #decay_steps_num = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
                #_, loss_value, g_step = sess.run([train_op, loss, global_step], feed_dict={batch_size: batch_size_num},  options=run_options, run_metadata=run_metadata)
                _, loss_value, g_step = sess.run([train_op, loss, global_step], feed_dict={batch_size: batch_size_num})
                   # tl = timeline.Timeline(run_metadata.step_stats)
                   # ctf = tl.generate_chrome_trace_format()
                
                fisrt_sessrun_done = time.time()
                if tag:
                    print("First sessrun time is @ %f" % (fisrt_sessrun_done - train_begin))
                    tag = 0
                    FirstSessRunTime = fisrt_sessrun_done - train_begin
                    
                if step % 10 == 0:
                        duration = time.time() - start_time
                        num_examples_per_step = batch_size_num
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)
                        format_str = ('%s:local_step %d (global_step %d), loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                        print(format_str % (datetime.now(), step, g_step, loss_value, examples_per_sec, sec_per_batch))
                        loss_file.write("%d\t%s\t%s\t%s\n" %(g_step, loss_value, examples_per_sec, sec_per_batch))
                step += 1
                
            train_end = time.time()
            loss_file.write("TrainTime\t%f\n" %(train_end-train_begin))
            loss_file.write("InitialTime\t%f\n" %InitialTime)
            loss_file.write("FirstSessRunTime\t%f\n" %FirstSessRunTime)
            loss_file.close()
            
            # end of while
            sv.stop()
            # end of with

def main(argv=None):
    #cifar10.maybe_download_and_extract()
    print("Success to ENTER!")
    #os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    train()

if __name__ == '__main__':
    tf.app.run()
