import tensorflow as tf
from model import Model
from util import _get_dataset, show_sample
import numpy as np
import os
import matplotlib.pyplot as plt
import time

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('root_dir', 'C:/Users/Admin/Desktop/deep_learning _local_datasets/GAN/tf_records_unscaled/',
                           'Path of the tfrecords files.'
                           )
tf.app.flags.DEFINE_string('save_path', 'C:/Users/Admin/Desktop/deep_learning _local_datasets/GAN/samples/',
                           'Path for saving samples.'
                           )
tf.app.flags.DEFINE_string('summary_path', 'C:/Users/Admin/Desktop/summary/1',
                           'Path of the summary for the tensorboard.'
                           )


tf.app.flags.DEFINE_integer('num_epoch', 100,
                            'Number of training epoch.'
                            )
tf.app.flags.DEFINE_integer('batch_size', 64,
                            'Batch size of the training set.'
                            )
tf.app.flags.DEFINE_float('learning_rate_dis', 0.0002,
                          'Learning rate of the discriminator.'
                          )
tf.app.flags.DEFINE_float('learning_rate_gen', 0.0002,
                          'Learning rate of the generator.'
                          )
tf.app.flags.DEFINE_string('gen_optimizer', 'adam',
                          'Optimizer kind of the generator.'
                          )
tf.app.flags.DEFINE_string('dis_optimizer', 'adam',
                          'Optimizer kind of the discriminator.'
                          )
tf.app.flags.DEFINE_float('beta1', 0.5,
                          'Beta parameter for the Adam optimizer.'
                          )
tf.app.flags.DEFINE_string('optimizer_type', 'adam',
                           'Kind of the optimizer for the training.'
                           )


tf.app.flags.DEFINE_string('init_op', 'random_normal',
                          'Init operation for the weights,'
                          )
tf.app.flags.DEFINE_float('mean', 0.0,
                          'Mean value of gauss distribution.'
                          )
tf.app.flags.DEFINE_float('stddev', 0.02,
                          'Standard deviation of gauss distribution.'
                          )
tf.app.flags.DEFINE_float('init_weight', 0.1,
                          'Range of weights for uniform distribution'
                          )


tf.app.flags.DEFINE_integer('shuffle_buffer', 1,
                            'Buffer for the shuffeling of the data.'
                            )
tf.app.flags.DEFINE_integer('num_parallel_calls', 4,
                            'Number of parallel threads for the mapping function.'
                            )
tf.app.flags.DEFINE_integer('prefetch_buffer_train', 16,
                            'Buffer size of the prefetch for the training.'
                            )
tf.app.flags.DEFINE_integer('val_after', 50,
                            'Validate after number of iterations.'
                            )


def main(_):

    filenames=[FLAGS.root_dir+f for f in os.listdir(FLAGS.root_dir)] 
    num_batches=int(len(filenames*250)/FLAGS.batch_size)
    summary_writer=tf.summary.FileWriter(FLAGS.summary_path)
    
    with tf.Graph().as_default():

        dataset=_get_dataset(FLAGS)
                 
        iterator_dis = dataset.make_initializable_iterator()
        iterator_gen = dataset.make_initializable_iterator()
            
        images_dis= iterator_dis.get_next()
        images_gen= iterator_gen.get_next()

        z_dis=tf.random_normal(shape=(FLAGS.batch_size, 100),mean=0.0,stddev=1.0,dtype=tf.float32)
        z_gen=tf.random_normal(shape=(FLAGS.batch_size, 100),mean=0.0,stddev=1.0,dtype=tf.float32)
        z_fixed=tf.convert_to_tensor(np.random.normal(0.0,1.0,(FLAGS.batch_size, 100)),dtype=tf.float32)
            
        model=Model(FLAGS)
        
        dis_optimizer_op, loss_dis, accuracy=model._update_discriminator(images_dis, z_dis, isTrain=True)
        gen_optimizer_op, loss_gen=model._update_generator(images_gen,z_gen,isTrain=True)
        gen_sample=model._generate_sample(z_fixed, isTrain=False)
        
        merged_summ_op = tf.summary.merge_all()
            
        with tf.Session() as sess:
             
            sess.run(tf.global_variables_initializer())
            summary_writer.add_graph(sess.graph)
                
            for epoch in range(0,FLAGS.num_epoch):
                        
                sess.run(iterator_dis.initializer)
                sess.run(iterator_gen.initializer)
                    
                losses_dis=0
                losses_gen=0
                acc_dis=0
                t_compute=0
                for batch_nr in range(num_batches):
                        
                    t_1=time.time()
                    
                    _, l_, acc_, summary=sess.run((dis_optimizer_op,loss_dis,accuracy, merged_summ_op))
                    summary_writer.add_summary(summary, batch_nr)
                    losses_dis+=l_  
                    acc_dis+=acc_
                
                    _, l_=sess.run((gen_optimizer_op,loss_gen))
                    losses_gen+=l_
                    
                    t_compute+=time.time()-t_1
                                  
                    if batch_nr>0 and batch_nr%FLAGS.val_after==0:

                        print('epoch_nr: %i, batch_nr: %i/%i, dis_loss: %.3f, gen_loss: %.3f, dis_acc: %.3f, time_per_batch: %.3f s.'%(
                                epoch, batch_nr, num_batches,(losses_dis/FLAGS.val_after),
                                                             (losses_gen/FLAGS.val_after),
                                                             (acc_dis/FLAGS.val_after),
                                                             (t_compute/FLAGS.val_after)))
                        
                        losses_dis=0
                        losses_gen=0
                        acc_dis=0    
                        t_compute=0
                        
                        test_images=sess.run(gen_sample)
                        show_sample(test_images, batch_nr,FLAGS.save_path)
                    
                        


if __name__ == "__main__":
    
    tf.app.run()
