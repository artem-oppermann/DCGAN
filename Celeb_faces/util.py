import tensorflow as tf
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from glob import glob
from scipy.misc import imread, imsave, imresize
from PIL import Image


def _get_dataset(FLAGS):
    
    filenames=[FLAGS.root_dir+f for f in os.listdir(FLAGS.root_dir)]

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse,num_parallel_calls=FLAGS.num_parallel_calls)
    dataset = dataset.repeat()
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_train)
    
    return dataset
    


def parse(serialized):
   
    features={'image/encoded': tf.FixedLenFeature([], tf.string)}

    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features
                                             )
    image_raw  = parsed_example['image/encoded']

    image = tf.decode_raw(image_raw, tf.uint8)
    image = tf.cast(image, tf.float32)
    
    image=tf.reshape(image,(64,64,3))
    image=tf.subtract(tf.multiply(tf.divide(image,255.0), 2.0),1)

    return image


def show_sample(test_images,num_iter, path = 'result.png'):
      
        d=64
        flat_image = np.empty((8*64, 8*64, 3))
        k = 0
        
        for i in range(8):
            for j in range(8):
                flat_image[i*d:(i+1)*d, j*d:(j+1)*d] = test_images[k]
                k += 1
        sp.misc.imsave(path+'/samples_at_iter_%i.png' % num_iter,flat_image)


