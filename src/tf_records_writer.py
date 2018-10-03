import os
import tensorflow as tf
import sys
from scipy.misc import imread, imsave, imresize
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR='C:/Users/Admin/Desktop/Deep Learning Local Datasets/GAN/img_celeb_resized/'

OUTPUT_DIR='C:/Users/Admin/Desktop/Deep Learning Local Datasets/GAN/tf_records_unscaled'



def _get_tf_filename(output_dir,name_tf_file,num_tf_file):
    return '%s/%s_%i.tfrecord' % (output_dir, name_tf_file, num_tf_file)


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _add_to_tf_records(dataset_dir,img_name, tf_writer):
    
    filename = dataset_dir +img_name
    
    image= imread(filename)

    example = tf.train.Example(features=tf.train.Features(feature={ 'image/encoded': bytes_feature(image.tostring())}))
    tf_writer.write(example.SerializeToString())

def scale_image(im):
  return (im / 255.0)*2 - 1


def run(dataset_dir, output_dir, name_tf_file):

    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    filenames = os.listdir(dataset_dir)
    
    i=0
    num_tf_file=0
    num_samples_per_file=250
    
    num_files=len(filenames)
    
    while i < num_files:

        tfrecords_filename=_get_tf_filename(output_dir,name_tf_file,num_tf_file)

        with tf.python_io.TFRecordWriter(tfrecords_filename) as tf_writer:
            
            j=0

            while i<num_files and j<num_samples_per_file:
                
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()
 
                filename=filenames[i]
          
                _add_to_tf_records(dataset_dir,filename, tf_writer)
                
                i+=1
                j+=1
            num_tf_file+=1
            
    print('\nFinished converting the dogs vs cats dataset!')
    
    


if __name__ == "__main__":
    
    run(DATASET_DIR,OUTPUT_DIR,'training_file')

