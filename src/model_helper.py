
import tensorflow as tf


def _get_initializer(init_op,init_weight,mean, stddev):
     
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer(-init_weight, init_weight)
    elif init_op=="random_normal":
        return tf.random_normal_initializer(mean,stddev)
    elif init_op == "xavier_normal":
        return tf.keras.initializers.glorot_normal()
    elif init_op == "xavier_uniform":
        return tf.keras.initializers.glorot_uniform()
    else:
        raise ValueError("Unknown init_op %s" % init_op)
    
def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)