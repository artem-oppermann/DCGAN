import tensorflow as tf
import model_helper

class Model:
    
    def __init__(self, FLAGS):
        
        self.FLAGS=FLAGS
        self.weight_initializer=model_helper._get_initializer(FLAGS.init_op, FLAGS.init_weight, FLAGS.mean, FLAGS.stddev)

    
    def _generate(self, z, isTrain):

        with tf.variable_scope('project_into_3d'):
            W_in=tf.get_variable(name='W_project',shape=(100,512*4*4),initializer=self.weight_initializer, dtype=tf.float32)
            bias_in = tf.get_variable("bias_project", [512*4*4],initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        
            lin=tf.matmul(z, W_in)+bias_in
            shaped_lin=tf.reshape(lin,[-1, 4, 4, 512])
            activ_1= model_helper.lrelu(tf.layers.batch_normalization(shaped_lin, training=isTrain), 0.2)  

        with tf.name_scope('conv_1_layer'):
            conv1 =self._conv2d_transpose(activ_1, filters=256)
            activ_2 = model_helper.lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        with tf.name_scope('conv_2_layer'):
            conv2 =self._conv2d_transpose(activ_2, filters=128)
            activ_3 = model_helper.lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        with tf.name_scope('conv_3_layer'):
            conv3 =self._conv2d_transpose(activ_3, filters=64)
            activ_4 = model_helper.lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        with tf.name_scope('conv_4_layer'):
            conv4 =self._conv2d_transpose(activ_4, filters=3)
            
        return tf.nn.tanh(conv4)
    

    def _discriminate(self, x, isTrain):
        
        with tf.name_scope('conv_1_layer'):
            conv1=self._conv2d(x,filters=64)
            activ_1 = model_helper.lrelu(conv1, 0.2)
        
        with tf.name_scope('conv_2_layer'):
            conv2=self._conv2d(activ_1,filters=128)
            activ_2 = model_helper.lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
       
        with tf.name_scope('conv_3_layer'):
            conv3=self._conv2d(activ_2,filters=256)
            activ_3 = model_helper.lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
       
        with tf.name_scope('conv_4_layer'):
            conv4=self._conv2d(activ_3,filters=512)
            activ_4 = model_helper.lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        
        flattened=tf.contrib.layers.flatten(activ_4)
      
        with tf.variable_scope('output_layer'):
            W_out=tf.get_variable(name='W_out',shape=(8192,1),initializer=self.weight_initializer, dtype=tf.float32)
            logits=tf.matmul(flattened, W_out)  
            
        p = tf.nn.sigmoid(logits)
        
        return p, logits
    

    def _update_discriminator(self, image, z, isTrain):
        
        with tf.variable_scope('generator'):
            G_z=self._generate(z, isTrain)
        
        with tf.variable_scope('discriminator'):    
            p_real, logits_real=self._discriminate(image, isTrain)
            
        with tf.variable_scope('discriminator', reuse=True):
            p_fake, logits_fake=self._discriminate(G_z, isTrain)
            
            
        with tf.name_scope('discriminator_accuracy'):
            
            true_pos = tf.cast(p_real> 0.5, tf.float32)
            true_neg = tf.cast(p_fake < 0.5, tf.float32)
            num_predictions = 2.0*self.FLAGS.batch_size
            num_correct = tf.reduce_sum(true_pos) + tf.reduce_sum(true_neg)
            acc = num_correct / num_predictions
            tf.summary.scalar('acc_discriminator', acc)  
            
        loss=self._discriminator_loss(logits_real, logits_fake, p_real, p_fake)
        tf.summary.scalar('discriminator_loss', loss)  
        
        dis_train_parameter = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator') 
        dis_gradients = tf.gradients(loss,dis_train_parameter)
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            
            with tf.name_scope('discriminator_optimizer'):
                
                if self.FLAGS.optimizer_type == "sgd":
                    dis_optimizer = tf.train.GradientDescentOptimizer(self.FLAGS.learning_rate_dis, beta1=self.FLAGS.beta1, name='sgd_dis')                                                     
                elif self.FLAGS.optimizer_type == "adam":
                    dis_optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate_dis, beta1=self.FLAGS.beta1, name='adam_dis')                                          
                else:
                    raise ValueError('Unknown optimizer type for the discriminator: %s' %self.FLAGS.optimizer_type)
                
                dis_optimizer_op=dis_optimizer.apply_gradients(zip(dis_gradients, dis_train_parameter))
                
                return dis_optimizer_op, loss, acc
            
            
    def _update_generator(self, image, z, isTrain):
        
        with tf.variable_scope('generator', reuse=True):
            G_z=self._generate(z, isTrain)
            
        with tf.variable_scope('discriminator',reuse=True):
            p_fake, logits_fake=self._discriminate(G_z, isTrain)
        
        loss=self._generator_loss(logits_fake, p_fake)
        tf.summary.scalar('generator_loss', loss)
        
        gen_train_parameter = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        gen_gradients = tf.gradients(loss,gen_train_parameter)
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        
            with tf.name_scope('generator_optimizer'):
                if self.FLAGS.optimizer_type == "sgd":
                    gen_optimizer = tf.train.GradientDescentOptimizer(self.FLAGS.learning_rate_dis, beta1=self.FLAGS.beta1, name='sgd_dis')  
                elif self.FLAGS.optimizer_type == "adam": 
                    gen_optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate_gen, beta1=self.FLAGS.beta1, name='adam_gen')
                else:
                    raise ValueError('Unknown optimizer type for the discriminator: %s' %self.FLAGS.optimizer_type)  
                
                gen_optimizer_op=gen_optimizer.apply_gradients(zip(gen_gradients, gen_train_parameter))
                    
                return gen_optimizer_op, loss
    
        
            
    def _discriminator_loss(self, logits_real, logits_fake, p_real, p_fake):
        
        with tf.name_scope('discriminator_loss'):
            d_loss_real=tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real,
                                                                labels=tf.ones_like(p_real))
        
            d_loss_gen=tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,
                                                               labels=tf.zeros_like(p_fake))   
        
            d_loss = tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_gen)
        
        return d_loss
        
    
    def _generator_loss(self, logits_fake, p_fake):
        
        with tf.name_scope('generator_loss'):
            g_cost= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,
                                                                           labels=tf.ones_like(p_fake)))
        return g_cost
    
    
    def _conv2d_transpose(self, x, filters, kernel_size=[4, 4], strides=(2, 2)):
          
        return tf.layers.conv2d_transpose(x, filters, 
                                          kernel_size, 
                                          strides, 
                                          kernel_initializer=self.weight_initializer,
                                          padding='same'
                                          )
    
    def _conv2d(self, x, filters, kernel_size=[4, 4], strides=(2, 2)):
        
        return tf.layers.conv2d(x, filters, kernel_size, strides, kernel_initializer=self.weight_initializer,padding='same')

    def _generate_sample(self, z, isTrain):
        
        with tf.variable_scope('generator', reuse=True):
            G_z=self._generate(z, isTrain)
        
        return G_z
    
    
    
    
    
    
    
    
    
    