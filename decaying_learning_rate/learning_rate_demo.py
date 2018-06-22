# -*- coding: utf-8 -*-
"""Functional test for learning rate decay."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import math


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', 'log/', 'Summaries log directory')


if __name__ == '__main__':
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    
    tf.gfile.MakeDirs(FLAGS.log_dir)
  
    global_step = tf.Variable(0, trainable=False)
    increment_op = tf.assign_add(global_step, tf.constant(1))
    
    
    
    '''
    decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    '''
    starter_learning_rate = 0.1
    decay_steps=100
    lr = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate=0.96, staircase=True)
    tf.summary.scalar('exponential_decay', lr)   
    
    lr = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate=0.9, staircase=False)    
    tf.summary.scalar('exponential_decay', lr)   
    
    

    '''
    use a learning rate that's 1.0 for the first 100001 steps, 0.5 for the next 10000 steps, and 0.1 for any additional steps.
    '''
    boundaries = [5000, 8000]
    values = [0.1, 0.01, 0.001]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    tf.summary.scalar('piecewise_constant', learning_rate)   
    
    
    '''
     global_step = min(global_step, decay_steps)
     decayed_learning_rate = (learning_rate - end_learning_rate) *
                          (1 - global_step / decay_steps) ^ (power) +
                          end_learning_rate
    '''
    starter_learning_rate = 0.1
    end_learning_rate = 0.001
    decay_steps = 10000
    learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,decay_steps, end_learning_rate,power=0.5)
    tf.summary.scalar('polynomial_decay', learning_rate)   
    
    
    '''
    decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
    '''
    starter_learning_rate = 0.1
    learning_rate = tf.train.natural_exp_decay(starter_learning_rate, global_step, decay_steps=1000, decay_rate=0.96, staircase=False)
    tf.summary.scalar('natural_exp_decay', learning_rate)   
    
    
    '''
      decayed_learning_rate = learning_rate / (1 + decay_rate * global_step /decay_step)
      
      decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step /decay_step))
    '''
    starter_learning_rate = 0.1
    decay_steps = 1.0
    decay_rate = 0.5
    learning_rate = tf.train.inverse_time_decay(starter_learning_rate, global_step,decay_steps, decay_rate)
    tf.summary.scalar('inverse_time_decay', learning_rate)   
    
    
    '''
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    decayed_learning_rate = learning_rate * decayed
    '''
    def np_cosine_decay(self, step, decay_steps, alpha=0.0):
        step = min(step, decay_steps)
        completed_fraction = step / decay_steps
        decay = 0.5 * (1.0 + math.cos(math.pi * completed_fraction))
        return (1.0 - alpha) * decay + alpha

    starter_learning_rate = 0.1
    decay_steps = 10000
    lr_decayed = tf.train.cosine_decay(starter_learning_rate, global_step, decay_steps)
    tf.summary.scalar('cosine_decay', lr_decayed)   
    
    
    def np_cosine_decay_restarts(self, step, decay_steps, t_mul=2.0, m_mul=1.0,
                               alpha=0.0):
        fac = 1.0
        while step >= decay_steps:
          step = step - decay_steps
          decay_steps *= t_mul
          fac *= m_mul
    
        completed_fraction = step / decay_steps
        decay = fac * 0.5 * (1.0 + math.cos(math.pi * completed_fraction))
        return (1.0 - alpha) * decay + alpha
    
    first_decay_steps = 1000
    lr_decayed = tf.train.cosine_decay_restarts(learning_rate, global_step,
                                     first_decay_steps)
    tf.summary.scalar('cosine_decay_restarts', lr_decayed)  
    
    
    
    
    '''
    global_step = min(global_step, decay_steps)
    linear_decay = (decay_steps - global_step) / decay_steps)
    cosine_decay = 0.5 * (
      1 + cos(pi * 2 * num_periods * global_step / decay_steps))
    decayed = (alpha + linear_decay) * cosine_decay + beta
    decayed_learning_rate = learning_rate * decayed
    '''
    def np_linear_cosine_decay(self,
                             step,
                             decay_steps,
                             alpha=0.0,
                             beta=0.001,
                             num_periods=0.5):
        step = min(step, decay_steps)
        linear_decayed = float(decay_steps - step) / decay_steps
        fraction = 2.0 * num_periods * step / float(decay_steps)
        cosine_decayed = 0.5 * (1.0 + math.cos(math.pi * fraction))
        return (alpha + linear_decayed) * cosine_decayed + beta

    starter_learning_rate = 0.1
    decay_steps = 10000
    lr_decayed = tf.train.linear_cosine_decay(starter_learning_rate, global_step, decay_steps)
    tf.summary.scalar('linear_cosine_decay', lr_decayed) 
    
    
    '''
    global_step = min(global_step, decay_steps)
    linear_decay = (decay_steps - global_step) / decay_steps)
    cosine_decay = 0.5 * (
      1 + cos(pi * 2 * num_periods * global_step / decay_steps))
    decayed = (alpha + linear_decay + eps_t) * cosine_decay + beta
    decayed_learning_rate = learning_rate * decayed
    ''' 
    starter_learning_rate = 0.1
    decay_steps = 5000
    lr_decayed = tf.train.noisy_linear_cosine_decay(starter_learning_rate, global_step, decay_steps)
    tf.summary.scalar('noisy_linear_cosine_decay', lr_decayed) 

    
    sum_ops = tf.summary.merge_all()          #获取所有的操作
    
    
    # Creates a variable to hold the global_step.
#     global_step_tensor = tf.Variable(100, trainable=False, name='global_step')
#     global_step = tf.train.get_global_step()
#     print(global_step)

    
#     global_step= tf.train.create_global_step()
#     print(global_step)
    global_step= tf.train.get_or_create_global_step()
    global_step.assign_add(1)
    print(global_step)
 
  
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        # Initializes the variable.
        for i in range(10):
            print('global_step: %s' % tf.train.global_step(sess, global_step))
        
        summary_writer = tf.summary.FileWriter('log/', sess.graph)       #将监测结果输出目录
        for step in range(0, 10000):     #迭代写入文件
            s_val = sess.run(sum_ops) # 获取serialized监测结果：bytes类型的字符串 
            summary_writer.add_summary(s_val, global_step=step) # 写入文件 
            sess.run(increment_op)
            
            
            
            