# -*- coding: utf-8 -*-
"""Functional test for learning rate decay."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_dir', 'log/', 'Summaries log directory')

def swish(features, beta=1):
  """Computes the Swish activation function: `x * sigmoid(x)`.

  Source: "Searching for Activation Functions" (Ramachandran et al. 2017)
  https://arxiv.org/abs/1710.05941

  Args:
    features: A `Tensor` representing preactivation values.
    name: A name for the operation (optional).

  Returns:
    The activation value.
  """
  # pylint: enable=g-doc-args
  features = tf.convert_to_tensor(features, name="features")
  return features * tf.nn.sigmoid(beta*features)
  
if __name__ == '__main__':
    
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    
    
    x_data = np.linspace(-15.0,15.0,10000)
#     x = tf.placeholder(tf.float32, shape=(None))
    x= tf.lin_space(-15.0,15.0,10000)
    
    
    
    # y = 1 / (1 + exp(-x))
    y_sigmoid = tf.nn.sigmoid(x)
#     tf.summary.scalar('tf_sigmoid', y_sigmoid) 


    # y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    y_tanh = tf.nn.tanh(x)
    
    # y = max(features, 0)
    y_relu = tf.nn.relu(x)
    
    
    # y = max(features, alpha*features)
    y_leaky_relu = tf.nn.leaky_relu(x, alpha=0.01)

    # y = min(max(features, 0), 6)
    y_relu6 = tf.nn.relu6(x)

    # exp(features) - 1 if < 0, features otherwise
    y_elu = tf.nn.elu(x)
    
    # y = log(exp(features) + 1)
    y_softplus = tf.nn.softplus(x)
    
    
    # y = features / (abs(features) + 1)
    y_softsign = tf.nn.softsign(x)
    
    #y = x * sigmoid(x)
    y_swish = tf.nn.swish(x)
    
    
    sum_ops = tf.summary.merge_all()
    

    global_step= tf.train.get_or_create_global_step()
    global_step = tf.assign_add(global_step,1)
    print(global_step)
 
  
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        f = plt.figure(figsize=(16,9))
        plt.subplot(331)
        plt.title("sigmoid y = 1 / (1 + exp(-x))")
        plt.plot(x_data,sess.run(y_sigmoid))
        plt.grid()
        
        plt.subplot(332)
        plt.title("tanh y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))")
        plt.plot(x_data,sess.run(y_tanh))
        plt.grid()
        
        plt.subplot(333)
        plt.title("relu y = max(features, 0)")
        plt.plot(x_data,sess.run(y_relu))
        plt.grid()
        
        plt.subplot(334)
        plt.title("leaky_relu  y = max(features, alpha*features)")
        plt.plot(x_data,sess.run(y_leaky_relu))
        plt.grid()
        
        plt.subplot(335)
        plt.title("relu6 y = min(max(features, 0), 6)")
        plt.plot(x_data,sess.run(y_relu6))
        plt.grid()
        
        plt.subplot(336)
        plt.title("elu y = exp(features) - 1 if < 0, features otherwise")
        plt.plot(x_data,sess.run(y_elu))
        plt.grid()
        
        plt.subplot(337)
        plt.title("softplus  y = log(exp(features) + 1)")
        plt.plot(x_data,sess.run(y_softplus))
        plt.grid()
        
        plt.subplot(338)
        plt.title("softsign  y = features / (abs(features) + 1)")
        plt.plot(x_data,sess.run(y_softsign))
        plt.grid()
        
        
        plt.subplot(339)
        plt.title("swish  y = x * sigmoid(x)")
        plt.plot(x_data,sess.run(y_swish))
        plt.grid()
        
        plt.show()
        
#         # Initializes the variable.
#         summary_writer = tf.summary.FileWriter('log/', sess.graph)       #将监测结果输出目录
#         for step in range(0, 10000):     #迭代写入文件
#             s_val = sess.run(sum_ops,feed_dict={x:x_data[tf.train.global_step(sess, global_step)-1]}) # 获取serialized监测结果：bytes类型的字符串 
#             summary_writer.add_summary(s_val, global_step=step) # 写入文件 
        
        
        