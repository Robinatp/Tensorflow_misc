# -*- coding: utf-8 -*-
"""Functional test for learning rate decay."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.python.ops.variable_scope import get_variable


if __name__ == '__main__':
    
    #下面这两个定义是等价的
#     v = tf.get_variable('v', shape=[1], initializer=tf.constant_initializer(1.0))
#     print(v)
#     v = tf.Variable(tf.constant(1.0, shape=[1]), name='v')
#     print(v)
    
    '''
    tf.get_variable和tf.Variable最大的区别就在于指定变量名称的参数。
    对于tf.Variable函数，变量名称是一个可选的参数，通过name=”v”的形式给出，
    对于tf.get_variable函数，变量名称是一个必填的参数，tf.get_variable会根据这个名称去创建或者获取变量。
    '''
    
    
    '''
    通过tf.variable_scope函数可以控制tf.get_variable函数的语义。
    当tf.variable_scope函数的参数reuse=True生成上下文管理器时，该上下文管理器内的所有的tf.get_variable函数会直接获取已经创建的变量，如果变量不存在则报错；
    当tf.variable_scope函数的参数reuse=False或者None时创建的上下文管理器中，tf.get_variable函数则直接创建新的变量，若同名的变量已经存在则报错。
    '''
    
    
    #tf.name_scope()
    with tf.name_scope('name_scope_test'):
        v1 = tf.get_variable('v', shape=[1],  initializer=tf.constant_initializer(1.0))
        v2 = tf.Variable(tf.constant(1.0, shape=[1]), name='v')
        v3 = tf.Variable(tf.constant(1.0, shape=[1]), name='v')

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print('the name of v1:', v1.name)
        print('the name of v2:', v2.name)
        print('the name of v3:', v3.name)
    '''
    tf.name_scope()对tf.get_variable没有影响，但对tf.Variable有影响，会在原有的name上加上命名空间，也就是name_scope_test/v:0，
    如果在tf.name_scope作用域下有相同的name，则会自动加上命名空间和会重命名name，即name_scope_test/v:0和name_scope_test/v_1:0
    '''
        
    #tf.variable_scope()
    with tf.variable_scope('variable_scope_test'):
        v1 = tf.get_variable('v', shape=[1],  initializer=tf.constant_initializer(1.0))
        v2 = tf.Variable(tf.constant(1.0, shape=[1]), name='v')
        v3 = tf.Variable(tf.constant(1.0, shape=[1]), name='v')
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print('the name of v1:', v1.name)
        print('the name of v2:', v2.name)
        print('the name of v3:', v3.name)
    '''
    tf.variable_scope对tf.get_variable有影响，会在原有的name上加上变量空间，也就是variable_scope_test/v:0。
    tf.Variable在tf.name_scope和tf.variable_scope作用域下的效果一样
    '''
        
    with tf.variable_scope('variable_scope_test2'):
        v1 = tf.get_variable('v', shape=[1],  initializer=tf.constant_initializer(1.0))
        v2 = tf.Variable(tf.constant(1.0, shape=[1]), name='v')
        v3 = tf.Variable(tf.constant(1.0, shape=[1]), name='v')

    with tf.variable_scope('variable_scope_test2'):
        v4 = tf.get_variable('vv', shape=[1])    #ValueError: Variable variable_scope_test2/v already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print('the name of v1:', v1.name)
        print('the name of v2:', v2.name)
        print('the name of v3:', v3.name)
        print('the name of v4:', v4.name)
    '''
    因为在第二块with tf.variable_scope('variable_scope_test2'):处又在variable_scope_test2变量命名空间下定义了name为v的变量，也就是这里（v4 = tf.get_variable('v', shape=[1])）重新定义了已存在的变量
    '''   
        
    with tf.variable_scope('variable_scope_test3'):
        v1 = tf.get_variable('v', shape=[1],  initializer=tf.constant_initializer(1.0))
        v2 = tf.Variable(tf.constant(1.0, shape=[1]), name='v')
        v3 = tf.Variable(tf.constant(1.0, shape=[1]), name='v')

    with tf.variable_scope('variable_scope_test3', reuse=True):
        v4 = tf.get_variable('v', shape=[1])    
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print('the name of v1:', v1.name)
        print('the name of v2:', v2.name)
        print('the name of v3:', v3.name)
        print('the name of v4:', v4.name)
    
    '''
    因为在第二块with tf.variable_scope('variable_scope_test', reuse=True)处设置了reuse=True
    '''
    
    
    
    
    with tf.variable_scope('variable_scope_test4'):
        v1 = tf.get_variable('v', shape=[1],  initializer=tf.constant_initializer(1.0))
        v2 = tf.Variable(tf.constant(1.0, shape=[1]), name='v')
        v3 = tf.Variable(tf.constant(1.0, shape=[1]), name='v')

    with tf.variable_scope('variable_scope_test4', reuse=True):
        v4 = tf.get_variable('v', shape=[1])    #ValueError: Variable variable_scope_test4/v1 does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=tf.AUTO_REUSE in VarScope?
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print('the name of v1:', v1.name)
        print('the name of v2:', v2.name)
        print('the name of v3:', v3.name)
        print('the name of v4:', v4.name)
    
    '''
    之所以报这个错，是因为设置reuse=True之后在该变量命名空间内，tf.get_variable只能获取已存在的变量而不能创建新变量
    '''
     
    #但如果又想创建变量，又想重用变量即获取变量呢？那可以用下面这个方法
    with tf.variable_scope('variable_scope_test5'):
        v1 = tf.get_variable('v', shape=[1], initializer=tf.constant_initializer(1.0))
        v2 = tf.Variable(tf.constant(1.0, shape=[1]), name='v')
        v3 = tf.Variable(tf.constant(1.0, shape=[1]), name='v')

    with tf.variable_scope('variable_scope_test5') as scope:
        v4 = tf.get_variable('v1', shape=[1], initializer=tf.constant_initializer(1.0))
        scope.reuse_variables()
        v5 = tf.get_variable('v', shape=[1])    
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print('the name of v1:', v1.name)
        print('the name of v2:', v2.name)
        print('the name of v3:', v3.name)
        print('the name of v4:', v4.name)
        print('the name of v5:', v5.name)
    
    
    
    
    
    # 在名字为foo的命名空间内创建名字为v的变量
    with tf.variable_scope("foo"):
        v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))
    
    '''
    # 因为命名空间foo内已经存在变量v，再次创建则报错
    with tf.variable_scope("foo"):
        v = tf.get_variable("v", [1])
    # ValueError: Variable foo/v already exists, disallowed.
    # Did you mean to set reuse=True in VarScope?
    '''
    # 将参数reuse参数设置为True，则tf.get_variable可直接获取已声明的变量
    with tf.variable_scope("foo", reuse=True):
        v1 = tf.get_variable("v", [1])
        print(v == v1) # True
    
    '''
    # 当reuse=True时，tf.get_variable只能获取指定命名空间内的已创建的变量
    with tf.variable_scope("bar", reuse=True):
        v2 = tf.get_variable("v", [1])
    # ValueError: Variable bar/v does not exist, or was not created with
    # tf.get_variable(). Did you mean to set reuse=None in VarScope?
    '''
    
    with tf.variable_scope("root"):
        # 通过tf.get_variable_scope().reuse函数获取当前上下文管理器内的reuse参数取值
        print(tf.get_variable_scope().reuse) # False
    
        with tf.variable_scope("foo1", reuse=True):
            print(tf.get_variable_scope().reuse) # True
    
            with tf.variable_scope("bar1"):
                # 嵌套在上下文管理器foo1内的bar1内未指定reuse参数，则保持与外层一致
                print(tf.get_variable_scope().reuse) # True
    
        print(tf.get_variable_scope().reuse) # False
    
    # tf.variable_scope函数提供了一个管理变量命名空间的方式
    u1 = tf.get_variable("u", [1])
    print(u1.name) # u:0
    with tf.variable_scope("foou"):
        u2 = tf.get_variable("u", [1])
        print(u2.name) # foou/u:0
    
    with tf.variable_scope("foou"):
        with tf.variable_scope("baru"):
            u3 = tf.get_variable("u", [1])
            print(u3.name) # foou/baru/u:0
    
        u4 = tf.get_variable("u1", [1])
        print(u4.name) # foou/u1:0
    
    # 可直接通过带命名空间名称的变量名来获取其命名空间下的变量
    with tf.variable_scope("", reuse=True):
        u5 = tf.get_variable("foou/baru/u", [1])
        print(u5.name)  # foou/baru/u:0
        print(u5 == u3) # True
        u6 = tf.get_variable("foou/u1", [1])
        print(u6.name)  # foou/u1:0
        print(u6 == u4) # True
    
        
        
    #reuse为True的时候表示用tf.get_variable 得到的变量可以在别的地方重复使用
    with tf.variable_scope('V1'):  
        a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))  
  
    with tf.variable_scope('V1', reuse=True):  
        a3 = tf.get_variable('a1')  
      
    with tf.Session() as sess:  
        sess.run(tf.initialize_all_variables())  
        print (a1.name)
        print (sess.run(a1)) 
        print (a3.name)
        print (sess.run(a3))   
        
        
    with tf.variable_scope('V1-') as scope:  
        a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))  
        scope.reuse_variables()  #
        a3 = tf.get_variable('a1')  
  
    with tf.Session() as sess:  
        sess.run(tf.initialize_all_variables())  
        print (a1.name)
        print (sess.run(a1)) 
        print (a3.name)
        print (sess.run(a3))   
        
        
        
        
        
        
        
        
        