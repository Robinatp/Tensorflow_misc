import tensorflow as tf

a = tf.constant(30.0,name='a1') 
g = tf.Graph()
with g.as_default():
    c = tf.constant(30.0,name='c1')
    d = tf.constant(30.0)
    e = c*d
    f = tf.divide(c,d, name='divi')


print a.graph is g
#False

print a.graph is tf.get_default_graph()
#True

print g is tf.get_default_graph()
#False

tf.get_default_graph()
c2 = tf.constant(4.0, name='c_2')
print c2.graph is tf.get_default_graph()
#True

print c.graph is tf.get_default_graph()
#False


print a.graph
print tf.get_default_graph()
print g
# <tensorflow.python.framework.ops.Graph object at 0x11c8b9f50>


print g.get_operations()
#[<tf.Operation 'c1' type=Const>, <tf.Operation 'Const' type=Const>, 
#<tf.Op#eration 'mul' type=Mul>, <tf.Operation 'divi' type=RealDiv>]



a_1 = tf.Variable(40.0, dtype=tf.float32, name='a1')
print a_1.op

print g.get_operation_by_name('divi')