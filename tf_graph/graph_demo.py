# -*- coding: utf-8 -*-  
import tensorflow as tf

c = tf.constant(4.0)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
c_out = sess.run(c)
print(c_out)
print(c.graph == tf.get_default_graph())
print(c.graph)
print(tf.get_default_graph())


g = tf.Graph()
with g.as_default():
    c = tf.constant(4.0)

sess = tf.Session(graph=g)
c_out = sess.run(c)
print(c_out)
print(g)
print(tf.get_default_graph())


g1 = tf.Graph()
with g1.as_default():
    c1 = tf.constant(4.0)

g2 = tf.Graph()
with g2.as_default():
    c2 = tf.constant(20.0)

with tf.Session(graph=g1) as sess1:
    print(sess1.run(c1))
    print(g1)
with tf.Session(graph=g2) as sess2:
    print(sess2.run(c2))
    print(g2)



g3 = tf.Graph()
with g3.as_default():
    # 需要加上名称，在读取pb文件的时候，是通过name和下标来取得对应的tensor的
    c1 = tf.constant(4.0, name='c1')

g4 = tf.Graph()
with g4.as_default():
    c2 = tf.constant(20.0)

with tf.Session(graph=g3) as sess1:
    print(sess1.run(c1))
with tf.Session(graph=g4) as sess2:
    print(sess2.run(c2))

# g1的图定义，包含pb的path, pb文件名，是否是文本默认False
tf.train.write_graph(g3.as_graph_def(),'.','graph.pb',False)





from tensorflow.python.platform import gfile  
#load graph  
with gfile.FastGFile("./graph.pb",'rb') as f:  
    graph_def = tf.GraphDef()  
    graph_def.ParseFromString(f.read())  
    tf.import_graph_def(graph_def, name='')  
  
sess = tf.Session()  
c1_tensor = sess.graph.get_tensor_by_name("c1:0")  
c1 = sess.run(c1_tensor)  
print(c1)  
print(tf.get_default_graph())






#.穿插调用
g5 = tf.Graph()
with g5.as_default():
    # 声明的变量有名称是一个好的习惯，方便以后使用
    c1 = tf.constant(4.0, name="c1")

g6 = tf.Graph()
with g6.as_default():
    c2 = tf.constant(20.0, name="c2")


with tf.Session(graph=g6) as sess1:
    # 通过名称和下标来得到相应的值
    c1_list = tf.import_graph_def(g5.as_graph_def(), return_elements = ["c1:0"], name = '')
    print(sess1.run(c1_list[0]+c2))





#our NN's output  
logits=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])  
#step1:do softmax  
y=tf.nn.softmax(logits)  
#true label  
y_=tf.constant([[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]])  
#step2:do cross_entropy  
cross_entropy = -tf.reduce_sum(y_*tf.log(y))  
#do cross_entropy just one step  
cross_entropy2=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))#dont forget tf.reduce_sum()!!  
  
with tf.Session() as sess:  
    softmax=sess.run(y)  
    c_e = sess.run(cross_entropy)  
    c_e2 = sess.run(cross_entropy2)  
    print("step1:softmax result=")  
    print(softmax)  
    print("step2:cross_entropy result=")  
    print(c_e)  
    print("Function(softmax_cross_entropy_with_logits) result=")  
    print(c_e2)  

