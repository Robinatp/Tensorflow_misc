
import tensorflow as tf
import cv2



import tensorflow as tf
import matplotlib.pyplot as plt

img_name = ["dog.jpg"]
filename_queue = tf.train.string_input_producer(img_name)
img_reader = tf.WholeFileReader()
_,image_jpg = img_reader.read(filename_queue)

image_decode_jpeg = tf.image.decode_jpeg(image_jpg,channels=3)
image_decode_jpeg = tf.image.convert_image_dtype(image_decode_jpeg, dtype=tf.float32)# [0-255]=>[0-1]
print(image_decode_jpeg)

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

image_bilinear = tf.image.resize_images(image_decode_jpeg, size=[600,600], method=tf.image.ResizeMethod.BILINEAR)
print(image_bilinear)


# filter = tf.reshape(tf.Variable(tf.constant([ [-1.0,-1.0,-1.0],
#                                               [0,0,0],
#                                               [1.0,1.0,1.0],
#                                               [-2.0,-2.0,-2.0],
#                                               [0,0,0],
#                                               [2.0,2.0,2.0],
#                                               [-1.0,-1.0,-1.0],
#                                               [0,0,0],
#                                               [1.0,1.0,1.0],
#                                               [-1.0,-1.0,-1.0],
#                                               [0,0,0],
#                                               [1.0,1.0,1.0],
#                                               [-2.0,-2.0,-2.0],
#                                               [0,0,0],
#                                               [2.0,2.0,2.0],
#                                               [-1.0,-1.0,-1.0],
#                                               [0,0,0],
#                                               [1.0,1.0,1.0],
#                                               [-1.0,-1.0,-1.0],
#                                               [0,0,0],
#                                               [1.0,1.0,1.0],
#                                               [-2.0,-2.0,-2.0],
#                                               [0,0,0],
#                                               [2.0,2.0,2.0],
#                                               [-1.0,-1.0,-1.0],
#                                               [0,0,0],
#                                               [1.0,1.0,1.0]                                          
#                                               ])),[3,3,3,3])


# filter = tf.reshape(tf.Variable(tf.constant([ [-1.0,-1.0,-1.0],
#                                               [0,0,0],
#                                               [1.0,1.0,1.0],
#                                               [-2.0,-2.0,-2.0],
#                                               [0,0,0],
#                                               [2.0,2.0,2.0],
#                                               [-1.0,-1.0,-1.0],
#                                               [0,0,0],
#                                               [1.0,1.0,1.0]                                         
#                                               ])),[3,3,3,1])

filter = tf.Variable(tf.constant([[-1.0,-1.0,-1.0],
                                              [0,0,0],
                                              [1.0,1.0,1.0],
                                              [-2.0,-2.0,-2.0],
                                              [0,0,0],
                                              [2.0,2.0,2.0],
                                              [-1.0,-1.0,-1.0],
                                              [0,0,0],
                                              [1.0,1.0,1.0]                                         
                                              ],shape=[3,3,3,1]))
init_op = tf.global_variables_initializer()
sess.run(init_op)
 

cnn_output = tf.nn.conv2d(tf.expand_dims(image_bilinear,axis=0),filter,strides=[1,1,1,1],padding="SAME")
transpose_output =tf.nn.conv2d_transpose(cnn_output, filter, output_shape=[1, 600, 600, 3], strides=[1,1,1,1],padding="SAME")

max_output = tf.nn.max_pool(cnn_output,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
avg_output = tf.nn.avg_pool(cnn_output,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
print(cnn_output,transpose_output,max_output,avg_output)
cv2.imshow("image_decode_jpeg", sess.run(image_decode_jpeg))
cv2.imshow("cnn_output", sess.run(tf.squeeze(cnn_output)))
cv2.imshow("transpose_output", sess.run(tf.squeeze(transpose_output)))
cv2.imshow("max_output", sess.run(tf.squeeze(max_output)))
cv2.imshow("avg_output", sess.run(tf.squeeze(avg_output)))
cv2.waitKey(0)

sess.close()