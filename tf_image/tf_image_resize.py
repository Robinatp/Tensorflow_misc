## Resizing

# *   @{tf.image.resize_images}
# *   @{tf.image.resize_area}
# *   @{tf.image.resize_bicubic}
# *   @{tf.image.resize_bilinear}
# *   @{tf.image.resize_nearest_neighbor}


import tensorflow as tf
import cv2



import tensorflow as tf
import matplotlib.pyplot as plt

img_name = ["dog.jpg"]
filename_queue = tf.train.string_input_producer(img_name)
img_reader = tf.WholeFileReader()
_,image_jpg = img_reader.read(filename_queue)

image_decode_jpeg = tf.image.decode_png(image_jpg)
image_decode_jpeg = tf.image.convert_image_dtype(image_decode_jpeg, dtype=tf.float32)# [0-255]=>[0-1]

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

image_bilinear = tf.image.resize_images(image_decode_jpeg, size=[1200,1920], method=tf.image.ResizeMethod.BILINEAR)
print(image_bilinear)
# print(sess.run(image_bilinear))

image_nearest_neighbor = tf.image.resize_images(image_decode_jpeg, size=[728,1280], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
print(image_nearest_neighbor)
# print(sess.run(image_nearest_neighbor))

image_bicubic = tf.image.resize_images(image_decode_jpeg, size=[720,1440], method=tf.image.ResizeMethod.BICUBIC)
print(image_bicubic)
# print(sess.run(image_bicubic))


image_area = tf.image.resize_images(image_decode_jpeg, size=[1080,1920], method=tf.image.ResizeMethod.AREA)
print(image_area)
# print(sess.run(image_area))



#uint8[0-255],float32[0-1]
plt.figure()
plt.subplot(221)
plt.imshow(sess.run(image_bilinear))
plt.title("bilinear interpolation")
plt.subplot(222)
plt.imshow(sess.run(image_nearest_neighbor))
plt.title("nearest neighbor interpolation")
plt.subplot(223)
plt.imshow(sess.run(tf.image.convert_image_dtype(image_bicubic, dtype=tf.uint8)))
# print(sess.run(tf.image.convert_image_dtype(image_bicubic, dtype=tf.uint8)))
plt.title("bicubic interpolation")
plt.subplot(224)
plt.imshow(sess.run(tf.image.convert_image_dtype(image_area, dtype=tf.float32)))
# print(sess.run(tf.image.convert_image_dtype(image_area, dtype=tf.float32)))
plt.title("area interpolation")
plt.show()