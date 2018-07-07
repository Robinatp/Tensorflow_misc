## Cropping

# *   @{tf.image.resize_image_with_crop_or_pad}
# *   @{tf.image.central_crop}
# *   @{tf.image.pad_to_bounding_box}
# *   @{tf.image.crop_to_bounding_box}
# *   @{tf.image.extract_glimpse}
# *   @{tf.image.crop_and_resize}


import tensorflow as tf
import matplotlib.pyplot as plt

img_name = ["dog.jpg"]
filename_queue = tf.train.string_input_producer(img_name)
img_reader = tf.WholeFileReader()
_,image_jpg = img_reader.read(filename_queue)

image_decode_jpeg = tf.image.decode_jpeg(image_jpg)
image_decode_jpeg = tf.image.convert_image_dtype(image_decode_jpeg, dtype=tf.float32)
#image_decode_jpeg = tf.expand_dims(image_decode_jpeg, 0)

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

image_crop = tf.image.resize_image_with_crop_or_pad(image_decode_jpeg, 500, 500)
image_pad = tf.image.resize_image_with_crop_or_pad(image_decode_jpeg, 1440, 1920)

image_central_crop = tf.image.central_crop(image_decode_jpeg, 0.5)

image_crop_to_bounding_box = tf.image.crop_to_bounding_box(image_decode_jpeg, 10, 10, 500, 400)


plt.figure(figsize=(12,12))
plt.subplot(221)
plt.imshow(sess.run(image_crop))
plt.title("resize image with crop")
plt.subplot(222)
plt.imshow(sess.run(image_pad))
plt.title("resize image with pad")
plt.subplot(223)
plt.imshow(sess.run(image_central_crop))
plt.title("central crop image")
plt.subplot(224)
plt.imshow(sess.run(image_crop_to_bounding_box))
plt.title("crop image to bounding box")
plt.show()




image_decode_jpeg = tf.image.decode_jpeg(image_jpg)
image_decode_jpeg = tf.image.convert_image_dtype(image_decode_jpeg, dtype=tf.float32)
#image_decode_jpeg = tf.expand_dims(image_decode_jpeg, 0)

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

image_flip_up_down = tf.image.flip_up_down(image_decode_jpeg)

image_random_flip_up_down = tf.image.random_flip_up_down(image_decode_jpeg)

image_flip_left_right = tf.image.flip_left_right(image_decode_jpeg)

image_random_flip_left_right = tf.image.random_flip_left_right(image_decode_jpeg)


plt.figure(figsize=(12,12))
plt.subplot(221)
plt.imshow(sess.run(image_flip_up_down))
plt.title("flip up down image")
plt.subplot(222)
plt.imshow(sess.run(image_random_flip_up_down))
plt.title("random flip up down image")
plt.subplot(223)
plt.imshow(sess.run(image_flip_left_right))
plt.title("flip left right image")
plt.subplot(224)
plt.imshow(sess.run(image_random_flip_left_right))
plt.title("random flip left right image")
plt.show()
