## Working with Bounding Boxes

# *   @{tf.image.draw_bounding_boxes}
# *   @{tf.image.non_max_suppression}
# *   @{tf.image.sample_distorted_bounding_box}


import tensorflow as tf
import matplotlib.pyplot as plt

img_name = ["dog.jpg"]
filename_queue = tf.train.string_input_producer(img_name)
img_reader = tf.WholeFileReader()
_,image_jpg = img_reader.read(filename_queue)

image_decode_jpeg = tf.image.decode_jpeg(image_jpg)
image_decode_jpeg = tf.image.convert_image_dtype(image_decode_jpeg, dtype=tf.float32)
img = image_decode_jpeg
image_decode_jpeg = tf.expand_dims(image_decode_jpeg, 0)

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

box = tf.constant([[[0.2, 0.6, 0.5, 0.8]]])
image_bilinear = tf.image.draw_bounding_boxes(image_decode_jpeg, box)




# Generate a single distorted bounding box.
begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
          tf.shape(img),
          bounding_boxes=box)
print(begin, size, bbox_for_draw)
# Draw the bounding box in an image summary.
image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(img, 0),
                                                    bbox_for_draw)
# Employ the bounding box to distort the image.
distorted_image = tf.slice(img, begin, size)


plt.figure()
plt.subplot(131)
plt.imshow(sess.run(tf.squeeze(image_with_box)))
plt.axis('off')

plt.subplot(132)
plt.imshow(sess.run(img))
plt.axis('off')

plt.subplot(133)
plt.imshow(sess.run(distorted_image))
plt.axis('off')
plt.show()