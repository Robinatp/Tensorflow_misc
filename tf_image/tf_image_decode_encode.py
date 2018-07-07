## Encoding and Decoding

# *   @{tf.image.decode_gif}
# *   @{tf.image.decode_jpeg}
# *   @{tf.image.encode_jpeg}
# *   @{tf.image.decode_png}
# *   @{tf.image.encode_png}
# *   @{tf.image.decode_image}

import tensorflow as tf
import cv2

if __name__ == '__main__':
    img_name = "dog.jpg"
    
    
    image_jpg = tf.read_file(img_name)
    imgage_decode_jpeg = tf.image.decode_jpeg(image_jpg, channels=1, ratio=1, name="decode_jpeg_1")
    print(imgage_decode_jpeg.shape)
    print(imgage_decode_jpeg.dtype)
    
    sess = tf.Session()
    imgage_encode_jpeg = tf.image.encode_jpeg(sess.run(imgage_decode_jpeg), format='grayscale',name="encode_jpeg")
    print(imgage_encode_jpeg.shape)
    print(imgage_encode_jpeg.dtype)
    img = tf.image.decode_jpeg(sess.run(imgage_encode_jpeg), ratio=1, name="decode_jpeg_2")
    
    img = cv2.imshow("img", sess.run(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sess.close()
    
    
    ######################################################################################################
    imgage_decode_jpeg = tf.image.decode_jpeg(image_jpg, channels=3, ratio=1, name="decode_jpeg_1")
    print(imgage_decode_jpeg.shape)
    print(imgage_decode_jpeg.dtype)
    
    sess = tf.Session()
    imgage_encode_png = tf.image.encode_png(sess.run(imgage_decode_jpeg), name="encode_png")
    print(imgage_encode_png.shape)
    print(imgage_encode_png.dtype)
    
    img = tf.image.decode_png(sess.run(imgage_encode_png), name="decode_png")
    
    img = cv2.imshow("img", sess.run(img))
    cv2.waitKey(0)
    
    #######################################################################################################
    imgage_decode = tf.image.decode_image(image_jpg, name="decode_image")
    print(imgage_decode.shape)
    print(imgage_decode.dtype)
    imgage_encode_jpeg = tf.image.encode_jpeg(imgage_decode, quality=100, progressive=True, chroma_downsampling=False, optimize_size=True, name="encode_jpeg")
    
    sess = tf.Session()
    
    print(imgage_encode_jpeg.shape)
    print(imgage_encode_jpeg.dtype)
    
    img = tf.image.decode_png(sess.run(imgage_encode_jpeg), name="decode_jpeg")
    
    img = cv2.imshow("img", sess.run(img))
    cv2.waitKey(0)

        