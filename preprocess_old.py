import tensorflow as tf

def dataset(directory):
  def decode_image(image):
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [870768])
    return image / 255.0

  def decode_label(label):
    label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
    label = tf.reshape(label, [])  # label is a scalar
    return tf.to_int32(label)

  images = tf.data.FixedLengthRecordDataset(directory, 870768, header_bytes=16).map(decode_image)
  return images

iter = dataset("chest_xray/train/NORMAL/IM-0115-0001.jpeg").make_one_shot_iterator()

with tf.Session() as sess:
    image = iter.get_next()
    print(sess.run(image))