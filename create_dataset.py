import os, sys

import tensorflow as tf
import numpy as np
from PIL import Image


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def average_image_size(files_list):
    assert len(files_list) > 0, "Please give a list of files greater than 0."
    average_width, average_height = (0, 0)
    total_files = len(files_list)
    for image_file in files_list:
        image = Image.open(image_file)
        width, height = image.size
        average_width += width
        average_height += height

    return int(average_width / total_files), int(average_height / total_files)


def load_image(filename, average_image_size, reduce=0.0, custom_resize=(0,0)):
    # Read an image & try to central crop
    image = Image.open(filename)
    left, right, top, bottom = (0, 0, 0, 0)
    scaled_width, scaled_height = average_image_size
    result = None
    if image is not None:
        width, height = image.size
        percent = 1 - reduce
        left = int((width - (width * percent))/2)
        right = int((width + (width * percent))/2)
        top = int((height - (height * percent))/2)
        bottom = int((height + (height * percent))/2)
        cropped = image.crop((left, top, right, bottom))
        result = cropped.resize((scaled_width, scaled_height)) if custom_resize[0] is 0 and custom_resize[1] is 0 else cropped.resize(custom_resize)

    return np.array(result)


def createDataRecord(record_filename, image_files, labels, average_image_size, image_reduction=0.0, custom_resize=(0,0), print_every=500):
    # Open a TFRecord file
    with tf.python_io.TFRecordWriter(record_filename) as writer:
        number_files = len(image_files)
        for i in range(number_files):
            # Print how many images are saved every 1000 iterations
            if not i % print_every:
                dataset_type = "Unknown"
                if ("train" in record_filename):
                    dataset_type = "Train"
                elif("test" in record_filename):
                    dataset_type = "Test"
                else:
                    dataset_type = "Val"
                print("{} data: {}/{} saved.".format(dataset_type, i, number_files))
                sys.stdout.flush()

            # Load the image as NumPy Array
            image = load_image(image_files[i], average_image_size, reduce=image_reduction, custom_resize=custom_resize)
            label = labels[i]
            if image is None:
                continue

            # Create a feature
            feature = {
                "image_raw": _bytes_feature(image.tostring()),
                "label": _int64_feature(label)
            }
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Serialize to string & write to file
            writer.write(example.SerializeToString())
    sys.stdout.flush()


def convertToDataset(file_pattern, labels, image_size=(252, 252), keep_percent=1.0, shuffle=False, batch_size=64, num_epochs=None, buffer_size=2048, prefetch_buffer_size=None):
    table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(labels))
    num_classes = len(labels)

    def _map_func(image_file):
        label = tf.string_split([image_file], delimiter=os.sep).values[-2]
        image = tf.image.decode_jpeg(tf.read_file(image_file), channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.central_crop(image, central_fraction=keep_percent)
        image = tf.image.resize_images(image, size=image_size)
        return (image, tf.one_hot(table.lookup(label), num_classes))

    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)

    if num_epochs is not None and shuffle:
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size, num_epochs)
        )
    elif shuffle:
        dataset = dataset.shuffle(buffer_size)
    elif num_epochs is not None:
        dataset = dataset.repeat(num_epochs)

    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(map_func=_map_func, batch_size=batch_size, num_parallel_calls=os.cpu_count())
    )
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    return dataset


if __name__ == "__main__":
    ### Example to show how loading image works with resizing ###
    Image.fromarray(load_image("chest_xray/train/NORMAL/IM-0115-0001.jpeg", average_image_size(["chest_xray/train/NORMAL/IM-0115-0001.jpeg"]), 0.25, custom_resize=(252, 252))).show()
