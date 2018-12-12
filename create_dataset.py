from random import shuffle
import glob
import sys

import tensorflow as tf
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

    return result


def createDataRecord(record_filename, image_files, labels, average_image_size, image_reduction=1.0, custom_resize=(0,0), print_every=500):
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

            # Load the image
            image = load_image(image_files[i], average_image_size, reduce=image_reduction, custom_resize=custom_resize)
            label = labels[i]
            if image is None:
                continue

            # Create a feature
            feature = {
                "image_raw": _bytes_feature(image.tobytes()),
                "label": _int64_feature(label)
            }
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Serialize to string & write to file
            writer.write(example.SerializeToString())
    sys.stdout.flush()

### Example to show how loading image works with resizing ###
#load_image("chest_xray/train/NORMAL/IM-0115-0001.jpeg", average_image_size(["chest_xray/train/NORMAL/IM-0115-0001.jpeg"]), 0.25, custom_resize=(512, 512)).show()
# Parameters for % files in training / testing / validating sets
train = 0.7
test = 0.2
# The regex for the training set
dataset_dir = "chest_xray/*/*/*.jpeg"
# List of all training data shuffled
dataset_files = glob.glob(dataset_dir)
shuffle(dataset_files)
# List of all labels according to training data
### NOTE: 0 = normal 1 = pmneumonia ###
labels = [0 if 'NORMAL' in filename else 1 for filename in dataset_files]

# Split up dataset into training / testing / validating sets
dataset_length = len(dataset_files)
train_end = int(train * dataset_length)
test_end = train_end + int(test * dataset_length)

train_set = dataset_files[0:train_end]
train_labels = labels[0:train_end]
test_set = dataset_files[train_end:test_end]
test_labels = labels[train_end:test_end]
val_set = dataset_files[test_end:dataset_length]
val_labels = labels[test_end:dataset_length]

print("Training: {} images Testing: {} images Validation: {} images".format(len(train_set), len(test_set), len(val_set)))
# Average image width / height for the whole dataset
average_image_dims = average_image_size(dataset_files)
# Create TFRecord files to easily import into Tensorflow for each dataset
createDataRecord('train.tfrecords', train_set, train_labels, average_image_dims, image_reduction=0.25, custom_resize=(512, 512))
createDataRecord('test.tfrecords', test_set, test_labels, average_image_dims, image_reduction=0.25, custom_resize=(512, 512))
createDataRecord('val.tfrecords', val_set, val_labels, average_image_dims, image_reduction=0.25, custom_resize=(512, 512))
