import os
from random import shuffle
import glob

import tensorflow as tf

from create_dataset import *

IMAGE_SIZE = 224
NUM_CHANNELS = 1
NUM_CLASSES = 2
REPLACE_DATA = True


def parser(record):
    keys_to_feature = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(serialized=record, features=keys_to_feature)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.cast(parsed["label"], tf.int32)

    return image, label


def train_input_fn():
    return input_fn(filenames=["train.tfrecords", "test.tfrecords"], train=True)


def eval_input_fn():
    return input_fn(filenames=["val.tfrecords"], train=False)


def input_fn(filenames, train, batch_size=32, buffer_size=1024):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(parser)

    if train:
        dataset = dataset.shuffle(buffer_size)
        num_repeat = None
    else:
        num_repeat = 1

    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(batch_size)
    iter = dataset.make_one_shot_iterator()
    images_batch, labels_batch = iter.get_next()

    return {'image': images_batch}, labels_batch


def model_fn(features, labels, mode, params):
    num_classes = NUM_CLASSES

    net = features["image"]
    net = tf.identity(net, name="input_tensor")

    net = tf.reshape(net, [-1, 224, 224, 3])
    net = tf.identity(net, name="input_tensor_after")

    net = tf.layers.conv2d(inputs=net, name='layer_conv1', filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.layers.conv2d(inputs=net, name='layer_conv2', filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.layers.conv2d(inputs=net, name='layer_conv3', filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(inputs=net, name='layer_fc1', units=128, activation=tf.nn.relu)
    net = tf.layers.dropout(net, rate=0.5, noise_shape=None, seed=None, training=(mode == tf.estimator.ModeKeys.TRAIN))

    net = tf.layers.dense(inputs=net, name='layer_fc_2', units=num_classes)

    logits = net
    y_pred = tf.nn.softmax(logits=logits)
    y_pred = tf.identity(y_pred, name="output_pred")

    y_pred_class = tf.argmax(y_pred, axis=1)
    y_pred_class = tf.identity(y_pred_class, name="output_class")

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred_class)
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        metrics = { "accuracy": tf.metrics.accuracy(labels, y_pred_class) }

        spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)

    return spec


if __name__ == '__main__':
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
    test_end = int((train+test) * dataset_length)

    train_set = dataset_files[0:train_end]
    train_labels = labels[0:train_end]
    test_set = dataset_files[train_end:test_end]
    test_labels = labels[train_end:test_end]
    val_set = dataset_files[test_end:]
    val_labels = labels[test_end:]

    print("Training: {} images Testing: {} images Validation: {} images".format(len(train_set), len(test_set), len(val_set)))
    # Average image width / height for the whole dataset
    average_image_dims = average_image_size(dataset_files)
    custom_image_size = (IMAGE_SIZE, IMAGE_SIZE)
    # Create TFRecord files to easily import into Tensorflow for each dataset
    list_files = os.listdir()
    print("List of files in directory: ", list_files)
    if "train.tfrecords" not in list_files or REPLACE_DATA:
        createDataRecord('train.tfrecords', train_set, train_labels, average_image_dims, image_reduction=0.25, custom_resize=custom_image_size)

    if "test.tfrecords" not in list_files or REPLACE_DATA:
        createDataRecord('test.tfrecords', test_set, test_labels, average_image_dims, image_reduction=0.25, custom_resize=custom_image_size)

    if "val.tfrecords" not in list_files or REPLACE_DATA:
        createDataRecord('val.tfrecords', val_set, val_labels, average_image_dims, image_reduction=0.25, custom_resize=custom_image_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model = tf.estimator.Estimator(
            model_fn=model_fn,
            params={"learning_rate": 1e-4},
            model_dir="./pneumonia_model/"
        )
        for i in range(1000000):
            model.train(input_fn=train_input_fn, steps=1000)
            result = model.evaluate(input_fn=eval_input_fn)
            print(result)
            print("Classification accuracy: {0:.2%}".format(result["accuracy"]))
            sys.stdout.flush()
