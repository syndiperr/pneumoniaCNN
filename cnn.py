import time

import tensorflow as tf

from create_dataset import *

IMAGE_SIZE = 252
NUM_CHANNELS = 3
NUM_CLASSES = 2
REPLACE_DATA = True


class TimeHistory(tf.train.SessionRunHook):
    def begin(self):
        self.times = []

    def before_run(self, run_context):
        self.iter_time_start = time.time()

    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.iter_time_start)


def parser(record):
    keys_to_feature = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(serialized=record, features=keys_to_feature)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.cast(parsed["label"], tf.int32)

    return { "image": image }, label


def input_fn(filenames, training, batch_size=32, buffer_size=1024):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(parser)

    if training:
        dataset = dataset.shuffle(buffer_size)
        num_repeat = None
    else:
        num_repeat = 1

    dataset = dataset.repeat(num_repeat)

    return dataset.batch(batch_size)


def train_input_fn():
    return input_fn(filenames=["train.tfrecords", "test.tfrecords"], training=True)


def eval_input_fn():
    return input_fn(filenames=["val.tfrecords"], training=False)


def tensor_shape(height, width, channels):
    # Condition to set up the tensor size correctly dependent on
    # usage of Tensorflow / Theano backend
    return (channels, height, width) if tf.keras.backend.image_data_format() == "channels_first" else (height, width, channels)


def cnn_functional_model(height, width, channels):
    # Retrieve the tensor shape for the inputs
    image_shape = tensor_shape(height, width, channels)
    # Making an input layer to specify the shape of inputs
    inputs = tf.keras.Input(shape=(image_shape))
    # Convolutional Layer 1 -> Max Pooling
    conv1 = tf.keras.layers.Conv2D(name='layer_conv1', filters=32, kernel_size=(3, 3), padding='same', activation=tf.keras.activations.relu)(inputs)
    maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=(2, 2))(conv1)
    # Convolutional Layer 2 -> Max Pooling
    conv2 = tf.layers.Conv2D(name='layer_conv2', filters=64, kernel_size=(3, 3), padding="same", activation=tf.keras.activations.relu)(maxpool1)
    maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=(2, 2))(conv2)
    # Convolutional Layer 3 -> Max Pooling
    conv3 = tf.layers.Conv2D(name='layer_conv3', filters=64, kernel_size=(3, 3), padding='same', activation=tf.keras.activations.relu)(maxpool2)
    maxpool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=(2, 2))(conv3)
    # Fully-Connected Layer 1 (Flattened Last Layer ->  Dense Layer 1)
    flatten = tf.keras.layers.Flatten()(maxpool3)
    fc1 = tf.keras.layers.Dense(name='layer_fc1', units=128, activation=tf.keras.activations.relu)(flatten)
    # Output Layer (Dropout -> Dense Layer 2)
    dropout = tf.keras.layers.Dropout(rate=0.5)(fc1)
    preds = tf.keras.layers.Dense(name='output_pred', units=NUM_CLASSES, activation=tf.keras.activations.softmax)(dropout)

    return inputs, preds


def cnn_sequential_model(height, width, channels):
    # Retrieve the tensor shapr for the inputs
    image_shape = tensor_shape(height, width, channels)
    # Making a model which has layers built upon it
    # NOTE: We need to specify the shape of the first layer being added to model with input_shape=
    net = tf.keras.Sequential(input_shape=image_shape)
    # Convolutional Layer 1 -> Max Pooling
    net.add(tf.keras.layers.Conv2D(
        name="layer_conv1", filters=32, kernel_size=(3, 3), padding="same", activation=tf.keras.activations.relu
    ))
    net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=(2, 2)))
    # Convolutional Layer 2 -> Max Pooling
    net.add(tf.keras.layers.Conv2D(
        name="layer_conv2", filters=64, kernel_size=(3, 3), padding="same", activation=tf.keras.activations.relu
    ))
    net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=(2, 2)))
    # Convolutional Layer 3 -> Max Pooling
    net.add(tf.keras.layers.Conv2D(
        name="layer_conv3", filters=64, kernel_size=(3, 3), padding="same", activation=tf.keras.activations.relu
    ))
    net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=(2, 2)))
    # Fully-Connected Layer 1 (Flattened Last Layer ->  Dense Layer 1)
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(name='layer_fc1', units=128, activation=tf.keras.activations.relu))
    # Output Layer (Dropout -> Dense Layer 2)
    net.add(tf.keras.layers.Dropout(rate=0.5))
    net.add(tf.keras.layers.Dense(name='output_pred', units=NUM_CLASSES, activation=tf.keras.activations.softmax))

    return net


if __name__ == '__main__':
    tf.logging.set_verbosity(2)
    # Pattern for train images
    train_dir = "chest_xray/train/*/*.jpeg"
    # Pattern for predicting images
    test_dir = "chest_xray/test/*/*.jpeg"
    # Pattern for val images
    val_dir = "chest_xray/val/*/*.jpeg"
    # Total available labels in data
    labels = ["NORMAL", "PNEUMONIA"]

    # Create a CNN model using the Keras Functional Way for Multi-Processing
    inputs, predictions = cnn_functional_model(height=IMAGE_SIZE, width=IMAGE_SIZE, channels=NUM_CHANNELS)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
        metrics=['accuracy']
    )
    # Set up for distributed GPU work
    strategy = None
    # strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=3)
    config = tf.estimator.RunConfig(model_dir="./model_ckpt/", train_distribute=strategy)
    estimator = tf.keras.estimator.model_to_estimator(model, config=config)

    time_hist = TimeHistory()

    BATCH_SIZE = 64
    NUM_EPOCHS = 1000
    estimator.train(
        input_fn=lambda: convertToDataset(
            train_dir,
            labels,
            shuffle=True,
            keep_percent=1 - 0.2,
            batch_size=BATCH_SIZE,
            buffer_size=2048,
            num_epochs=NUM_EPOCHS,
            prefetch_buffer_size=4
        ),
        hooks=[time_hist]
    )
    estimator.evaluate(
        input_fn=lambda: convertToDataset(
            val_dir,
            labels,
            shuffle=False,
            batch_size=BATCH_SIZE,
            buffer_size=1024,
            num_epochs=1
        )
    )
