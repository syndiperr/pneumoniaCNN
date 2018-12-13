import time, glob

import tensorflow as tf

from create_dataset import *

IMAGE_SIZE = 252
NUM_CHANNELS = 3
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

    return { "image": image }, label


def train_input_fn():
    return input_fn(filenames=["train.tfrecords", "test.tfrecords"], training=True)


def eval_input_fn():
    return input_fn(filenames=["val.tfrecords"], training=False)


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


def cnn_model(image_shape):
    inputs = tf.keras.Input(shape=(image_shape))
    conv1 = tf.keras.layers.Conv2D(name='layer_conv1', filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=(2, 2))(conv1)
    conv2 = tf.layers.Conv2D(name='layer_conv2', filters=64, kernel_size=(3, 3), padding="same", activation='relu')(maxpool1)
    maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=(2, 2))(conv2)
    conv3 = tf.layers.Conv2D(name='layer_conv3', filters=64, kernel_size=(3, 3), padding='same', activation='relu')(maxpool2)
    maxpool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=(2, 2))(conv3)
    flatten = tf.keras.layers.Flatten()(maxpool3)
    fc1 = tf.keras.layers.Dense(name='layer_fc1', units=128, activation='relu')(flatten)
    dropout = tf.keras.layers.Dropout(rate=0.5)(fc1)
    predictions = tf.keras.layers.Dense(name='output_pred', units=NUM_CLASSES, activation='softmax', input_shape=(None, 1))(dropout)

    return inputs, predictions

class TimeHistory(tf.train.SessionRunHook):
    def begin(self):
        self.times = []

    def before_run(self, run_context):
        self.iter_time_start = time.time()

    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.iter_time_start)


if __name__ == '__main__':
    # Pattern for train / test images
    train_test_dir = "chest_xray/[!val]*/*/*.jpeg"
    # Pattern for val images
    val_dir = "chest_xray/val/*/*.jpeg"
    # Total available labels in data
    labels = ["NORMAL", "PNEUMONIA"]

    if tf.keras.backend.image_data_format() == "channels_first":
        image_shape = (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    else:
        image_shape = (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    # Create a CNN model using Keras
    #inputs, preds = cnn_model(image_shape=image_shape)
    inputs = tf.keras.Input(shape=(image_shape))
    conv1 = tf.keras.layers.Conv2D(name='layer_conv1', filters=32, kernel_size=(3, 3), padding='same', activation=tf.keras.activations.relu)(inputs)
    maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=(2, 2))(conv1)
    conv2 = tf.layers.Conv2D(name='layer_conv2', filters=64, kernel_size=(3, 3), padding="same", activation=tf.keras.activations.relu)(maxpool1)
    maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=(2, 2))(conv2)
    conv3 = tf.layers.Conv2D(name='layer_conv3', filters=64, kernel_size=(3, 3), padding='same', activation=tf.keras.activations.relu)(maxpool2)
    maxpool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=(2, 2))(conv3)
    flatten = tf.keras.layers.Flatten()(maxpool3)
    fc1 = tf.keras.layers.Dense(name='layer_fc1', units=128, activation=tf.keras.activations.relu)(flatten)
    dropout = tf.keras.layers.Dropout(rate=0.5)(fc1)
    predictions = tf.keras.layers.Dense(name='output_pred', units=NUM_CLASSES, activation=tf.keras.activations.softmax)(dropout)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
    )
    """
    # Set up for distributed GPU work
    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=3)
    config = tf.estimator.RunConfig(train_distribute=strategy)
    estimator = tf.keras.estimator.model_to_estimator(model, config=config)
    """
    estimator = tf.keras.estimator.model_to_estimator(model)

    time_hist = TimeHistory()

    BATCH_SIZE = 64
    NUM_EPOCHS = 1
    estimator.train(
        input_fn=lambda: convertToDataset(
            train_test_dir,
            labels,
            shuffle=True,
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

    #print('Test loss: ', score[0])
    #print('Test accuracy: ', score[1])
