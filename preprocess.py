from __future__ import absolute_import, division, print_function
from datetime import datetime

import os
import random
import sys
import threading
import multiprocessing
import glob

import numpy as np
import tensorflow as tf
from PIL import Image


def average_size(files_list):
    average_height = -1
    average_width = -1
    number_files = 0
    assert files_list is not None, "Files list given is empty."
    number_files = len(files_list)
    # Check if the directory exists
    for image in files_list:
        width, height = Image.open(image).size
        average_width += width
        average_height += height

    return int(average_width / number_files), int(average_height / number_files)


def crop_image(files_list):
    for image in files_list:
        image_read = Image.open(image)
        width, height = image_read.size


files_list = glob.glob(pathname="chest_xray/train/NORMAL*.jpeg")
print(average_size(files_list))

"""
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    image_resized = tf.image.resize_images(image_decoded, [128, 128])
    return image_resized, label

# Vector of filenames
filenames = tf.constant(["chest_xray/train/NORMAL/IM-0115-0001.jpeg", "chest_xray/train/NORMAL/IM-0117-0001.jpeg", "chest_xray/train/NORMAL/IM-0119-0001.jpeg"])

# Labels[i] is the label name for the image at index i
labels = tf.constant([0, 0, 0])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
"""
