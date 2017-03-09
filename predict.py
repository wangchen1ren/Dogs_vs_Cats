import os

import numpy as np
import tflearn

from image_utils import preprocess_image
from train import resnet

pjoin = os.path.join
MODEL_PATH = pjoin(os.path.dirname(__file__), 'model')
MODEL_NAME = 'resnet_dogs_vs_cats.model'

import tensorflow as tf

tf.app.flags.DEFINE_string('image', None, 'image to classify')
FLAGS = tf.app.flags.FLAGS


def predict(image_file):
    image = preprocess_image(image_file, [256, 256, 3])
    # Training
    model = tflearn.DNN(resnet())
    model.load(pjoin(MODEL_PATH, MODEL_NAME))
    y_pred = model.predict([image])
    label = np.argmax(y_pred[0])
    return 'Cat' if label == 0 else 'Dog'


if __name__ == '__main__':
    pred = predict(FLAGS.image)
    print('It\'s a picture of %s!' % pred)
