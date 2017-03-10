import os

import tensorflow as tf
import tflearn
from tensorflow.contrib.session_bundle import exporter

from train import MODEL_NAME, MODEL_PATH, resnet

pjoin = os.path.join


def export_model(path):
    net = resnet()
    g = net.graph
    X = g.get_operation_by_name('InputData/X')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        model = tflearn.DNN(net, session=sess)
        # model.load(pjoin(MODEL_PATH, MODEL_NAME))
        saver.restore(sess, pjoin(MODEL_PATH, MODEL_NAME))
        model_exporter = exporter.Exporter(saver)
        model_exporter.init(
            sess.graph.as_graph_def(),
            named_graph_signatures={
                'inputs': exporter.generic_signature({'x': X}),
                'outputs': exporter.generic_signature({'y': net})})
        model_exporter.export(path, tf.constant(200), sess)
        print 'Successfully exported model to %s' % path


if __name__ == '__main__':
    export_model('export')
