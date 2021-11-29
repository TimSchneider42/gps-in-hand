from typing import Tuple, List

import tensorflow as tf

from gps.neural_network import NeuralNetwork


class NeuralNetworkFullyConnected(NeuralNetwork):
    def __init__(self, action_dimensions: int, observation_dimensions: int, layer_units: List[int]):
        self.__layer_units = layer_units
        super(NeuralNetworkFullyConnected, self).__init__(action_dimensions, observation_dimensions)

    def _build_network(self, observation_tensor: tf.Tensor, action_tensor: tf.Tensor,
                       action_precision_tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Operation, tf.Tensor]:
        with tf.name_scope("common"):
            layers = [observation_tensor]
            for d in self.__layer_units:
                layers.append(
                    tf.layers.dense(inputs=layers[-1], units=d, activation=tf.nn.relu, name=f"FC{len(layers)}"))
            output = tf.layers.dense(inputs=layers[-1], units=self.action_dimensions, name="action_out")
        with tf.name_scope("training"):
            diff = tf.reshape(action_tensor - output, (-1, self.action_dimensions, 1), name="diff")
            diff_scaled = tf.matmul(action_precision_tensor, diff, name="diff_scaled")
            loss = tf.matmul(tf.transpose(diff, perm=(0, 2, 1)), diff_scaled, name="loss")
            loss_mean = tf.reduce_mean(loss)
            train_op = tf.train.AdamOptimizer().minimize(loss_mean)
        w = tf.summary.FileWriter("/tmp/tf")
        w.add_graph(tf.get_default_graph())
        return output, train_op, loss_mean
