from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List

import tensorflow as tf


class NeuralNetwork(ABC):
    def __init__(self, action_dimensions: int, observation_dimensions: int):
        self.__action_dimensions = action_dimensions
        self.__observation_dimensions = observation_dimensions
        self.__initialize()

    def __initialize(self):
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            self.__observation_in = tf.placeholder("float", (None, self.__observation_dimensions),
                                                   name="observation_in")
            self.__action_in = tf.placeholder("float", (None, self.__action_dimensions), name="action_in")
            self.__action_precision_in = tf.placeholder(
                "float", (None, self.__action_dimensions, self.__action_dimensions), name="action_precision_in")
            self.__action_out, self.__training_op, self.__loss = self._build_network(
                self.__observation_in, self.__action_in, self.__action_precision_in)

    @abstractmethod
    def _build_network(self, observation_tensor: tf.Tensor, action_tensor: tf.Tensor,
                       action_precision_tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Operation, tf.Tensor]:
        pass

    @property
    def observation_dimensions(self) -> int:
        return self.__observation_dimensions

    @property
    def action_dimensions(self) -> int:
        return self.__action_dimensions

    @property
    def variables(self) -> List[tf.Variable]:
        with self.__graph.as_default():
            return [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]

    @property
    def loss(self) -> tf.Tensor:
        return self.__loss

    @property
    def observation_in(self) -> tf.Tensor:
        return self.__observation_in

    @property
    def action_in(self) -> tf.Tensor:
        return self.__action_in

    @property
    def action_precision_in(self) -> tf.Tensor:
        return self.__action_precision_in

    @property
    def action_out(self) -> tf.Tensor:
        return self.__action_out

    @property
    def training_op(self) -> tf.Operation:
        return self.__training_op

    @property
    def graph(self) -> tf.Graph:
        return self.__graph

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        for n in ["__graph", "__observation_in", "__action_in", "__action_precision_in", "__action_out",
                  "__training_op", "__loss"]:
            del state["_NeuralNetwork" + n]
        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__ = state
        self.__initialize()
