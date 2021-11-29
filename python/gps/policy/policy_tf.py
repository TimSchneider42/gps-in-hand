from typing import Optional, Dict, Any

import numpy as np
import tensorflow as tf

from allegro_pybullet.util import ReadOnlyDict
from gps.policy import Policy
from gps.neural_network import NeuralNetwork


class PolicyTf(Policy):
    """
    A neural network controller implemented in tensor flow. The network output is
    taken to be the mean, and Gaussian noise is added on top of it.
    U = net.forward(obs) + noise, where noise ~ N(0, diag(var))
    Args:
        obs_tensor: tensor representing tf observation. Used in feed dict for forward pass.
        act_op: tf op to execute the forward pass. Use sess.run on this op.
        var: Du-dimensional noise variance vector.
        sess: tf session.
        device_string: tf device string for running on either gpu or cpu.
    """

    def __init__(self, neural_network: NeuralNetwork, covariance: np.ndarray,
                 cholesky_covariance: Optional[np.ndarray] = None, inv_covariance: Optional[np.ndarray] = None,
                 nn_variable_values: Optional[Dict[tf.Variable, np.ndarray]] = None,
                 scale: Optional[np.ndarray] = None, bias: Optional[np.ndarray] = None,
                 random_seed: int = 0, device_string: Optional[str] = None):
        super(PolicyTf, self).__init__(neural_network.observation_dimensions, covariance, cholesky_covariance,
                                       inv_covariance)
        self.__scale_internal = np.ones(self.observation_dimensions) if scale is None else scale
        self.__bias_internal = np.zeros(self.observation_dimensions) if bias is None else bias
        self.__scale = scale
        self.__bias = bias
        assert self.__scale_internal.shape == (self.observation_dimensions,)
        assert self.__bias_internal.shape == (self.observation_dimensions,)

        self.__random_seed = random_seed
        self.__neural_network = neural_network
        self.__device_string = device_string

        self.__session: Optional[tf.Session] = None
        if nn_variable_values is None:
            tf.set_random_seed(self.__random_seed)
            with self.neural_network.graph.as_default():
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    nn_variable_values = {v: v.initial_value.eval() for v in neural_network.variables}
        self.__nn_variable_values = ReadOnlyDict({var: val.copy() for var, val in nn_variable_values.items()})

    def _act_mean(self, t: int, state: Optional[np.ndarray], obs: Optional[np.ndarray]) -> np.ndarray:
        assert self.is_active
        # Normalize obs.
        feed_dict = {
            self.__neural_network.observation_in: [obs * self.__scale_internal + self.__bias_internal]
        }
        with tf.device(self.__device_string):
            action_mean = self.__session.run(self.__neural_network.action_out, feed_dict=feed_dict)[0]
        return action_mean

    def probe(self, states: Optional[np.ndarray] = None, observations: Optional[np.ndarray] = None,
              noise: Optional[np.ndarray] = None):
        self.activate()
        feed_dict = {
            self.__neural_network.observation_in: observations * self.__scale_internal + self.__bias_internal
        }
        with tf.device(self.__device_string):
            action_mean = self.__session.run(self.__neural_network.action_out, feed_dict=feed_dict)
        if noise is not None:
            actions = action_mean + noise
        else:
            actions = action_mean
        return actions

    def activate(self):
        """
        Creates a session if it does not exist.
        :return:
        """
        if not self.is_active:
            with self.__neural_network.graph.as_default():
                self.__session = tf.Session()
                self.__session.run(tf.global_variables_initializer())
                self.__session.run([var.assign(val) for var, val in self.__nn_variable_values.items()])

    def deactivate(self):
        """
        Closes the current session if it exists.
        :return:
        """
        if self.is_active:
            self.__session.close()
            self.__session = None

    def prepare_sampling(self):
        self.activate()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        del state["_PolicyTf__session"]
        del state["_PolicyTf__nn_variable_values"]
        state["_PolicyTf__nn_variable_values"] = {var.name: val for var, val in self.__nn_variable_values.items()}
        return state

    def __setstate__(self, state: Dict[str, Any]):
        variable_values = state["_PolicyTf__nn_variable_values"]
        del state["_PolicyTf__nn_variable_values"]
        self.__dict__ = state
        self.__session = None
        with self.__neural_network.graph.as_default():
            variables = {v.name: v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)}
            self.__nn_variable_values = ReadOnlyDict({variables[n]: v for n, v in variable_values.items()})

    @property
    def neural_network(self) -> NeuralNetwork:
        return self.__neural_network

    @property
    def is_active(self) -> bool:
        return self.__session is not None and not self.__session._closed

    @property
    def scale(self) -> np.ndarray:
        return self.__scale

    @property
    def bias(self) -> np.ndarray:
        return self.__bias

    @property
    def random_seed(self) -> int:
        return self.__random_seed

    @property
    def nn_variable_values(self) -> ReadOnlyDict[tf.Variable, np.ndarray]:
        return self.__nn_variable_values

    @property
    def device_string(self) -> Optional[str]:
        return self.__device_string
