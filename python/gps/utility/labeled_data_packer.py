from collections import OrderedDict
from typing import List, Any, Tuple, Iterable, Dict
import numpy as np


class LabeledDataPacker:
    """
    This class is responsible for packing data from multiple numpy arrays into a single numpy array and vice versa. Each
    unpacked array has a label assigned, which will determine its position in the packed array. The labels and positions
    can be specified in the constructor.
    """

    def __init__(self, labels: Iterable[Tuple[Any, int]]):
        """

        :param labels: Labels with dimensions in the preferred order. Please note that labels must be unique.
        """
        self._label_dimensions = OrderedDict(labels)
        # Check if labels are unique
        assert len(self._label_dimensions) == len(list(labels)), "Labels must be unique"
        assert not any(d <= 0 for _, d in labels), "Dimensions must be greater zero"

        label_slices = [[l, None] for l in self._label_dimensions.keys()]
        next_index = 0
        for i, (l, _) in enumerate(label_slices):
            label_slices[i][1] = slice(next_index, next_index + self._label_dimensions[l])
            next_index += self._label_dimensions[l]

        self._label_slices = OrderedDict(label_slices)
        self._dimensions = next_index

    def unpack(self, vector: np.array, concatenation_axis: int = -1) -> Dict[Any, np.array]:
        """
        Returns a dictionary, containing the labels and their unpacked vectors
        :param vector: The vector to be unpacked
        :param concatenation_axis: The axis on which the data was concatenated when packing
        :return:A dictionary, containing the labels and their unpacked vectors
        """
        assert vector.shape[concatenation_axis] == self.dimensions, \
            "Vector has the wrong dimension: expected {0}, got {1} (axis {2})".format(
                self.dimensions, vector.shape[concatenation_axis], concatenation_axis)
        slices = [slice(None)] * (concatenation_axis % len(vector.shape))
        return {l: vector[slices + [s]] for l, s in self._label_slices.items()}

    def pack(self, data: Dict[Any, np.array], concatenation_axis: int = -1) -> np.array:
        """
        Returns the packed form of the given data
        :param data: A dictionary containing the labels and vectors of the data to pack
        :param concatenation_axis: The axis on which to perform the concatenation of the data
        :return: The packed data
        """
        return np.concatenate([data[l] for l in self.labels], axis=concatenation_axis)

    @property
    def labels(self) -> List[Any]:
        """
        The labels specified in the constructor
        :return:
        """
        return list(self._label_dimensions.keys())

    @property
    def label_dimensions(self) -> OrderedDict:
        """
        A ordered dictionary containing the dimensions of each label
        :return:
        """
        return self._label_dimensions.copy()

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def label_slices(self) -> OrderedDict:
        return self._label_slices.copy()
