from collections import OrderedDict
from typing import Mapping, TypeVar

KT = TypeVar('KT')  # Key type.
VT = TypeVar('VT')  # Value type.


class ReadOnlyOrderedDict(Mapping[KT, VT]):
    """
    Read-only wrapper around an OrderedDict.
    """

    def __init__(self, *args, **kwargs):
        self.__inner_dict = OrderedDict(*args, **kwargs)  # type: OrderedDict[KT, VT]

    def __getitem__(self, item):
        return self.__inner_dict.__getitem__(item)

    def __iter__(self):
        return self.__inner_dict.__iter__()

    def __len__(self):
        return self.__inner_dict.__len__()
