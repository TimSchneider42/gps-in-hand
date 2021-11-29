from collections import defaultdict
from typing import List, Optional, Dict

from allegro_pybullet.util import ReadOnlyDefaultDict
from gps.algorithm import Algorithm
from gps.cost import Cost


class DebugCostFunction:
    def __init__(self, cost: Cost, desc: str, exclude_from_total_sum: bool = False):
        """

        :param cost:                    Cost function
        :param desc:                    Description of the cost function
        :param exclude_from_total_sum:  True, to exclude this cost function from the total sum when executing
                                        analyze_cost
        """
        self.__cost = cost
        self.__desc = desc
        self.__exclude_from_total_sum = exclude_from_total_sum

    @property
    def cost(self) -> Cost:
        return self.__cost

    @property
    def desc(self) -> str:
        return self.__desc

    @property
    def exclude_from_total_sum(self) -> bool:
        return self.__exclude_from_total_sum

    def __repr__(self):
        return "DebugCostFunction " + self.__desc


class Config:
    def __init__(self, algorithm: Algorithm, iterations: int = 10,
                 debug_cost_functions: Optional[Dict[int, List[DebugCostFunction]]] = None):
        """

        :param algorithm:               Algorithm to use in experiment.
        :param iterations:              Number of iterations to run.
        :param debug_cost_functions:    Dictionary mapping each condition number to a list of debug cost functions.
        """
        self.__iterations = iterations
        self.__algorithm: Algorithm = algorithm
        self.__debug_cost_functions = ReadOnlyDefaultDict(defaultdict(list, debug_cost_functions))

    @property
    def iterations(self) -> int:
        return self.__iterations

    @property
    def algorithm(self) -> Algorithm:
        return self.__algorithm

    @property
    def debug_cost_functions(self) -> ReadOnlyDefaultDict[int, List[DebugCostFunction]]:
        return self.__debug_cost_functions
