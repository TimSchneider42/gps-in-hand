import sys
import pickle
from typing import Iterable, NamedTuple

import numpy as np


def asdict(v):
    if isinstance(v, dict):
        return v
    elif hasattr(v, "__dict__"):
        return v.__dict__
    elif hasattr(v, "_asdict"):
        return v._asdict()
    else:
        return None


def analyse(d):
    d_dict = asdict(d)
    if d_dict is not None:
        results = {n: analyse(v) for n, v in d_dict.items()}
        results = {n: a for n, a in results.items() if a is not None}
        output = {"total_size": sum([a["total_size"] for a in results.values()]), "fields": results}
        return output if output["total_size"] > 0 else None
    elif isinstance(d, np.ndarray):
        return {"total_size": d.size}
    elif isinstance(d, str):
        return None
    elif isinstance(d, Iterable):
        results = [analyse(v) for v in d]
        results = [a for a in results if a is not None]
        output = {"total_size": sum([a["total_size"] for a in results]), "fields": results}
        return output if output["total_size"] > 0 else None
    else:
        return None


filename = sys.argv[1]

with open(filename, "rb") as f:
    data = pickle.load(f)

result = analyse(data)

print(result)
