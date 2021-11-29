from typing import Optional, Callable


def human_readable_formatter(precision: int = 2, target_exponent: int = 0, min_exp_tol: Optional[int] = None,
                             max_exp_tol: Optional[int] = None) -> Callable[[float], str]:
    if min_exp_tol is None:
        min_exp_tol = precision // 2
    if max_exp_tol is None:
        max_exp_tol = precision // 2

    def format(value: float) -> str:
        if 10 ** (target_exponent - min_exp_tol) <= abs(value) <= 10 ** (target_exponent + max_exp_tol):
            if target_exponent != 0:
                return "{f:.{p}f}e{e:02d}".format(p=precision, f=value / 10 ** target_exponent, e=target_exponent)
            else:
                return "{f:.{p}f}".format(p=precision, f=value / 10 ** target_exponent)
        else:
            return "{f:.{p}e}".format(p=precision, f=value / 10 ** target_exponent)

    return format
