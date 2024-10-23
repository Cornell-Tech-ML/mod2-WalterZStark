"""Collection of the core mathematical operators used throughout the code base."""

import math


from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of prelud of elementary functions


def mul(x: float, y: float) -> float:
    """Multiply 'x' by 'y'"""
    return x * y


def id(x: float) -> float:
    """Outputs 'x' unchanged"""
    return x


def add(x: float, y: float) -> float:
    """Adds 'x' and 'y'"""
    return x + y


def neg(x: float) -> float:
    """Returns the negated value of 'x'"""
    return -x


def lt(x: float, y: float) -> float:
    """Checks if 'x' is less than 'y'"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if 'x' is equal to 'y'"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the maximum of number between 'x' and 'y'"""
    if x > y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> bool:
    """Checks if 'x' and 'y' are close in value"""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Maps input 'x' to a value between 0 and 1 using the sigmoid activation function"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.e ** (x))


def relu(x: float) -> float:
    """Maps input 'x' to a value between 0 and 1 using the rectified linear unit activation function"""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Returns the natural logarithm of 'x'"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculates the exponential function of 'x'"""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal of 'x'"""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Computes the derivative of the log of 'x' times 'd'"""
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of the reciprocal of 'x' times 'd'"""
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of the rectified linear unit activation function of 'x' times 'd'"""
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Apply a given function to each element within an iterable.

    Args:
    ----
      fn: A function to perform operations on a float and return a float.

    Returns:
    -------
       A function that takes a list, applies 'fn' to each element, and returns a new list.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Apply a given function to combine elements from two iterables.

    Args:
    ----
      fn: combine two values

    Returns:
    -------
       Function that takes two equallity sized lists 'ls1' and 'ls2', and produced a new list
       by applying fn(x,y) on each pair of elements

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""Apply a given function to reduce an iterable to a single value.

    Args:
    ----
      fn: A function to perform operations on two floats and return a float.
      start: start value $x_0$

    Returns:
    -------
       function that takes a list 'ls' of elements
       $x_1 \ldits x_n$ and computers the reduction :math: 'fn(x_3, fn(x_2,
       fn(x_1, x_0)))

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list.

    Args:
    ----
      ls: A list containing floats.

    Returns:
    -------
       A list of all negated values in 'lst'.

    """
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding values from two lists together into one combined list.

    Args:
    ----
      ls1: A list containing floats.
      ls2: A list containing floats.

    Returns:
    -------
       A combined Iterable usiung corresponding values from 'lst_a' and 'lst_b'.

    """
    return zipWith(add)(ls1, ls2)


def sum(lst: Iterable[float]) -> float:
    """Sum all elements in a list.

    Args:
    ----
      lst: A list containing floats.

    Returns:
    -------
       A float of summed values from 'lst'.

    """
    return reduce(add, 0.0)(lst)


def prod(lst: Iterable[float]) -> float:
    """Product all elements in a list.

    Args:
    ----
      lst: A list containing floats.

    Returns:
    -------
       A float of multiplied values from 'lst'.

    """
    return reduce(mul, 1.0)(lst)
