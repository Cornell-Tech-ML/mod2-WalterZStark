import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Creates a list of randomly placed points in 2D space.

    Args:
    ----
       N (int): A integer repesenting the number of desired points to create.

    Returns:
    -------
       List[Tuple[float, float]]: A list of tuples with randomly spaced 2D points.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Creates a simple graph with randomly placed points classified based on whether or not 'x_1' < 0.5.

    Args:
    ----
       N (int): A integer repesenting the number of desired points to create.

    Returns:
    -------
       Graph: A graph containing randomly spaced points split into groups over a centered vertical line.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Creates a simple graph with randomly placed points classified based on whether or not 'x_1' + 'x_2' < 0.5.

    Args:
    ----
       N (int): A integer repesenting the number of desired points to create.

    Returns:
    -------
       Graph: A graph containing randomly spaced points split into groups over a diagonal vertical line.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Creates a graph with randomly placed points, classified based on whether 'x_1' is less than 0.2 or greater than 0.8.

    Args:
    ----
       N (int): A integer repesenting the number of desired points to create.

    Returns:
    -------
       Graph: A graph with points labeled as 1 if x_1 < 0.2 or x_1 > 0.8, otherwise labeled as 0.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Creates a graph with randomly placed points, classified based on the XOR condition: whether 'x_1' is on one side of 0.5 and 'x_2' is on the opposite side.

    Args:
    ----
       N (int): Number of points to create.

    Returns:
    -------
       Graph: A graph with points labeled as 1 if (x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5), otherwise labeled as 0.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Creates a graph with randomly placed points, classified based on their distance from the center of the unit square.

    Args:
    ----
        N (int): Number of points to create.

    Returns:
    -------
        Graph: A graph with points labeled as 1 if they lie outside a circle of radius sqrt(0.1) centered at (0.5, 0.5), otherwise labeled as 0.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Creates a graph with points arranged in two intertwined spirals, classified by their spiral.

    Args:
    ----
        N (int): Number of points to create.

    Returns:
    -------
        Graph: A graph with points arranged in two intertwined spirals. Points in one spiral are labeled as 0, and points in the other spiral are labeled as 1.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
