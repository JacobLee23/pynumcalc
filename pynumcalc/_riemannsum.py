"""
"""

import functools
import itertools
import operator
import typing

import numpy as np

from ._partitions import Partitions
from ._typedef import (
    RealFunction, RealNFunction
)


class RiemannSum:
    """
    :param f:
    """
    def __init__(self, f: RealFunction):
        self._f = f

    @property
    def f(self) -> RealFunction:
        """
        """
        return self._f

    @classmethod
    def _delta(cls, interval: typing.Tuple[float, float, int]) -> float:
        """
        :param intervals:
        :return:
        """
        return (interval[1] - interval[0]) / interval[2]

    def integrate(self, interval: typing.Tuple[float, float, int]) -> float:
        """
        :param interval:
        :return:
        """
        return (
            self.f(Partitions.left(*interval)).sum()
            + self.f(Partitions.right(*interval)).sum()
        ) / 2 * self._delta(*interval)


class RiemannSumN:
    """
    :param f:
    """
    def __init__(self, f: RealNFunction):
        self._f = f

    @property
    def f(self) -> RealNFunction:
        """
        """
        return self._f

    @classmethod
    def _delta(cls, *intervals: typing.Tuple[float, float, int]) -> float:
        """
        :param intervals:
        :return:
        """
        return functools.reduce(
            operator.mul, (b - a for a, b, _ in intervals)
        ) / functools.reduce(
            operator.mul, (n for _, _, n in intervals)
        )

    def _summation(self, delta: float, *axes: np.ndarray) -> float:
        """
        :param delta:
        :param axes:
        :return:
        """
        return sum(map(self.f, itertools.product(*axes))) * delta

    def integrate(self, *intervals: typing.Tuple[float, float, int]) -> float:
        """
        :param intervals:
        :return:
        """
        dimensions = np.array(
            [[Partitions.left(*x), Partitions.right(*x)] for x in intervals]
        )
        return sum(
            self._summation(
                self._delta(*intervals), np.array([a[i] for a, i in zip(dimensions, indices)])
            ) for indices in itertools.product((0, 1), repeat=len(intervals))
        ) / pow(2, len(intervals))
