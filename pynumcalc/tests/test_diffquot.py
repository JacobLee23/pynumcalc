"""
Unit tests for :py:mod:`pynumcalc.diffquot`.
"""

import itertools
import typing

import numpy as np
import pytest

from pynumcalc.diffquot import (
    DifferenceQuotient, PDifferenceQuotient
)
from pynumcalc.typedef import (
    RealFunction, RealNFunction
)


LOWER, UPPER = -10, 10
TEST_INTERVAL = range(LOWER, UPPER + 1)

REAL_FUNCTIONS = [
    lambda x: 0,
    lambda x: 1,
    lambda x: -1,
    lambda x: x,
    lambda x: -x,
    lambda x: x ** 2,
    lambda x: -x ** 2,
    lambda x: x ** 3,
    lambda x: -x ** 3
]
REAL_NFUNCTIONS = [
    ((lambda x: 0), 1),
    ((lambda x: 0), 2),
    ((lambda x: 0), 3),
    ((lambda x: 1), 1),
    ((lambda x: 1), 2),
    ((lambda x: 1), 3),
    ((lambda x: x[0]), 1),
    ((lambda x: x[0]), 2),
    ((lambda x: x[1]), 2),
    ((lambda x: x[0]), 3),
    ((lambda x: x[1]), 3),
    ((lambda x: x[2]), 3),
    ((lambda x: x[0] + x[1]), 2),
    ((lambda x: x[0] + x[1]), 3),
    ((lambda x: x[1] + x[2]), 3),
    ((lambda x: x[0] + x[2]), 3),
    ((lambda x: x[0] + x[1] + x[2]), 3),
    ((lambda x: x[0] ** 2), 1),
    ((lambda x: x[0] ** 2), 2),
    ((lambda x: x[1] ** 2), 2),
    ((lambda x: x[0] ** 2), 3),
    ((lambda x: x[1] ** 2), 3),
    ((lambda x: x[2] ** 2), 3),
    ((lambda x: x[0] ** 2 + x[1] ** 2), 2),
    ((lambda x: x[0] ** 2 + x[1] ** 2), 3),
    ((lambda x: x[1] ** 2 + x[2] ** 2), 3),
    ((lambda x: x[0] ** 2 + x[2] ** 2), 3),
    ((lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2), 3),
    ((lambda x: x[0] * x[1]), 2),
    ((lambda x: x[0] * x[1]), 3),
    ((lambda x: x[1] * x[2]), 3),
    ((lambda x: x[0] * x[2]), 3),
    ((lambda x: x[0] * x[1] * x[2]), 3),
    ((lambda x: (x[0] * x[1]) ** 2), 2),
    ((lambda x: (x[0] * x[1]) ** 2), 3),
    ((lambda x: (x[1] * x[2]) ** 2), 3),
    ((lambda x: (x[0] * x[2]) ** 2), 3),
    ((lambda x: (x[0] * x[1] * x[2]) ** 2), 3)
]


@pytest.mark.parametrize("h", [pow(2, n) for n in range(0, -10, -1)])
class TestDifferenceQuotient:
    """
    Unit tests for :py:class:`DifferenceQuotient`.
    """
    @pytest.mark.parametrize(
        ("f", "expected"), zip(
            REAL_FUNCTIONS, [
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 1,
                lambda h, x: -1,
                lambda h, x: 2 * x,
                lambda h, x: -2 * x,
                lambda h, x: 3 * x ** 2 + h ** 2 / 4,
                lambda h, x: -3 * x ** 2 + -h ** 2 / 4
            ]
        )
    )
    def test_first(
        self, f: RealFunction, expected: typing.Callable[[float, float], float], h: float
    ):
        """
        Unit test for :py:meth:`DifferenceQuotient.first`.
        """
        diffq = DifferenceQuotient(f, h)

        for x in TEST_INTERVAL:
            assert diffq.first(x) == expected(h, x)

    @pytest.mark.parametrize(
        ("f", "expected"), zip(
            REAL_FUNCTIONS, [
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 2,
                lambda h, x: -2,
                lambda h, x: 6 * x,
                lambda h, x: -6 * x
            ]
        )
    )
    def test_second(
        self, f: RealFunction, expected: typing.Callable[[float, float], float], h: float
    ):
        """
        Unit test for :py:meth:`DifferenceQuotient.second`.
        """
        diffq = DifferenceQuotient(f, h)

        for x in TEST_INTERVAL:
            assert diffq.second(x) == expected(h, x)

    @pytest.mark.parametrize("f", REAL_FUNCTIONS)
    def test_nth(self, f: RealFunction, h: float):
        """
        Unit test for :py:meth:`DifferenceQuotient.nth`
        """
        diffq = DifferenceQuotient(f, h)

        for x in TEST_INTERVAL:
            assert diffq.nth(x, 1) == diffq.first(x)
            assert diffq.nth(x, 2) == diffq.second(x)


@pytest.mark.parametrize("h", [pow(2, n) for n in range(0, -10, -1)])
class TestPDifferenceQuotient:
    """
    Unit tests for :py:class:`PDifferenceQuotient`.
    """
    @pytest.mark.parametrize(
        ("f", "ndim", "expected"), zip(
            *zip(*REAL_NFUNCTIONS), [
                lambda h, x: np.array([0]),
                lambda h, x: np.array([0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0]),
                lambda h, x: np.array([0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([1]),
                lambda h, x: np.array([1, 0]),
                lambda h, x: np.array([0, 1]),
                lambda h, x: np.array([1, 0, 0]),
                lambda h, x: np.array([0, 1, 0]),
                lambda h, x: np.array([0, 0, 1]),
                lambda h, x: np.array([1, 1]),
                lambda h, x: np.array([1, 1, 0]),
                lambda h, x: np.array([0, 1, 1]),
                lambda h, x: np.array([1, 0, 1]),
                lambda h, x: np.array([1, 1, 1]),
                lambda h, x: np.array([2 * x[0]]),
                lambda h, x: np.array([2 * x[0], 0]),
                lambda h, x: np.array([0, 2 * x[1]]),
                lambda h, x: np.array([2 * x[0], 0, 0]),
                lambda h, x: np.array([0, 2 * x[1], 0]),
                lambda h, x: np.array([0, 0, 2 * x[2]]),
                lambda h, x: np.array([2 * x[0], 2 * x[1]]),
                lambda h, x: np.array([2 * x[0], 2 * x[1], 0]),
                lambda h, x: np.array([0, 2 * x[1], 2 * x[2]]),
                lambda h, x: np.array([2 * x[0], 0, 2 * x[2]]),
                lambda h, x: np.array([2 * x[0], 2 * x[1], 2 * x[2]]),
                lambda h, x: np.array([x[1], x[0]]),
                lambda h, x: np.array([x[1], x[0], 0]),
                lambda h, x: np.array([0, x[2], x[1]]),
                lambda h, x: np.array([x[2], 0, x[0]]),
                lambda h, x: np.array([x[1] * x[2], x[0] * x[2], x[0] * x[1]]),
                lambda h, x: np.array([2 * x[0] * x[1] ** 2, 2 * x[0] ** 2 * x[1]]),
                lambda h, x: np.array([2 * x[0] * x[1] ** 2, 2 * x[0] ** 2 * x[1], 0]),
                lambda h, x: np.array([0, 2 * x[1] * x[2] ** 2, 2 * x[1] ** 2 * x[2]]),
                lambda h, x: np.array([2 * x[0] * x[2] ** 2, 0, 2 * x[0] ** 2 * x[2]]),
                lambda h, x: np.array(
                    [
                        2 * (x[0] * x[1] * x[2]) * (x[1] * x[2]),
                        2 * (x[0] * x[1] * x[2]) * (x[0] * x[2]),
                        2 * (x[0] * x[1] * x[2]) * (x[0] * x[1])
                    ]
                )
            ]
        )
    )
    def test_first(
        self, f: RealNFunction, ndim: int,
        expected: typing.Callable[[float, np.ndarray], np.ndarray], h: float
    ):
        """
        Unit test for :py:meth:`PDifferenceQuotient.first`.
        """
        diffq = PDifferenceQuotient(f, ndim, h)

        for x in map(np.array, itertools.product(TEST_INTERVAL, repeat=ndim)):
            assert np.equal(diffq.first(x), expected(h, x)).all()

    @pytest.mark.parametrize(
        ("f", "ndim", "expected"), zip(
            *zip(*REAL_NFUNCTIONS), [
                lambda h, x: np.array([0]),
                lambda h, x: np.array([0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0]),
                lambda h, x: np.array([0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0]),
                lambda h, x: np.array([0, 0]),
                lambda h, x: np.array([0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([2]),
                lambda h, x: np.array([2, 0]),
                lambda h, x: np.array([0, 2]),
                lambda h, x: np.array([2, 0, 0]),
                lambda h, x: np.array([0, 2, 0]),
                lambda h, x: np.array([0, 0, 2]),
                lambda h, x: np.array([2, 2]),
                lambda h, x: np.array([2, 2, 0]),
                lambda h, x: np.array([0, 2, 2]),
                lambda h, x: np.array([2, 0, 2]),
                lambda h, x: np.array([2, 2, 2]),
                lambda h, x: np.array([0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([2 * x[1] ** 2, 2 * x[0] ** 2]),
                lambda h, x: np.array([2 * x[1] ** 2, 2 * x[0] ** 2, 0]),
                lambda h, x: np.array([0, 2 * x[2] ** 2, 2 * x[1] ** 2]),
                lambda h, x: np.array([2 * x[2] ** 2, 0, 2 * x[0] ** 2]),
                lambda h, x: np.array(
                    [2 * (x[1] * x[2]) ** 2, 2 * (x[0] * x[2]) ** 2, 2 * (x[0] * x[1]) ** 2]
                )
            ]
        )
    )
    def test_second(
        self, f: RealNFunction, ndim: int,
        expected: typing.Callable[[float, np.ndarray], np.ndarray], h: float
    ):
        """
        Unit test for :py:meth:`PDifferenceQuotient.second`.
        """
        diffq = PDifferenceQuotient(f, ndim, h)

        for x in map(np.array, itertools.product(TEST_INTERVAL, repeat=ndim)):
            assert np.equal(diffq.second(x), expected(h, x)).all()

    @pytest.mark.parametrize(("f", "ndim"), REAL_NFUNCTIONS)
    def test_nth(self, f: RealNFunction, ndim: int, h: float):
        """
        Unit test for :py:meth:`PDifferenceQuotient.nth`.
        """
        diffq = PDifferenceQuotient(f, ndim, h)

        for x in map(np.array, itertools.product(TEST_INTERVAL, repeat=ndim)):
            assert np.equal(diffq.nth(x, 1), diffq.first(x)).all()
            assert np.equal(diffq.nth(x, 2), diffq.second(x)).all()
