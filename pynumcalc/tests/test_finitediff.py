"""
Unit tests for :py:mod:`pynumcalc._finitediff`.
"""

import itertools
import typing

import numpy as np
import pytest

from pynumcalc._finitediff import (
    Forward, Backward, Central, PForward, PBackward, PCentral
)
from pynumcalc._typedef import (
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
class TestForward:
    """
    Unit tests for :py:class:`Forward`.
    """
    @pytest.mark.parametrize(
        ("f", "expected"), zip(
            REAL_FUNCTIONS, [
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: h,
                lambda h, x: -h,
                lambda h, x: 2 * x * h + h ** 2,
                lambda h, x: -2 * x * h + -h ** 2,
                lambda h, x: 3 * x ** 2 * h + 3 * x * h ** 2 + h ** 3,
                lambda h, x: -3 * x ** 2 * h + -3 * x * h ** 2 + -h ** 3
            ]
        )
    )
    def test_first(
        self, f: RealFunction, expected: typing.Callable[[float, float], float], h: float
    ):
        """
        Unit test for :py:meth:`Forward.first`.
        """
        fdiff = Forward(f, h)

        for x in TEST_INTERVAL:
            assert fdiff.first(x) == expected(h, x)

    @pytest.mark.parametrize(
        ("f", "expected"), zip(
            REAL_FUNCTIONS, [
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 2 * h ** 2,
                lambda h, x: -2 * h ** 2,
                lambda h, x: 6 * x * h ** 2 + 6 * h ** 3,
                lambda h, x: -6 * x * h ** 2 + -6 * h ** 3
            ]
        )
    )
    def test_second(
        self, f: RealFunction, expected: typing.Callable[[float, float], float], h: float
    ):
        """
        Unit test for :py:meth:`Forward.second`.
        """
        fdiff = Forward(f, h)

        for x in TEST_INTERVAL:
            assert fdiff.second(x) == expected(h, x)

    @pytest.mark.parametrize("f", REAL_FUNCTIONS)
    def test_nth(self, f: typing.Callable[[float], float], h: float):
        """
        Unit test for :py:meth:`Forward.nth`.
        """
        fdiff = Forward(f, h)

        for x in TEST_INTERVAL:
            assert fdiff.nth(x, 1) == fdiff.first(x)
            assert fdiff.nth(x, 2) == fdiff.second(x)


@pytest.mark.parametrize("h", [pow(2, n) for n in range(0, -10, -1)])
class TestBackward:
    """
    Unit tests for :py:class:`Backward`.
    """
    @pytest.mark.parametrize(
        ("f", "expected"), zip(
            REAL_FUNCTIONS, [
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: h,
                lambda h, x: -h,
                lambda h, x: 2 * x * h + -h ** 2,
                lambda h, x: -2 * x * h + h ** 2,
                lambda h, x: 3 * x ** 2 * h - 3 * x * h ** 2 + h ** 3,
                lambda h, x: -3 * x ** 2 * h + 3 * x * h ** 2 + -h ** 3
            ]
        )
    )
    def test_first(
        self, f: RealFunction, expected: typing.Callable[[float, float], float], h: float
    ):
        """
        Unit test for :py:meth:`Backward.first`.
        """
        fdiff = Backward(f, h)

        for x in TEST_INTERVAL:
            assert fdiff.first(x) == expected(h, x)

    @pytest.mark.parametrize(
        ("f", "expected"), zip(
            REAL_FUNCTIONS, [
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 2 * h ** 2,
                lambda h, x: -2 * h ** 2,
                lambda h, x: 6 * x * h ** 2 - 6 * h ** 3,
                lambda h, x: -6 * x * h ** 2 + 6 * h ** 3
            ]
        )
    )
    def test_second(
        self, f: RealFunction, expected: typing.Callable[[float, float], float], h: float
    ):
        """
        Unit test for :py:meth:`Backward.second`.
        """
        fdiff = Backward(f, h)

        for x in TEST_INTERVAL:
            assert fdiff.second(x) == expected(h, x)

    @pytest.mark.parametrize("f", REAL_FUNCTIONS)
    def test_nth(self, f: typing.Callable[[float], float], h: float):
        """
        Unit test for :py:meth:`Backward.nth`.
        """
        fdiff = Backward(f, h)

        for x in TEST_INTERVAL:
            assert fdiff.nth(x, 1) == fdiff.first(x)
            assert fdiff.nth(x, 2) == fdiff.second(x)


@pytest.mark.parametrize("h", [pow(2, n) for n in range(0, -10, -1)])
class TestCentral:
    """
    Unit tests for :py:class:`Central`.
    """
    @pytest.mark.parametrize(
        ("f", "expected"), zip(
            REAL_FUNCTIONS, [
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: h,
                lambda h, x: -h,
                lambda h, x: 2 * x * h,
                lambda h, x: -2 * x * h,
                lambda h, x: 3 * x ** 2 * h + h ** 3 / 4,
                lambda h, x: -3 * x ** 2 * h + -h ** 3 / 4
            ]
        )
    )
    def test_first(
        self, f: RealFunction, expected: typing.Callable[[float, float], float], h: float
    ):
        """
        Unit test for :py:meth:`Central.first`.
        """
        fdiff = Central(f, h)

        for x in TEST_INTERVAL:
            assert fdiff.first(x) == expected(h, x)

    @pytest.mark.parametrize(
        ("f", "expected"), zip(
            REAL_FUNCTIONS, [
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 0,
                lambda h, x: 2 * h ** 2,
                lambda h, x: -2 * h ** 2,
                lambda h, x: 6 * x * h ** 2,
                lambda h, x: -6 * x * h ** 2
            ]
        )
    )
    def test_second(
        self, f: RealFunction, expected: typing.Callable[[float, float], float], h: float
    ):
        """
        Unit test for :py:meth:`Central.second`.
        """
        fdiff = Central(f, h)

        for x in TEST_INTERVAL:
            assert fdiff.second(x) == expected(h, x)

    @pytest.mark.parametrize("f", REAL_FUNCTIONS)
    def test_nth(self, f: RealFunction, h: float):
        """
        Unit test for :py:meth:`Central.nth`.
        """
        fdiff = Central(f, h)

        for x in TEST_INTERVAL:
            assert fdiff.nth(x, 1) == fdiff.first(x)
            assert fdiff.nth(x, 2) == fdiff.second(x)


@pytest.mark.parametrize("h", [pow(2, n) for n in range(0, -10, -1)])
class TestPForward:
    """
    Unit tests for :py:class:`PForward`.
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
                lambda h, x: np.array([h]),
                lambda h, x: np.array([h, 0]),
                lambda h, x: np.array([0, h]),
                lambda h, x: np.array([h, 0, 0]),
                lambda h, x: np.array([0, h, 0]),
                lambda h, x: np.array([0, 0, h]),
                lambda h, x: np.array([h, h]),
                lambda h, x: np.array([h, h, 0]),
                lambda h, x: np.array([0, h, h]),
                lambda h, x: np.array([h, 0, h]),
                lambda h, x: np.array([h, h, h]),
                lambda h, x: np.array([2 * x[0] * h + h ** 2]),
                lambda h, x: np.array([2 * x[0] * h + h ** 2, 0]),
                lambda h, x: np.array([0, 2 * x[1] * h + h ** 2]),
                lambda h, x: np.array([2 * x[0] * h + h ** 2, 0, 0]),
                lambda h, x: np.array([0, 2 * x[1] * h + h ** 2, 0]),
                lambda h, x: np.array([0, 0, 2 * x[2] * h + h ** 2]),
                lambda h, x: np.array([2 * x[0] * h + h ** 2, 2 * x[1] * h + h ** 2]),
                lambda h, x: np.array([2 * x[0] * h + h ** 2, 2 * x[1] * h + h ** 2, 0]),
                lambda h, x: np.array([0, 2 * x[1] * h + h ** 2, 2 * x[2] * h + h ** 2]),
                lambda h, x: np.array([2 * x[0] * h + h ** 2, 0, 2 * x[2] * h + h ** 2]),
                lambda h, x: np.array(
                    [2 * x[0] * h + h ** 2, 2 * x[1] * h + h ** 2, 2 * x[2] * h + h ** 2]
                ),
                lambda h, x: np.array([x[1] * h, x[0] * h]),
                lambda h, x: np.array([x[1] * h, x[0] * h, 0]),
                lambda h, x: np.array([0, x[2] * h, x[1] * h]),
                lambda h, x: np.array([x[2] * h, 0, x[0] * h]),
                lambda h, x: np.array([x[1] * x[2] * h, x[0] * x[2] * h, x[0] * x[1] * h]),
                lambda h, x: np.array(
                    [
                        2 * x[0] * x[1] ** 2 * h + (h * x[1]) ** 2,
                        2 * x[0] ** 2 * x[1] * h + (h * x[0]) ** 2
                    ]
                ),
                lambda h, x: np.array(
                    [
                        2 * x[0] * x[1] ** 2 * h + (h * x[1]) ** 2,
                        2 * x[0] ** 2 * x[1] * h + (h * x[0]) ** 2,
                        0
                    ]
                ),
                lambda h, x: np.array(
                    [
                        0,
                        2 * x[1] * x[2] ** 2 * h + (h * x[2]) ** 2,
                        2 * x[1] ** 2 * x[2] * h + (h * x[1]) ** 2
                    ]
                ),
                lambda h, x: np.array(
                    [
                        2 * x[0] * x[2] ** 2 * h + (h * x[2]) ** 2,
                        0,
                        2 * x[0] ** 2 * x[2] * h + (h * x[0]) ** 2
                    ]
                ),
                lambda h, x: np.array(
                    [
                        2 * (x[0] * x[1] * x[2]) * (x[1] * x[2] * h) + (x[1] * x[2] * h) ** 2,
                        2 * (x[0] * x[1] * x[2]) * (x[0] * x[2] * h) + (x[0] * x[2] * h) ** 2,
                        2 * (x[0] * x[1] * x[2]) * (x[0] * x[1] * h) + (x[0] * x[1] * h) ** 2
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
        Unit test for :py:meth:`PForward.first`.
        """
        fdiff = PForward(f, ndim, h)

        for x in map(np.array, itertools.product(TEST_INTERVAL, repeat=ndim)):
            assert np.equal(fdiff.first(x), expected(h, x)).all()

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
                lambda h, x: np.array([2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 0]),
                lambda h, x: np.array([0, 2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 0, 0]),
                lambda h, x: np.array([0, 2 * h ** 2, 0]),
                lambda h, x: np.array([0, 0, 2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 2 * h ** 2, 0]),
                lambda h, x: np.array([0, 2 * h ** 2, 2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 0, 2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 2 * h ** 2, 2 * h ** 2]),
                lambda h, x: np.array([0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([2 * (h * x[1]) ** 2, 2 * (h * x[0]) ** 2]),
                lambda h, x: np.array([2 * (h * x[1]) ** 2, 2 * (h * x[0]) ** 2, 0]),
                lambda h, x: np.array([0, 2 * (h * x[2]) ** 2, 2 * (h * x[1]) ** 2]),
                lambda h, x: np.array([2 * (h * x[2]) ** 2, 0, 2 * (h * x[0]) ** 2]),
                lambda h, x: np.array(
                    [
                        2 * (x[1] * x[2] * h) ** 2,
                        2 * (x[0] * x[2] * h) ** 2,
                        2 * (x[0] * x[1] * h) ** 2
                    ]
                )
            ]
        )
    )
    def test_second(
        self, f: RealNFunction, ndim: int,
        expected: typing.Callable[[float, np.ndarray], np.ndarray], h: float
    ):
        """
        Unit test for :py:meth:`PForward.second`.
        """
        fdiff = PForward(f, ndim, h)

        for x in map(np.array, itertools.product(TEST_INTERVAL, repeat=ndim)):
            assert np.equal(fdiff.second(x), expected(h, x)).all()

    @pytest.mark.parametrize(("f", "ndim"), REAL_NFUNCTIONS)
    def test_nth(self, f: RealNFunction, ndim: int, h: float):
        """
        Unit test for :py:meth:`PForward.nth`.
        """
        fdiff = PForward(f, ndim, h)

        for x in map(np.array, itertools.product(TEST_INTERVAL, repeat=ndim)):
            assert np.equal(fdiff.nth(x, 1), fdiff.first(x)).all()
            assert np.equal(fdiff.nth(x, 2), fdiff.second(x)).all()


@pytest.mark.parametrize("h", [pow(2, n) for n in range(0, -10, -1)])
class TestPBackward:
    """
    Unit tests for :py:class:`PBackward`.
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
                lambda h, x: np.array([h]),
                lambda h, x: np.array([h, 0]),
                lambda h, x: np.array([0, h]),
                lambda h, x: np.array([h, 0, 0]),
                lambda h, x: np.array([0, h, 0]),
                lambda h, x: np.array([0, 0, h]),
                lambda h, x: np.array([h, h]),
                lambda h, x: np.array([h, h, 0]),
                lambda h, x: np.array([0, h, h]),
                lambda h, x: np.array([h, 0, h]),
                lambda h, x: np.array([h, h, h]),
                lambda h, x: np.array([2 * x[0] * h + -h ** 2]),
                lambda h, x: np.array([2 * x[0] * h + -h ** 2, 0]),
                lambda h, x: np.array([0, 2 * x[1] * h + -h ** 2]),
                lambda h, x: np.array([2 * x[0] * h + -h ** 2, 0, 0]),
                lambda h, x: np.array([0, 2 * x[1] * h + -h ** 2, 0]),
                lambda h, x: np.array([0, 0, 2 * x[2] * h + -h ** 2]),
                lambda h, x: np.array([2 * x[0] * h + -h ** 2, 2 * x[1] * h + -h ** 2]),
                lambda h, x: np.array([2 * x[0] * h + -h ** 2, 2 * x[1] * h + -h ** 2, 0]),
                lambda h, x: np.array([0, 2 * x[1] * h + -h ** 2, 2 * x[2] * h + -h ** 2]),
                lambda h, x: np.array([2 * x[0] * h + -h ** 2, 0, 2 * x[2] * h + -h ** 2]),
                lambda h, x: np.array(
                    [2 * x[0] * h + -h ** 2, 2 * x[1] * h + -h ** 2, 2 * x[2] * h + -h ** 2]
                ),
                lambda h, x: np.array([x[1] * h, x[0] * h]),
                lambda h, x: np.array([x[1] * h, x[0] * h, 0]),
                lambda h, x: np.array([0, x[2] * h, x[1] * h]),
                lambda h, x: np.array([x[2] * h, 0, x[0] * h]),
                lambda h, x: np.array([x[1] * x[2] * h, x[0] * x[2] * h, x[0] * x[1] * h]),
                lambda h, x: np.array(
                    [
                        2 * x[0] * x[1] ** 2 * h + -(h * x[1]) ** 2,
                        2 * x[0] ** 2 * x[1] * h + -(h * x[0]) ** 2
                    ]
                ),
                lambda h, x: np.array(
                    [
                        2 * x[0] * x[1] ** 2 * h + -(h * x[1]) ** 2,
                        2 * x[0] ** 2 * x[1] * h + -(h * x[0]) ** 2,
                        0
                    ]
                ),
                lambda h, x: np.array(
                    [
                        0,
                        2 * x[1] * x[2] ** 2 * h + -(h * x[2]) ** 2,
                        2 * x[1] ** 2 * x[2] * h + -(h * x[1]) ** 2
                    ]
                ),
                lambda h, x: np.array(
                    [
                        2 * x[0] * x[2] ** 2 * h + -(h * x[2]) ** 2,
                        0,
                        2 * x[0] ** 2 * x[2] * h + -(h * x[0]) ** 2
                    ]
                ),
                lambda h, x: np.array(
                    [
                        2 * (x[0] * x[1] * x[2]) * (x[1] * x[2] * h) + -(x[1] * x[2] * h) ** 2,
                        2 * (x[0] * x[1] * x[2]) * (x[0] * x[2] * h) + -(x[0] * x[2] * h) ** 2,
                        2 * (x[0] * x[1] * x[2]) * (x[0] * x[1] * h) + -(x[0] * x[1] * h) ** 2
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
        Unit test for :py:meth:`PBackward.first`.
        """
        fdiff = PBackward(f, ndim, h)

        for x in map(np.array, itertools.product(TEST_INTERVAL, repeat=ndim)):
            assert np.equal(fdiff.first(x), expected(h, x)).all()

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
                lambda h, x: np.array([2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 0]),
                lambda h, x: np.array([0, 2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 0, 0]),
                lambda h, x: np.array([0, 2 * h ** 2, 0]),
                lambda h, x: np.array([0, 0, 2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 2 * h ** 2, 0]),
                lambda h, x: np.array([0, 2 * h ** 2, 2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 0, 2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 2 * h ** 2, 2 * h ** 2]),
                lambda h, x: np.array([0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([2 * (h * x[1]) ** 2, 2 * (h * x[0]) ** 2]),
                lambda h, x: np.array([2 * (h * x[1]) ** 2, 2 * (h * x[0]) ** 2, 0]),
                lambda h, x: np.array([0, 2 * (h * x[2]) ** 2, 2 * (h * x[1]) ** 2]),
                lambda h, x: np.array([2 * (h * x[2]) ** 2, 0, 2 * (h * x[0]) ** 2]),
                lambda h, x: np.array(
                    [
                        2 * (x[1] * x[2] * h) ** 2,
                        2 * (x[0] * x[2] * h) ** 2,
                        2 * (x[0] * x[1] * h) ** 2
                    ]
                )
            ]
        )
    )
    def test_second(
        self, f: RealNFunction, ndim: int,
        expected: typing.Callable[[float, np.ndarray], np.ndarray], h: float
    ):
        """
        Unit test for :py:meth:`PBackward.second`.
        """
        fdiff = PBackward(f, ndim, h)

        for x in map(np.array, itertools.product(TEST_INTERVAL, repeat=ndim)):
            assert np.equal(fdiff.second(x), expected(h, x)).all()

    @pytest.mark.parametrize(("f", "ndim"), REAL_NFUNCTIONS)
    def test_nth(self, f: RealNFunction, ndim: int, h: float):
        """
        Unit test for :py:meth:`PBackward.nth`.
        """
        fdiff = PBackward(f, ndim, h)

        for x in map(np.array, itertools.product(TEST_INTERVAL, repeat=ndim)):
            assert np.equal(fdiff.nth(x, 1), fdiff.first(x)).all()
            assert np.equal(fdiff.nth(x, 2), fdiff.second(x)).all()


@pytest.mark.parametrize("h", [pow(2, n) for n in range(0, -10, -1)])
class TestPCentral:
    """
    Unit tests for :py:class:`PCentral`.
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
                lambda h, x: np.array([h]),
                lambda h, x: np.array([h, 0]),
                lambda h, x: np.array([0, h]),
                lambda h, x: np.array([h, 0, 0]),
                lambda h, x: np.array([0, h, 0]),
                lambda h, x: np.array([0, 0, h]),
                lambda h, x: np.array([h, h]),
                lambda h, x: np.array([h, h, 0]),
                lambda h, x: np.array([0, h, h]),
                lambda h, x: np.array([h, 0, h]),
                lambda h, x: np.array([h, h, h]),
                lambda h, x: np.array([2 * x[0] * h]),
                lambda h, x: np.array([2 * x[0] * h, 0]),
                lambda h, x: np.array([0, 2 * x[1] * h]),
                lambda h, x: np.array([2 * x[0] * h, 0, 0]),
                lambda h, x: np.array([0, 2 * x[1] * h, 0]),
                lambda h, x: np.array([0, 0, 2 * x[2] * h]),
                lambda h, x: np.array([2 * x[0] * h, 2 * x[1] * h]),
                lambda h, x: np.array([2 * x[0] * h, 2 * x[1] * h, 0]),
                lambda h, x: np.array([0, 2 * x[1] * h, 2 * x[2] * h]),
                lambda h, x: np.array([2 * x[0] * h, 0, 2 * x[2] * h]),
                lambda h, x: np.array([2 * x[0] * h, 2 * x[1] * h, 2 * x[2] * h]),
                lambda h, x: np.array([x[1] * h, x[0] * h]),
                lambda h, x: np.array([x[1] * h, x[0] * h, 0]),
                lambda h, x: np.array([0, x[2] * h, x[1] * h]),
                lambda h, x: np.array([x[2] * h, 0, x[0] * h]),
                lambda h, x: np.array([x[1] * x[2] * h, x[0] * x[2] * h, x[0] * x[1] * h]),
                lambda h, x: np.array([2 * x[0] * x[1] ** 2 * h, 2 * x[0] ** 2 * x[1] * h]),
                lambda h, x: np.array([2 * x[0] * x[1] ** 2 * h, 2 * x[0] ** 2 * x[1] * h, 0]),
                lambda h, x: np.array([0, 2 * x[1] * x[2] ** 2 * h, 2 * x[1] ** 2 * x[2] * h]),
                lambda h, x: np.array([2 * x[0] * x[2] ** 2 * h, 0, 2 * x[0] ** 2 * x[2] * h]),
                lambda h, x: np.array(
                    [
                        2 * (x[0] * x[1] * x[2]) * (x[1] * x[2] * h),
                        2 * (x[0] * x[1] * x[2]) * (x[0] * x[2] * h),
                        2 * (x[0] * x[1] * x[2]) * (x[0] * x[1] * h)
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
        Unit test for :py:meth:`PCentral.first`.
        """
        fdiff = PCentral(f, ndim, h)

        for x in map(np.array, itertools.product(TEST_INTERVAL, repeat=ndim)):
            assert np.equal(fdiff.first(x), expected(h, x)).all()

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
                lambda h, x: np.array([2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 0]),
                lambda h, x: np.array([0, 2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 0, 0]),
                lambda h, x: np.array([0, 2 * h ** 2, 0]),
                lambda h, x: np.array([0, 0, 2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 2 * h ** 2, 0]),
                lambda h, x: np.array([0, 2 * h ** 2, 2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 0, 2 * h ** 2]),
                lambda h, x: np.array([2 * h ** 2, 2 * h ** 2, 2 * h ** 2]),
                lambda h, x: np.array([0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([0, 0, 0]),
                lambda h, x: np.array([2 * (h * x[1]) ** 2, 2 * (h * x[0]) ** 2]),
                lambda h, x: np.array([2 * (h * x[1]) ** 2, 2 * (h * x[0]) ** 2, 0]),
                lambda h, x: np.array([0, 2 * (h * x[2]) ** 2, 2 * (h * x[1]) ** 2]),
                lambda h, x: np.array([2 * (h * x[2]) ** 2, 0, 2 * (h * x[0]) ** 2]),
                lambda h, x: np.array(
                    [
                        2 * (x[1] * x[2] * h) ** 2,
                        2 * (x[0] * x[2] * h) ** 2,
                        2 * (x[0] * x[1] * h) ** 2
                    ]
                )
            ]
        )
    )
    def test_second(
        self, f: RealNFunction, ndim: int,
        expected: typing.Callable[[float, np.ndarray], np.ndarray], h: float
    ):
        """
        Unit test for :py:meth:`PCentral.second`.
        """
        fdiff = PCentral(f, ndim, h)

        for x in map(np.array, itertools.product(TEST_INTERVAL, repeat=ndim)):
            assert np.equal(fdiff.second(x), expected(h, x)).all()

    @pytest.mark.parametrize(("f", "ndim"), REAL_NFUNCTIONS)
    def test_nth(self, f: RealNFunction, ndim: int, h: float):
        """
        Unit test for :py:meth:`PCentral.nth`.
        """
        fdiff = PCentral(f, ndim, h)

        for x in map(np.array, itertools.product(TEST_INTERVAL, repeat=ndim)):
            assert np.equal(fdiff.nth(x, 1), fdiff.first(x)).all()
            assert np.equal(fdiff.nth(x, 2), fdiff.second(x)).all()
