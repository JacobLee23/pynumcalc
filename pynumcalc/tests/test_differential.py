"""
Unit tests for :py:mod:`pynumcalc.differential`.
"""

import itertools
import typing

import numpy as np
import pytest

from pynumcalc import differential


LOWER, UPPER = -10, 10
TEST_INTERVAL = range(LOWER, UPPER + 1)


@pytest.mark.parametrize(
    "h", [2 ** n for n in range(0, -10, -1)]
)
class TestFiniteDifference:
    """
    Unit tests for :py:class:`differential.FiniteDifference`.
    """
    @pytest.mark.parametrize(
        ("f", "expected"), [
            (lambda x: 0, lambda h: (lambda x: 0)),
            (lambda x: 1, lambda h: (lambda x: 0)),
            (lambda x: -1, lambda h: (lambda x: 0)),
            (lambda x: x, lambda h: (lambda x: h)),
            (lambda x: -x, lambda h: (lambda x: -h)),

            (lambda x: x ** 2, lambda h: (lambda x: 2 * x * h + h ** 2)),
            (lambda x: -x ** 2, lambda h: (lambda x: -2 * x * h - h ** 2)),
            (lambda x: x ** 3, lambda h: (lambda x: 3 * x ** 2 * h + 3 * x * h ** 2 + h ** 3)),
            (lambda x: -x ** 3, lambda h: (lambda x: -3 * x ** 2 * h + -3 * x * h ** 2 + -h ** 3)),
            (lambda x: (-x) ** 3, lambda h: (lambda x: -3 * x ** 2 * h + -3 * x * h ** 2 + -h ** 3))
        ]
    )
    def test_forward(
        self, f: typing.Callable[[float], float], h: float,
        expected: typing.Callable[[float], typing.Callable[[float], float]]
    ):
        """
        Unit test for :py:`differential.FiniteDifference.forward`.
        """
        fdiff, xfdiff = differential.FiniteDifference.forward(f, h), expected(h)

        for x in TEST_INTERVAL:
            assert fdiff(x) == xfdiff(x)

    @pytest.mark.parametrize(
        ("f", "expected"), [
            (lambda x: 0, lambda h: (lambda x: 0)),
            (lambda x: 1, lambda h: (lambda x: 0)),
            (lambda x: -1, lambda h: (lambda x: 0)),
            (lambda x: x, lambda h: (lambda x: 0)),
            (lambda x: -x, lambda h: (lambda x: 0)),

            (lambda x: x ** 2, lambda h: (lambda x: 2 * h ** 2)),
            (lambda x: -x ** 2, lambda h: (lambda x: -2 * h ** 2)),
            (lambda x: x ** 3, lambda h: (lambda x: 6 * x * h ** 2 + 6 * h ** 3)),
            (lambda x: -x ** 3, lambda h: (lambda x: -6 * x * h ** 2 + -6 * h ** 3)),
            (lambda x: (-x) ** 3, lambda h: (lambda x: -6 * x * h ** 2 + -6 * h ** 3))
        ]
    )
    def test_forward2(
        self, f: typing.Callable[[float], float], h: float,
        expected: typing.Callable[[float], typing.Callable[[float], float]]
    ):
        """
        Unit test for :py:`differential.FiniteDifference.forward2`.
        """
        fdiff, xfdiff = differential.FiniteDifference.forward2(f, h), expected(h)

        for x in TEST_INTERVAL:
            assert fdiff(x) == xfdiff(x)

    @pytest.mark.parametrize(
        "f", [
            lambda x: 0, lambda x: 1, lambda x: -1, lambda x: x, lambda x: -x,
            lambda x: x ** 2, lambda x: -x ** 3, lambda x: x ** 3, lambda x: -x ** 3, lambda x: (-x) ** 3
        ]
    )
    def test_forwardn(self, f: typing.Callable[[float], float], h: float):
        """
        Unit test for :py:`differential.FiniteDifference.forwardn`.
        """
        fdiff1 = differential.FiniteDifference.forwardn(f, h, 1)
        xfdiff1 = differential.FiniteDifference.forward(f, h)

        fdiff2 = differential.FiniteDifference.forwardn(f, h, 2)
        xfdiff2 = differential.FiniteDifference.forward2(f, h)

        for x in TEST_INTERVAL:
            assert fdiff1(x) == xfdiff1(x), (h, x, 1)
            assert fdiff2(x) == xfdiff2(x), (h, x, 2)

    @pytest.mark.parametrize(
        ("f", "expected"), [
            (lambda x: 0, lambda h: (lambda x: 0)),
            (lambda x: 1, lambda h: (lambda x: 0)),
            (lambda x: -1, lambda h: (lambda x: 0)),
            (lambda x: x, lambda h: (lambda x: h)),
            (lambda x: -x, lambda h: (lambda x: -h)),

            (lambda x: x ** 2, lambda h: (lambda x: 2 * x * h - h ** 2)),
            (lambda x: -x ** 2, lambda h: (lambda x: -2 * x * h + h ** 2)),
            (lambda x: x ** 3, lambda h: (lambda x: 3 * x ** 2 * h - 3 * x * h ** 2 + h ** 3)),
            (lambda x: -x ** 3, lambda h: (lambda x: -3 * x ** 2 * h + 3 * x * h ** 2 - h ** 3)),
            (lambda x: (-x) ** 3, lambda h: (lambda x: -3 * x ** 2 * h + 3 * x * h ** 2 - h ** 3))
        ]
    )
    def test_backward(
        self, f: typing.Callable[[float], float], h: float,
        expected: typing.Callable[[float], typing.Callable[[float], float]]
    ):
        """
        Unit test for :py:`differential.FiniteDifference.backward`.
        """
        fdiff, xfdiff = differential.FiniteDifference.backward(f, h), expected(h)

        for x in TEST_INTERVAL:
            assert fdiff(x) == xfdiff(x)

    @pytest.mark.parametrize(
        ("f", "expected"), [
            (lambda x: 0, lambda h: (lambda x: 0)),
            (lambda x: 1, lambda h: (lambda x: 0)),
            (lambda x: -1, lambda h: (lambda x: 0)),
            (lambda x: x, lambda h: (lambda x: 0)),
            (lambda x: -x, lambda h: (lambda x: 0)),

            (lambda x: x ** 2, lambda h: (lambda x: 2 * h ** 2)),
            (lambda x: -x ** 2, lambda h: (lambda x: -2 * h ** 2)),
            (lambda x: x ** 3, lambda h: (lambda x: 6 * x * h ** 2 - 6 * h ** 3)),
            (lambda x: -x ** 3, lambda h: (lambda x: -6 * x * h ** 2 + 6 * h ** 3)),
            (lambda x: (-x) ** 3, lambda h: (lambda x: -6 * x * h ** 2 + 6 * h ** 3))
        ]
    )
    def test_backward2(
        self, f: typing.Callable[[float], float], h: float,
        expected: typing.Callable[[float], typing.Callable[[float], float]]
    ):
        """
        Unit test for :py:`differential.FiniteDifference.backward2`.
        """
        fdiff, xfdiff = differential.FiniteDifference.backward2(f, h), expected(h)

        for x in TEST_INTERVAL:
            assert fdiff(x) == xfdiff(x)

    @pytest.mark.parametrize(
        "f", [
            lambda x: 0, lambda x: 1, lambda x: -1, lambda x: x, lambda x: -x,
            lambda x: x ** 2, lambda x: -x ** 3, lambda x: x ** 3, lambda x: -x ** 3, lambda x: (-x) ** 3
        ]
    )
    def test_backwardn(self, f: typing.Callable[[float], float], h: float):
        """
        Unit test for :py:`differential.FiniteDifference.backwardn`.
        """
        fdiff1 = differential.FiniteDifference.backwardn(f, h, 1)
        xfdiff1 = differential.FiniteDifference.backward(f, h)

        fdiff2 = differential.FiniteDifference.backwardn(f, h, 2)
        xfdiff2 = differential.FiniteDifference.backward2(f, h)

        for x in TEST_INTERVAL:
            assert fdiff1(x) == xfdiff1(x)
            assert fdiff2(x) == xfdiff2(x)

    @pytest.mark.parametrize(
        ("f", "expected"), [
            (lambda x: 0, lambda h: (lambda x: 0)),
            (lambda x: 1, lambda h: (lambda x: 0)),
            (lambda x: -1, lambda h: (lambda x: 0)),
            (lambda x: x, lambda h: (lambda x: h)),
            (lambda x: -x, lambda h: (lambda x: -h)),

            (lambda x: x ** 2, lambda h: (lambda x: 2 * x * h)),
            (lambda x: -x ** 2, lambda h: (lambda x: -2 * x * h)),
            (lambda x: x ** 3, lambda h: (lambda x: 3 * x ** 2 * h + h ** 3 / 4)),
            (lambda x: -x ** 3, lambda h: (lambda x: -3 * x ** 2 * h - h ** 3 / 4)),
            (lambda x: (-x) ** 3, lambda h: (lambda x: -3 * x ** 2 * h - h ** 3 / 4))
        ]
    )
    def test_central(
        self, f: typing.Callable[[float], float], h: float,
        expected: typing.Callable[[float], typing.Callable[[float], float]]
    ):
        """
        Unit test for :py:`differential.FiniteDifference.central`.
        """
        fdiff, xfdiff = differential.FiniteDifference.central(f, h), expected(h)

        for x in TEST_INTERVAL:
            assert fdiff(x) == xfdiff(x)

    @pytest.mark.parametrize(
        ("f", "expected"), [
            (lambda x: 0, lambda h: (lambda x: 0)),
            (lambda x: 1, lambda h: (lambda x: 0)),
            (lambda x: -1, lambda h: (lambda x: 0)),
            (lambda x: x, lambda h: (lambda x: 0)),
            (lambda x: -x, lambda h: (lambda x: 0)),

            (lambda x: x ** 2, lambda h: (lambda x: 2 * h ** 2)),
            (lambda x: -x ** 2, lambda h: (lambda x: -2 * h ** 2)),
            (lambda x: x ** 3, lambda h: (lambda x: 6 * x * h ** 2)),
            (lambda x: -x ** 3, lambda h: (lambda x: -6 * x * h ** 2)),
            (lambda x: (-x) ** 3, lambda h: (lambda x: -6 * x * h ** 2))
        ]
    )
    def test_central2(
        self, f: typing.Callable[[float], float], h: float,
        expected: typing.Callable[[float], typing.Callable[[float], float]]
    ):
        """
        Unit test for :py:`differential.FiniteDifference.central2`.
        """
        fdiff, xfdiff = differential.FiniteDifference.central2(f, h), expected(h)

        for x in TEST_INTERVAL:
            assert fdiff(x) == xfdiff(x)

    @pytest.mark.parametrize(
        "f", [
            lambda x: 0, lambda x: 1, lambda x: -1, lambda x: x, lambda x: -x,
            lambda x: x ** 2, lambda x: -x ** 3, lambda x: x ** 3, lambda x: -x ** 3, lambda x: (-x) ** 3
        ]
    )
    def test_centraln(self, f: typing.Callable[[float], float], h: float):
        """
        Unit test for :py:`differential.FiniteDifference.centraln`.
        """
        fdiff1 = differential.FiniteDifference.centraln(f, h, 1)
        xfdiff1 = differential.FiniteDifference.central(f, h)

        fdiff2 = differential.FiniteDifference.centraln(f, h, 2)
        xfdiff2 = differential.FiniteDifference.central2(f, h)

        for x in TEST_INTERVAL:
            assert fdiff1(x) == xfdiff1(x)
            assert fdiff2(x) == xfdiff2(x)


@pytest.mark.parametrize(
    "h", [2 ** n for n in range(0, -10, -1)]
)
class TestPFiniteDifference:
    """
    Unit tests for :py:class:`differential.PFiniteDifference`.
    """
    @pytest.mark.parametrize(
        ("f", "dim", "expected"), [
            (lambda x: 0, 1, lambda h: [lambda x: 0]),
            (lambda x: 0, 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: 0, 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),

            (lambda x: 1, 1, lambda h: [lambda x: 0]),
            (lambda x: 1, 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: 1, 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),

            (lambda x: x[0], 1, lambda h: [lambda x: h]),
            (lambda x: x[0], 2, lambda h: [lambda x: h, lambda x: 0]),
            (lambda x: x[1], 2, lambda h: [lambda x: 0, lambda x: h]),
            (lambda x: x[0] + x[1], 2, lambda h: [lambda x: h, lambda x: h]),
            (lambda x: x[0] + x[1] + x[2], 3, lambda h: [lambda x: h, lambda x: h, lambda x: h]),

            (lambda x: x[0] ** 2, 1, lambda h: [lambda x: 2 * x[0] * h + h ** 2]),
            (lambda x: x[0] ** 2, 2, lambda h: [lambda x: 2 * x[0] * h + h ** 2, lambda x: 0]),
            (lambda x: x[1] ** 2, 2, lambda h: [lambda x: 0, lambda x: 2 * x[1] * h + h ** 2]),
            (lambda x: x[0] ** 2 + x[1] ** 2, 2, lambda h: [
                lambda x: 2 * x[0] * h + h ** 2,
                lambda x: 2 * x[1] * h + h ** 2
            ]),
            (lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2, 3, lambda h: [
                lambda x: 2 * x[0] * h + h ** 2,
                lambda x: 2 * x[1] * h + h ** 2,
                lambda x: 2 * x[2] * h + h ** 2
            ]),

            (lambda x: x[0] * x[1], 2, lambda h: [lambda x: h * x[1], lambda x: h * x[0]]),
            (lambda x: (x[0] * x[1]) ** 2, 2, lambda h: [
                lambda x: 2 * x[0] * x[1] ** 2 * h + (h * x[1]) ** 2,
                lambda x: 2 * x[0] ** 2 * x[1] * h + (h * x[0]) ** 2
            ]),
            (lambda x: x[0] * x[1] * x[2], 3, lambda h: [
                lambda x: h * x[1] * x[2],
                lambda x: h * x[0] * x[2],
                lambda x: h * x[0] * x[1]
            ]),
            (lambda x: (x[0] * x[1] * x[2]) ** 2, 3, lambda h: [
                lambda x: 2 * (x[0] * x[1] * x[2]) * (x[1] * x[2] * h) + (x[1] * x[2] * h) ** 2,
                lambda x: 2 * (x[0] * x[1] * x[2]) * (x[0] * x[2] * h) + (x[0] * x[2] * h) ** 2,
                lambda x: 2 * (x[0] * x[1] * x[2]) * (x[0] * x[1] * h) + (x[0] * x[1] * h) ** 2
            ])
        ]
    )
    def test_pforward(
        self, f: typing.Callable[[typing.Sequence[float]], float], h: float, dim: int,
        expected: typing.Callable[
            [float], typing.Sequence[typing.Callable[[typing.Sequence[float]], float]]
        ]
    ):
        """
        Unit tests for :py:`differential.PFiniteDifference.pforward`.
        """
        fdiff, xfdiff = differential.PFiniteDifference.pforward(f, h, dim), expected(h)

        for ndim in range(dim):
            for x in itertools.product(TEST_INTERVAL, repeat=dim):
                assert fdiff(x, ndim=ndim) == xfdiff[ndim](x)

    @pytest.mark.parametrize(
        ("f", "dim", "expected"), [
            (lambda x: 0, 1, lambda h: [lambda x: 0]),
            (lambda x: 0, 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: 0, 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),

            (lambda x: 1, 1, lambda h: [lambda x: 0]),
            (lambda x: 1, 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: 1, 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),

            (lambda x: x[0], 1, lambda h: [lambda x: 0]),
            (lambda x: x[0], 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: x[1], 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: x[0] + x[1], 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: x[0] + x[1] + x[2], 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),

            (lambda x: x[0] ** 2, 1, lambda h: [lambda x: 2 * h ** 2]),
            (lambda x: x[0] ** 2, 2, lambda h: [lambda x: 2 * h ** 2, lambda x: 0]),
            (lambda x: x[1] ** 2, 2, lambda h: [lambda x: 0, lambda x: 2 * h ** 2]),
            (lambda x: x[0] ** 2 + x[1] ** 2, 2, lambda h: [
                lambda x: 2 * h ** 2, lambda x: 2 * h ** 2
            ]),
            (lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2, 3, lambda h: [
                lambda x: 2 * h ** 2, lambda x: 2 * h ** 2, lambda x: 2 * h ** 2
            ]),

            (lambda x: x[0] * x[1], 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: (x[0] * x[1]) ** 2, 2, lambda h: [
                lambda x: 2 * (x[1] * h) ** 2, lambda x: 2 * (x[0] * h) ** 2
            ]),
            (lambda x: x[0] * x[1] * x[2], 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),
            (lambda x: (x[0] * x[1] * x[2]) ** 2, 3, lambda h: [
                lambda x: 2 * (x[1] * x[2] * h) ** 2,
                lambda x: 2 * (x[0] * x[2] * h) ** 2,
                lambda x: 2 * (x[0] * x[1] * h) ** 2
            ])
        ]
    )
    def test_pforward2(
        self, f: typing.Callable[[typing.Sequence[float]], float], h: float, dim: int,
        expected: typing.Callable[
            [float], typing.Sequence[typing.Callable[[typing.Sequence[float]], float]]
        ]
    ):
        """
        Unit tests for :py:`differential.PFiniteDifference.pforward2`.
        """
        fdiff, xfdiff = differential.PFiniteDifference.pforward2(f, h, dim), expected(h)

        for ndim in range(dim):
            for x in itertools.product(TEST_INTERVAL, repeat=dim):
                assert fdiff(x, ndim=ndim) == xfdiff[ndim](x)

    @pytest.mark.parametrize(
        ("f", "dim"), [
            (lambda x: 0, 1), (lambda x: 0, 2), (lambda x: 0, 3),
            (lambda x: 1, 1), (lambda x: 1, 2), (lambda x: 1, 3),

            (lambda x: x[0], 1), (lambda x: x[0], 2), (lambda x: x[0], 3),
            (lambda x: x[0] + x[1], 2), (lambda x: x[0] + x[1] + x[2], 3)
        ]
    )
    def test_pforwardn(
        self, f: typing.Callable[[typing.Sequence[float]], float], h: float, dim: int
    ):
        """
        Unit test for :py:`differential.PFiniteDifference.pforwardn`.
        """
        fdiff1 = differential.PFiniteDifference.pforwardn(f, h, dim, 1)
        xfdiff1 = differential.PFiniteDifference.pforward(f, h, dim)

        fdiff2 = differential.PFiniteDifference.pforwardn(f, h, dim, 2)
        xfdiff2 = differential.PFiniteDifference.pforward2(f, h, dim)

        for ndim in range(dim):
            for x in itertools.product(TEST_INTERVAL, repeat=dim):
                assert fdiff1(x, ndim=ndim) == xfdiff1(x, ndim=ndim)
                assert fdiff2(x, ndim=ndim) == xfdiff2(x, ndim=ndim)

    @pytest.mark.parametrize(
        ("f", "dim", "expected"), [
            (lambda x: 0, 1, lambda h: [lambda x: 0]),
            (lambda x: 0, 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: 0, 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),

            (lambda x: 1, 1, lambda h: [lambda x: 0]),
            (lambda x: 1, 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: 1, 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),

            (lambda x: x[0], 1, lambda h: [lambda x: h]),
            (lambda x: x[0], 2, lambda h: [lambda x: h, lambda x: 0]),
            (lambda x: x[1], 2, lambda h: [lambda x: 0, lambda x: h]),
            (lambda x: x[0] + x[1], 2, lambda h: [lambda x: h, lambda x: h]),
            (lambda x: x[0] + x[1] + x[2], 3, lambda h: [lambda x: h, lambda x: h, lambda x: h]),

            (lambda x: x[0] ** 2, 1, lambda h: [lambda x: 2 * x[0] * h - h ** 2]),
            (lambda x: x[0] ** 2, 2, lambda h: [lambda x: 2 * x[0] * h - h ** 2, lambda x: 0]),
            (lambda x: x[1] ** 2, 2, lambda h: [lambda x: 0, lambda x: 2 * x[1] * h - h ** 2]),
            (lambda x: x[0] ** 2 + x[1] ** 2, 2, lambda h: [
                lambda x: 2 * x[0] * h - h ** 2,
                lambda x: 2 * x[1] * h - h ** 2
            ]),
            (lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2, 3, lambda h: [
                lambda x: 2 * x[0] * h - h ** 2,
                lambda x: 2 * x[1] * h - h ** 2,
                lambda x: 2 * x[2] * h - h ** 2
            ]),

            (lambda x: x[0] * x[1], 2, lambda h: [lambda x: h * x[1], lambda x: h * x[0]]),
            (lambda x: (x[0] * x[1]) ** 2, 2, lambda h: [
                lambda x: 2 * x[0] * x[1] ** 2 * h - (h * x[1]) ** 2,
                lambda x: 2 * x[0] ** 2 * x[1] * h - (h * x[0]) ** 2
            ]),
            (lambda x: x[0] * x[1] * x[2], 3, lambda h: [
                lambda x: h * x[1] * x[2],
                lambda x: h * x[0] * x[2],
                lambda x: h * x[0] * x[1]
            ]),
            (lambda x: (x[0] * x[1] * x[2]) ** 2, 3, lambda h: [
                lambda x: 2 * (x[0] * x[1] * x[2]) * (x[1] * x[2] * h) - (x[1] * x[2] * h) ** 2,
                lambda x: 2 * (x[0] * x[1] * x[2]) * (x[0] * x[2] * h) - (x[0] * x[2] * h) ** 2,
                lambda x: 2 * (x[0] * x[1] * x[2]) * (x[0] * x[1] * h) - (x[0] * x[1] * h) ** 2
            ])
        ]
    )
    def test_pbackward(
        self, f: typing.Callable[[typing.Sequence[float]], float], h: float, dim: int,
        expected: typing.Callable[
            [float], typing.Sequence[typing.Callable[[typing.Sequence[float]], float]]
        ]
    ):
        """
        Unit test for :py:`differential.PFiniteDifference.pbackward`.
        """
        fdiff, xfdiff = differential.PFiniteDifference.pbackward(f, h, dim), expected(h)

        for ndim in range(dim):
            for x in itertools.product(TEST_INTERVAL, repeat=dim):
                assert fdiff(x, ndim=ndim) == xfdiff[ndim](x)

    @pytest.mark.parametrize(
        ("f", "dim", "expected"), [
            (lambda x: 0, 1, lambda h: [lambda x: 0]),
            (lambda x: 0, 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: 0, 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),

            (lambda x: 1, 1, lambda h: [lambda x: 0]),
            (lambda x: 1, 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: 1, 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),

            (lambda x: x[0], 1, lambda h: [lambda x: 0]),
            (lambda x: x[0], 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: x[1], 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: x[0] + x[1], 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: x[0] + x[1] + x[2], 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),

            (lambda x: x[0] ** 2, 1, lambda h: [lambda x: 2 * h ** 2]),
            (lambda x: x[0] ** 2, 2, lambda h: [lambda x: 2 * h ** 2, lambda x: 0]),
            (lambda x: x[1] ** 2, 2, lambda h: [lambda x: 0, lambda x: 2 * h ** 2]),
            (lambda x: x[0] ** 2 + x[1] ** 2, 2, lambda h: [
                lambda x: 2 * h ** 2, lambda x: 2 * h ** 2
            ]),
            (lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2, 3, lambda h: [
                lambda x: 2 * h ** 2, lambda x: 2 * h ** 2, lambda x: 2 * h ** 2
            ]),

            (lambda x: x[0] * x[1], 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: (x[0] * x[1]) ** 2, 2, lambda h: [
                lambda x: 2 * (x[1] * h) ** 2, lambda x: 2 * (x[0] * h) ** 2
            ]),
            (lambda x: x[0] * x[1] * x[2], 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),
            (lambda x: (x[0] * x[1] * x[2]) ** 2, 3, lambda h: [
                lambda x: 2 * (x[1] * x[2] * h) ** 2,
                lambda x: 2 * (x[0] * x[2] * h) ** 2,
                lambda x: 2 * (x[0] * x[1] * h) ** 2
            ])
        ]
    )
    def test_pbackward2(
        self, f: typing.Callable[[typing.Sequence[float]], float], h: float, dim: int,
        expected: typing.Callable[
            [float], typing.Sequence[typing.Callable[[typing.Sequence[float]], float]]
        ]
    ):
        """
        Unit test for :py:`differential.PFiniteDifference.pbackward2`.
        """
        fdiff, xfdiff = differential.PFiniteDifference.pbackward2(f, h, dim), expected(h)

        for ndim in range(dim):
            for x in itertools.product(TEST_INTERVAL, repeat=dim):
                assert fdiff(x, ndim=ndim) == xfdiff[ndim](x)

    @pytest.mark.parametrize(
        ("f", "dim"), [
            (lambda x: 0, 1), (lambda x: 0, 2), (lambda x: 0, 3),
            (lambda x: 1, 1), (lambda x: 1, 2), (lambda x: 1, 3),

            (lambda x: x[0], 1), (lambda x: x[0], 2), (lambda x: x[0], 3),
            (lambda x: x[0] + x[1], 2), (lambda x: x[0] + x[1] + x[2], 3)
        ]
    )
    def test_pbackwardn(
        self, f: typing.Callable[[typing.Sequence[float]], float], h: float, dim: int
    ):
        """
        Unit test for :py:`differential.PFiniteDifference.pbackwardn`.
        """
        fdiff1 = differential.PFiniteDifference.pbackwardn(f, h, dim, 1)
        xfdiff1 = differential.PFiniteDifference.pbackward(f, h, dim)

        fdiff2 = differential.PFiniteDifference.pbackwardn(f, h, dim, 2)
        xfdiff2 = differential.PFiniteDifference.pbackward2(f, h, dim)

        for ndim in range(dim):
            for x in itertools.product(TEST_INTERVAL, repeat=dim):
                assert fdiff1(x, ndim=ndim) == xfdiff1(x, ndim=ndim)
                assert fdiff2(x, ndim=ndim) == xfdiff2(x, ndim=ndim)

    @pytest.mark.parametrize(
        ("f", "dim", "expected"), [
            (lambda x: 0, 1, lambda h: [lambda x: 0]),
            (lambda x: 0, 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: 0, 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),

            (lambda x: 1, 1, lambda h: [lambda x: 0]),
            (lambda x: 1, 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: 1, 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),

            (lambda x: x[0], 1, lambda h: [lambda x: h]),
            (lambda x: x[0], 2, lambda h: [lambda x: h, lambda x: 0]),
            (lambda x: x[1], 2, lambda h: [lambda x: 0, lambda x: h]),
            (lambda x: x[0] + x[1], 2, lambda h: [lambda x: h, lambda x: h]),
            (lambda x: x[0] + x[1] + x[2], 3, lambda h: [lambda x: h, lambda x: h, lambda x: h]),

            (lambda x: x[0] ** 2, 1, lambda h: [lambda x: 2 * x[0] * h]),
            (lambda x: x[0] ** 2, 2, lambda h: [lambda x: 2 * x[0] * h, lambda x: 0]),
            (lambda x: x[1] ** 2, 2, lambda h: [lambda x: 0, lambda x: 2 * x[1] * h]),
            (lambda x: x[0] ** 2 + x[1] ** 2, 2, lambda h: [
                lambda x: 2 * x[0] * h,
                lambda x: 2 * x[1] * h
            ]),
            (lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2, 3, lambda h: [
                lambda x: 2 * x[0] * h,
                lambda x: 2 * x[1] * h,
                lambda x: 2 * x[2] * h
            ]),

            (lambda x: x[0] * x[1], 2, lambda h: [lambda x: x[1] * h, lambda x: x[0] * h]),
            (lambda x: (x[0] * x[1]) ** 2, 2, lambda h: [
                lambda x: 2 * x[0] * x[1] ** 2 * h,
                lambda x: 2 * x[0] ** 2 * x[1] * h
            ]),
            (lambda x: x[0] * x[1] * x[2], 3, lambda h: [
                lambda x: x[1] * x[2] * h,
                lambda x: x[0] * x[2] * h,
                lambda x: x[0] * x[1] * h
            ]),
            (lambda x: (x[0] * x[1] * x[2]) ** 2, 3, lambda h: [
                lambda x: 2 * (x[0] * x[1] * x[2]) * (x[1] * x[2] * h),
                lambda x: 2 * (x[0] * x[1] * x[2]) * (x[0] * x[2] * h),
                lambda x: 2 * (x[0] * x[1] * x[2]) * (x[0] * x[1] * h)
            ])
        ]
    )
    def test_pcentral(
        self, f: typing.Callable[[typing.Sequence[float]], float], h: float, dim: int,
        expected: typing.Callable[
            [float], typing.Sequence[typing.Callable[[typing.Sequence[float]], float]]
        ]
    ):
        """
        Unit test for :py:`differential.PFiniteDifference.pcentral`.
        """
        fdiff, xfdiff = differential.PFiniteDifference.pcentral(f, h, dim), expected(h)

        for ndim in range(dim):
            for x in itertools.product(TEST_INTERVAL, repeat=dim):
                assert fdiff(x, ndim=ndim) == xfdiff[ndim](x)

    @pytest.mark.parametrize(
        ("f", "dim", "expected"), [
            (lambda x: 0, 1, lambda h: [lambda x: 0]),
            (lambda x: 0, 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: 0, 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),

            (lambda x: 1, 1, lambda h: [lambda x: 0]),
            (lambda x: 1, 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: 1, 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),

            (lambda x: x[0], 1, lambda h: [lambda x: 0]),
            (lambda x: x[0], 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: x[1], 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: x[0] + x[1], 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: x[0] + x[1] + x[2], 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),

            (lambda x: x[0] ** 2, 1, lambda h: [lambda x: 2 * h ** 2]),
            (lambda x: x[0] ** 2, 2, lambda h: [lambda x: 2 * h ** 2, lambda x: 0]),
            (lambda x: x[1] ** 2, 2, lambda h: [lambda x: 0, lambda x: 2 * h ** 2]),
            (lambda x: x[0] ** 2 + x[1] ** 2, 2, lambda h: [
                lambda x: 2 * h ** 2, lambda x: 2 * h ** 2
            ]),
            (lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2, 3, lambda h: [
                lambda x: 2 * h ** 2, lambda x: 2 * h ** 2, lambda x: 2 * h ** 2
            ]),

            (lambda x: x[0] * x[1], 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: (x[0] * x[1]) ** 2, 2, lambda h: [
                lambda x: 2 * (x[1] * h) ** 2, lambda x: 2 * (x[0] * h) ** 2
            ]),
            (lambda x: x[0] * x[1] * x[2], 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),
            (lambda x: (x[0] * x[1] * x[2]) ** 2, 3, lambda h: [
                lambda x: 2 * (x[1] * x[2] * h) ** 2,
                lambda x: 2 * (x[0] * x[2] * h) ** 2,
                lambda x: 2 * (x[0] * x[1] * h) ** 2
            ])
        ]
    )
    def test_pcentral2(
        self, f: typing.Callable[[typing.Sequence[float]], float], h: float, dim: int,
        expected: typing.Callable[
            [float], typing.Sequence[typing.Callable[[typing.Sequence[float]], float]]
        ]
    ):
        """
        Unit test for :py:`differential.PFiniteDifference.pcentral2`.
        """
        fdiff, xfdiff = differential.PFiniteDifference.pcentral2(f, h, dim), expected(h)

        for ndim in range(dim):
            for x in itertools.product(TEST_INTERVAL, repeat=dim):
                assert fdiff(x, ndim=ndim) == xfdiff[ndim](x)

    @pytest.mark.parametrize(
        ("f", "dim"), [
            (lambda x: 0, 1), (lambda x: 0, 2), (lambda x: 0, 3),
            (lambda x: 1, 1), (lambda x: 1, 2), (lambda x: 1, 3),

            (lambda x: x[0], 1), (lambda x: x[0], 2), (lambda x: x[0], 3),
            (lambda x: x[0] + x[1], 2), (lambda x: x[0] + x[1] + x[2], 3)
        ]
    )
    def test_pcentraln(
        self, f: typing.Callable[[typing.Sequence[float]], float], h: float, dim: int
    ):
        """
        Unit test for :py:`differential.FiniteDifference.pcentraln`.
        """
        fdiff1 = differential.PFiniteDifference.pcentraln(f, h, dim, 1)
        xfdiff1 = differential.PFiniteDifference.pcentral(f, h, dim)

        fdiff2 = differential.PFiniteDifference.pcentraln(f, h, dim, 2)
        xfdiff2 = differential.PFiniteDifference.pcentral2(f, h, dim)

        for ndim in range(dim):
            for x in itertools.product(TEST_INTERVAL, repeat=dim):
                assert fdiff1(x, ndim=ndim) == xfdiff1(x, ndim=ndim)
                assert fdiff2(x, ndim=ndim) == xfdiff2(x, ndim=ndim)


@pytest.mark.parametrize(
    "h", [2 ** n for n in range(0, -10, -1)]
)
class TestDifferenceQuotient:
    """
    Unit tests for :py:class:`differential.DifferenceQuotient`.
    """
    @pytest.mark.parametrize(
        ("f", "expected"), [
            (lambda x: 0, lambda h: (lambda x: 0)),
            (lambda x: 1, lambda h: (lambda x: 0)),
            (lambda x: -1, lambda h: (lambda x: 0)),
            (lambda x: x, lambda h: (lambda x: 1)),
            (lambda x: -x, lambda h: (lambda x: -1)),

            (lambda x: x ** 2, lambda h: (lambda x: 2 * x)),
            (lambda x: -x ** 2, lambda h: (lambda x: -2 * x)),
            (lambda x: x ** 3, lambda h: (lambda x: 3 * x ** 2 + h ** 2 / 4)),
            (lambda x: -x ** 3, lambda h: (lambda x: -3 * x ** 2 - h ** 2 / 4)),
            (lambda x: (-x) ** 3, lambda h: (lambda x: -3 * x ** 2 - h ** 2 / 4))
        ]
    )
    def test_quotient(
        self, f: typing.Callable[[float], float], h: float,
        expected: typing.Callable[[float], typing.Callable[[float], float]]
    ):
        """
        Unit test for :py:meth:`differential.DifferenceQuotient.quotient`.
        """
        dquot, xdquot = differential.DifferenceQuotient.quotient(f, h), expected(h)

        for x in TEST_INTERVAL:
            assert dquot(x) == xdquot(x)

    @pytest.mark.parametrize(
        "f", [
            lambda x: 0, lambda x: 1, lambda x: -1, lambda x: x, lambda x: -x,
            lambda x: x ** 2, lambda x: -x ** 2, lambda x: x ** 3, lambda x: -x ** 3, lambda x: (-x) ** 3
        ]
    )
    def test_quotient2(self, f: typing.Callable[[float], float], h: float):
        """
        Unit test for :py:meth:`differential.DifferenceQuotient.quotient2`.
        """
        dquot = differential.DifferenceQuotient.quotient2(f, h)
        xdquot = differential.DifferenceQuotient.quotient(
            differential.DifferenceQuotient.quotient(f, h), h
        )

        for x in TEST_INTERVAL:
            assert dquot(x) == xdquot(x)

    @pytest.mark.parametrize(
        "f", [
            lambda x: 0, lambda x: 1, lambda x: -1, lambda x: x, lambda x: -x,
            lambda x: x ** 2, lambda x: -x ** 2, lambda x: x ** 3, lambda x: -x ** 3, lambda x: (-x) ** 3
        ]
    )
    def test_quotientn(self, f: typing.Callable[[float], float], h: float):
        """
        Unit test for :py:meth:`differential.DifferenceQuotient.quotientn`.
        """
        dquot1 = differential.DifferenceQuotient.quotientn(f, h, 1)
        xdquot1 = differential.DifferenceQuotient.quotient(f, h)

        dquot2 = differential.DifferenceQuotient.quotientn(f, h, 2)
        xdquot2 = differential.DifferenceQuotient.quotient2(f, h)

        for x in TEST_INTERVAL:
            assert dquot1(x) == xdquot1(x)
            assert dquot2(x) == xdquot2(x)


@pytest.mark.parametrize(
    "h", [2 ** n for n in range(0, -10, -1)]
)
class TestPDifferenceQuotient:
    """
    Unit tests for :py:class:`differential.PDifferenceQuotient`.
    """
    @pytest.mark.parametrize(
        ("f", "dim", "expected"), [
            (lambda x: 0, 1, lambda h: [lambda x: 0]),
            (lambda x: 0, 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: 0, 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),

            (lambda x: 1, 1, lambda h: [lambda x: 0]),
            (lambda x: 1, 2, lambda h: [lambda x: 0, lambda x: 0]),
            (lambda x: 1, 3, lambda h: [lambda x: 0, lambda x: 0, lambda x: 0]),

            (lambda x: x[0], 1, lambda h: [lambda x: 1]),
            (lambda x: x[0], 2, lambda h: [lambda x: 1, lambda x: 0]),
            (lambda x: x[1], 2, lambda h: [lambda x: 0, lambda x: 1]),
            (lambda x: x[0] + x[1], 2, lambda h: [lambda x: 1, lambda x: 1]),
            (lambda x: x[0] + x[1] + x[2], 3, lambda h: [lambda x: 1, lambda x: 1, lambda x: 1]),

            (lambda x: x[0] ** 2, 1, lambda h: [lambda x: 2 * x[0]]),
            (lambda x: x[0] ** 2, 2, lambda h: [lambda x: 2 * x[0], lambda x: 0]),
            (lambda x: x[1] ** 2, 2, lambda h: [lambda x: 0, lambda x: 2 * x[1]]),
            (lambda x: x[0] ** 2 + x[1] ** 2, 2, lambda h: [
                lambda x: 2 * x[0],
                lambda x: 2 * x[1]
            ]),
            (lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2, 3, lambda h: [
                lambda x: 2 * x[0],
                lambda x: 2 * x[1],
                lambda x: 2 * x[2]
            ]),

            (lambda x: x[0] * x[1], 2, lambda h: [lambda x: x[1], lambda x: x[0]]),
            (lambda x: (x[0] * x[1]) ** 2, 2, lambda h: [
                lambda x: 2 * x[0] * x[1] ** 2,
                lambda x: 2 * x[0] ** 2 * x[1]
            ]),
            (lambda x: x[0] * x[1] * x[2], 3, lambda h: [
                lambda x: x[1] * x[2],
                lambda x: x[0] * x[2],
                lambda x: x[0] * x[1]
            ]),
            (lambda x: (x[0] * x[1] * x[2]) ** 2, 3, lambda h: [
                lambda x: 2 * (x[0] * x[1] * x[2]) * (x[1] * x[2]),
                lambda x: 2 * (x[0] * x[1] * x[2]) * (x[0] * x[2]),
                lambda x: 2 * (x[0] * x[1] * x[2]) * (x[0] * x[1])
            ])
        ]
    )
    def test_pquotient(
        self, f: typing.Callable[[float], float], h: float, dim: int,
        expected: typing.Callable[[float], typing.Callable[[float], float]]
    ):
        """
        Unit test for :py:meth:`differential.PDifferenceQuotient.pquotient`.
        """
        dquot, xdquot = differential.PDifferenceQuotient.pquotient(f, h, dim), expected(h)

        for ndim in range(dim):
            for x in itertools.product(TEST_INTERVAL, repeat=dim):
                assert dquot[ndim](x) == xdquot[ndim](x)

    @pytest.mark.parametrize(
        ("f", "dim"), [
            (lambda x: 0, 1), (lambda x: 0, 2), (lambda x: 0, 3),
            (lambda x: 1, 1), (lambda x: 1, 2), (lambda x: 1, 3),

            (lambda x: x[0], 1), (lambda x: x[0], 2), (lambda x: x[0], 3),
            (lambda x: x[0] + x[1], 2), (lambda x: x[0] + x[1] + x[2], 3)
        ]
    )
    def test_quotient2(self, f: typing.Callable[[float], float], h: float, dim: int):
        """
        Unit test for :py:meth:`differential.PDifferenceQuotient.pquotient2`.
        """
        dquot = differential.PDifferenceQuotient.pquotient2(f, h, dim)
        xdquot = [
            differential.PDifferenceQuotient.pquotient(partial, h, dim, ndim=ndim)
            for ndim, partial in enumerate(differential.PDifferenceQuotient.pquotient(f, h, dim))
        ]

        for ndim in range(dim):
            for x in itertools.product(TEST_INTERVAL, repeat=dim):
                assert dquot[ndim](x) == xdquot[ndim](x)

    @pytest.mark.parametrize(
        ("f", "dim"), [
            (lambda x: 0, 1), (lambda x: 0, 2), (lambda x: 0, 3),
            (lambda x: 1, 1), (lambda x: 1, 2), (lambda x: 1, 3),

            (lambda x: x[0], 1), (lambda x: x[0], 2), (lambda x: x[0], 3),
            (lambda x: x[0] + x[1], 2), (lambda x: x[0] + x[1] + x[2], 3)
        ]
    )
    def test_pquotientn(self, f: typing.Callable[[float], float], h: float, dim: int):
        """
        Unit test for :py:meth:`differential.PDifferenceQuotient.pquotientn`.
        """
        dquot1 = differential.PDifferenceQuotient.pquotientn(f, h, 1, dim)
        xdquot1 = differential.PDifferenceQuotient.pquotient(f, h, dim)

        dquot2 = differential.PDifferenceQuotient.pquotientn(f, h, 2, dim)
        xdquot2 = differential.PDifferenceQuotient.pquotient2(f, h, dim)

        for ndim in range(dim):
            for x in itertools.product(TEST_INTERVAL, repeat=dim):
                assert dquot1[ndim](x) == xdquot1[ndim](x)
                assert dquot2[ndim](x) == xdquot2[ndim](x)
