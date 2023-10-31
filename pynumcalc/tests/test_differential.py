"""
Unit tests for :py:mod:`pynumcalc.differential`.
"""

import typing

import pytest

from pynumcalc import differential


@pytest.mark.parametrize(
    "h", [2 ** n for n in range(0, -10, -1)]
)
class TestFiniteDifference:
    """
    Unit tests for :py:class:`finitediff.FiniteDifference`.
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
            (lambda x: -x ** 3, lambda h: (lambda x: -3 * x ** 2 * h - 3 * x * h ** 2 - h ** 3)),
            (lambda x: (-x) ** 3, lambda h: (lambda x: -3 * x ** 2 * h - 3 * x * h ** 2 - h ** 3))
        ]
    )
    def test_forward(
        self, f: typing.Callable[[float], float], h: float,
        expected: typing.Callable[[float], typing.Callable[[float], float]]
    ):
        """
        Unit tests for :py:`finitediff.FiniteDifference.forward`.
        """
        fdiff, xfdiff = differential.FiniteDifference.forward(f, h), expected(h)

        for x in range(-10, 11):
            assert fdiff(x) == xfdiff(x), (h, x)

    @pytest.mark.parametrize(
        ("f", "expected"), []
    )
    def test_foward2(
        self, f: typing.Callable[[float], float], h: float,
        expected: typing.Callable[[float], typing.Callable[[float], float]]
    ):
        """
        Unit tests for :py:`finitediff.FiniteDifference.forward`.
        """

    @pytest.mark.parametrize(
        ("f", "expected"), []
    )
    def test_forwardn(
        self, f: typing.Callable[[float], float], h: float,
        expected: typing.Callable[[float], typing.Callable[[float], float]]
    ):
        """
        Unit tests for :py:`finitediff.FiniteDifference.forwardn`.
        """

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
        Unit tests for :py:`finitediff.FiniteDifference.backward`.
        """
        fdiff, xfdiff = differential.FiniteDifference.backward(f, h), expected(h)

        for x in range(-10, 11):
            assert fdiff(x) == xfdiff(x), (h, x)

    @pytest.mark.parametrize(
        ("f", "expected"), []
    )
    def test_backward2(
        self, f: typing.Callable[[float], float], h: float,
        expected: typing.Callable[[float], typing.Callable[[float], float]]
    ):
        """
        Unit tests for :py:`finitediff.FiniteDifference.backward2`.
        """

    @pytest.mark.parametrize(
        ("f", "expected"), []
    )
    def test_backwardn(
        self, f: typing.Callable[[float], float], h: float,
        expected: typing.Callable[[float], typing.Callable[[float], float]]
    ):
        """
        Unit tests for :py:`finitediff.FiniteDifference.backwardn`.
        """

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
        Unit tests for :py:`finitediff.FiniteDifference.central`.
        """
        fdiff, xfdiff = differential.FiniteDifference.central(f, h), expected(h)

        for x in range(-10, 11):
            assert fdiff(x) == xfdiff(x), (h, x)

    @pytest.mark.parametrize(
        ("f", "expected"), []
    )
    def test_central2(
        self, f: typing.Callable[[float], float], h: float,
        expected: typing.Callable[[float], typing.Callable[[float], float]]
    ):
        """
        Unit tests for :py:`finitediff.FiniteDifference.central2`.
        """

    @pytest.mark.parametrize(
        ("f", "expected"), []
    )
    def test_centraln(
        self, f: typing.Callable[[float], float], h: float,
        expected: typing.Callable[[float], typing.Callable[[float], float]]
    ):
        """
        Unit tests for :py:`finitediff.FiniteDifference.centraln`.
        """
