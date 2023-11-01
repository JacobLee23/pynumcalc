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
    lower, upper = -10, 10

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
        Unit test for :py:`finitediff.FiniteDifference.forward`.
        """
        fdiff, xfdiff = differential.FiniteDifference.forward(f, h), expected(h)

        for x in range(self.lower, self.upper + 1):
            assert fdiff(x) == xfdiff(x), (h, x)

    @pytest.mark.parametrize(
        "f", [
            lambda x: 0, lambda x: 1, lambda x: -1, lambda x: x, lambda x: -x,
            lambda x: x ** 2, lambda x: -x ** 3, lambda x: x ** 3, lambda x: -x ** 3, lambda x: (-x) ** 3
        ]
    )
    def test_foward2(self, f: typing.Callable[[float], float], h: float):
        """
        Unit test for :py:`finitediff.FiniteDifference.forward`.
        """
        fdiff = differential.FiniteDifference.forward2(f, h)
        xfdiff = differential.FiniteDifference.forward(
            differential.FiniteDifference.forward(f, h), h
        )

        for x in range(self.lower, self.upper + 1):
            assert fdiff(x) == xfdiff(x), (h, x)

    @pytest.mark.parametrize(
        "f", [
            lambda x: 0, lambda x: 1, lambda x: -1, lambda x: x, lambda x: -x,
            lambda x: x ** 2, lambda x: -x ** 3, lambda x: x ** 3, lambda x: -x ** 3, lambda x: (-x) ** 3
        ]
    )
    def test_forwardn(self, f: typing.Callable[[float], float], h: float):
        """
        Unit test for :py:`finitediff.FiniteDifference.forwardn`.
        """
        fdiff1 = differential.FiniteDifference.forwardn(f, h, 1)
        xfdiff1 = differential.FiniteDifference.forward(f, h)

        fdiff2 = differential.FiniteDifference.forwardn(f, h, 2)
        xfdiff2 = differential.FiniteDifference.forward2(f, h)

        for x in range(self.lower, self.upper + 1):
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
        Unit test for :py:`finitediff.FiniteDifference.backward`.
        """
        fdiff, xfdiff = differential.FiniteDifference.backward(f, h), expected(h)

        for x in range(self.lower, self.upper + 1):
            assert fdiff(x) == xfdiff(x), (h, x)

    @pytest.mark.parametrize(
        "f", [
            lambda x: 0, lambda x: 1, lambda x: -1, lambda x: x, lambda x: -x,
            lambda x: x ** 2, lambda x: -x ** 3, lambda x: x ** 3, lambda x: -x ** 3, lambda x: (-x) ** 3
        ]
    )
    def test_backward2(self, f: typing.Callable[[float], float], h: float):
        """
        Unit test for :py:`finitediff.FiniteDifference.backward2`.
        """
        fdiff = differential.FiniteDifference.backward2(f, h)
        xfdiff = differential.FiniteDifference.backward(
            differential.FiniteDifference.backward(f, h), h
        )

        for x in range(self.lower, self.upper + 1):
            assert fdiff(x) == xfdiff(x), (h, x)

    @pytest.mark.parametrize(
        "f", [
            lambda x: 0, lambda x: 1, lambda x: -1, lambda x: x, lambda x: -x,
            lambda x: x ** 2, lambda x: -x ** 3, lambda x: x ** 3, lambda x: -x ** 3, lambda x: (-x) ** 3
        ]
    )
    def test_backwardn(self, f: typing.Callable[[float], float], h: float):
        """
        Unit test for :py:`finitediff.FiniteDifference.backwardn`.
        """
        fdiff1 = differential.FiniteDifference.backwardn(f, h, 1)
        xfdiff1 = differential.FiniteDifference.backward(f, h)

        fdiff2 = differential.FiniteDifference.backwardn(f, h, 2)
        xfdiff2 = differential.FiniteDifference.backward2(f, h)

        for x in range(self.lower, self.upper + 1):
            assert fdiff1(x) == xfdiff1(x), (h, x, 1)
            assert fdiff2(x) == xfdiff2(x), (h, x, 2)

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
        Unit test for :py:`finitediff.FiniteDifference.central`.
        """
        fdiff, xfdiff = differential.FiniteDifference.central(f, h), expected(h)

        for x in range(self.lower, self.upper + 1):
            assert fdiff(x) == xfdiff(x), (h, x)

    @pytest.mark.parametrize(
        "f", [
            lambda x: 0, lambda x: 1, lambda x: -1, lambda x: x, lambda x: -x,
            lambda x: x ** 2, lambda x: -x ** 3, lambda x: x ** 3, lambda x: -x ** 3, lambda x: (-x) ** 3
        ]
    )
    def test_central2(self, f: typing.Callable[[float], float], h: float):
        """
        Unit test for :py:`finitediff.FiniteDifference.central2`.
        """
        fdiff = differential.FiniteDifference.central2(f, h)
        xfdiff = differential.FiniteDifference.central(
            differential.FiniteDifference.central(f, h), h
        )

        for x in range(self.lower, self.upper + 1):
            assert fdiff(x) == xfdiff(x), (h, x)

    @pytest.mark.parametrize(
        "f", [
            lambda x: 0, lambda x: 1, lambda x: -1, lambda x: x, lambda x: -x,
            lambda x: x ** 2, lambda x: -x ** 3, lambda x: x ** 3, lambda x: -x ** 3, lambda x: (-x) ** 3
        ]
    )
    def test_centraln(self, f: typing.Callable[[float], float], h: float):
        """
        Unit test for :py:`finitediff.FiniteDifference.centraln`.
        """
        fdiff1 = differential.FiniteDifference.centraln(f, h, 1)
        xfdiff1 = differential.FiniteDifference.central(f, h)

        fdiff2 = differential.FiniteDifference.centraln(f, h, 2)
        xfdiff2 = differential.FiniteDifference.central2(f, h)

        for x in range(self.lower, self.upper + 1):
            assert fdiff1(x) == xfdiff1(x), (h, x, 1)
            assert fdiff2(x) == xfdiff2(x), (h, x, 2)


@pytest.mark.parametrize(
    "h", [2 ** n for n in range(0, -10, -1)]
)
class TestDifferenceQuotient:
    """
    Unit tests for :py:class:`differential.DifferenceQuotient`.
    """
    lower, upper = -10, 10

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

        for x in range(self.lower, self.upper + 1):
            assert dquot(x) == xdquot(x), (h, x)

    @pytest.mark.parametrize(
        "f", [
            lambda x: 0, lambda x: 1, lambda x: -1, lambda x: x, lambda x: -x,
            lambda x: x ** 2, lambda x: -x ** 2, lambda x: x ** 3, lambda x: -x ** 3, lambda x: (-x) ** 3
        ]
    )
    def test_quotient2(self, f: typing.Callable[[float], float], h: float):
        """
        Unit test for :py:meth:`differential.DifferenceQuotient.quotient`.
        """
        dquot = differential.DifferenceQuotient.quotient2(f, h)
        xdquot = differential.DifferenceQuotient.quotient(
            differential.DifferenceQuotient.quotient(f, h), h
        )

        for x in range(self.lower, self.upper + 1):
            assert dquot(x) == xdquot(x), (h, x)

    @pytest.mark.parametrize(
        "f", [
            lambda x: 0, lambda x: 1, lambda x: -1, lambda x: x, lambda x: -x,
            lambda x: x ** 2, lambda x: -x ** 2, lambda x: x ** 3, lambda x: -x ** 3, lambda x: (-x) ** 3
        ]
    )
    def test_quotientn(self, f: typing.Callable[[float], float], h: float):
        """
        Unit test for :py:meth:`differential.DifferenceQuotient.quotient`.
        """
        dquot1 = differential.DifferenceQuotient.quotientn(f, h, 1)
        xdquot1 = differential.DifferenceQuotient.quotient(f, h)

        dquot2 = differential.DifferenceQuotient.quotientn(f, h, 2)
        xdquot2 = differential.DifferenceQuotient.quotient2(f, h)

        for x in range(self.lower, self.upper + 1):
            assert dquot1(x) == xdquot1(x), (h, x, 1)
            assert dquot2(x) == xdquot2(x), (h, x, 2)
