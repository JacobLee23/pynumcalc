"""
Unit tests for :py:mod:`pynumcalc.differential`.
"""

import numpy as np
import pytest

from . import functions
from .functions import RealFunctionCase
from pynumcalc.differential import FiniteDifference, Forward, Backward, Central, DifferenceQuotient


PMIN, PMAX = 1, 10
HVALUES = np.power(2, -np.arange(PMIN, PMAX + 1, dtype=FiniteDifference._dtype))


@pytest.mark.parametrize("h", HVALUES)
@pytest.mark.parametrize("function", functions.FUNCTIONS)
class TestForward:
    """
    Unit tests for :py:class:`Forward`.
    """
    def test_first(self, function: RealFunctionCase, h: float):
        """
        Unit tests for :py:meth:`Forward.first`.
        """
        finite_difference = Forward(function.f, h)

        for x in function.domain:
            assert finite_difference.first(x) == pytest.approx(function.forward1(h)(x)), x

    def test_second(self, function: RealFunctionCase, h: float):
        """
        Unit tests for :py:meth:`Forward.second`.
        """
        finite_difference = Forward(function.f, h)

        for x in function.domain:
            assert finite_difference.second(x) == pytest.approx(function.forward2(h)(x)), x

    def test_nth(self, function: RealFunctionCase, h: float):
        """
        Unit tests for :py:meth:`Forward.nth`.
        """
        finite_difference = Forward(function.f, h)

        for x in function.domain:
            assert finite_difference.nth(x, 1) == pytest.approx(finite_difference.first(x)), x
            assert finite_difference.nth(x, 2) == pytest.approx(finite_difference.second(x)), x


@pytest.mark.parametrize("h", HVALUES)
@pytest.mark.parametrize("function", functions.FUNCTIONS)
class TestBackward:
    """
    Unit tests for :py:class:`Backward`.
    """
    def test_first(self, function: RealFunctionCase, h: float):
        """
        Unit tests for :py:meth:`Backward.first`.
        """
        finite_difference = Backward(function.f, h)

        for x in function.domain:
            assert finite_difference.first(x) == pytest.approx(function.backward1(h)(x)), x

    def test_second(self, function: RealFunctionCase, h: float):
        """
        Unit tests for :py:meth:`Backward.second`.
        """
        finite_difference = Backward(function.f, h)

        for x in function.domain:
            assert finite_difference.second(x) == pytest.approx(function.backward2(h)(x)), x

    def test_nth(self, function: RealFunctionCase, h: float):
        """
        Unit tests for :py:meth:`Backward.nth`.
        """
        finite_difference = Backward(function.f, h)

        for x in function.domain:
            assert finite_difference.nth(x, 1) == pytest.approx(finite_difference.first(x)), x
            assert finite_difference.nth(x, 2) == pytest.approx(finite_difference.second(x)), x


@pytest.mark.parametrize("h", HVALUES)
@pytest.mark.parametrize("function", functions.FUNCTIONS)
class TestCentral:
    """
    Unit tests for :py:class:`Central`.
    """
    def test_first(self, function: RealFunctionCase, h: float):
        """
        Unit tests for :py:meth:`Central.first`.
        """
        finite_difference = Central(function.f, h)

        for x in function.domain:
            assert finite_difference.first(x) == pytest.approx(function.central1(h)(x)), x

    def test_second(self, function: RealFunctionCase, h: float):
        """
        Unit tests for :py:meth:`Central.second`.
        """
        finite_difference = Central(function.f, h)

        for x in function.domain:
            assert finite_difference.second(x) == pytest.approx(function.central2(h)(x)), x

    def test_nth(self, function: RealFunctionCase, h: float):
        """
        Unit tests for :py:meth:`Central.nth`.
        """
        finite_difference = Central(function.f, h)

        for x in function.domain:
            assert finite_difference.nth(x, 1) == pytest.approx(finite_difference.first(x)), x
            assert finite_difference.nth(x, 2) == pytest.approx(finite_difference.second(x)), x


@pytest.mark.parametrize("h", HVALUES)
@pytest.mark.parametrize("function", functions.FUNCTIONS)
class TestDifferenceQuotient:
    """
    Unit tests for :py:class:`DifferenceQuotient`.
    """
    def test_first(self, function: RealFunctionCase, h: float):
        """
        Unit tests for :py:meth:`DifferenceQuotient.first`.
        """
        difference_quotient = DifferenceQuotient(function.f, h)

        for x in function.domain:
            assert difference_quotient.first(x) == pytest.approx(function.dquotient1(h)(x)), x

    def test_second(self, function: RealFunctionCase, h: float):
        """
        Unit tests for :py:meth:`DifferenceQuotient.second`.
        """
        difference_quotient = DifferenceQuotient(function.f, h)

        for x in function.domain:
            assert difference_quotient.second(x) == pytest.approx(function.dquotient2(h)(x)), x

    def test_nth(self, function: RealFunctionCase, h: float):
        """
        Unit tests for :py:meth:`DifferenceQuotient.nth`.
        """
        difference_quotient = DifferenceQuotient(function.f, h)

        for x in function.domain:
            assert difference_quotient.nth(x, 1) == pytest.approx(function.dquotient1(h)(x)), x
            assert difference_quotient.nth(x, 2) == pytest.approx(function.dquotient2(h)(x)), x
