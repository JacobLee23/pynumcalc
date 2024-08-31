"""
Unit tests for :py:mod:`pynumcalc.differential`.
"""

import numpy as np
import pytest

from pynumcalc.differential import FiniteDifference, Forward, Backward, Central, DifferenceQuotient
from . import functions
from .functions import RealFunctionCase


PMIN, PMAX = 1, 10
HVALUES = np.power(2, -np.arange(PMIN, PMAX + 1, dtype=FiniteDifference.dtype))


@pytest.mark.parametrize("h", HVALUES)
@pytest.mark.parametrize("function", functions.FUNCTIONS)
class TestForward:
    """
    Unit tests for :py:class:`Forward`.
    """
    def test_first(self, function: RealFunctionCase, h: float):
        """
        Unit test for :py:meth:`Forward.first`.
        """
        findiff = Forward(function.f, h)

        for x in function.domain:
            assert findiff.first(x) == pytest.approx(function.forward1(h)(x)), x

    def test_second(self, function: RealFunctionCase, h: float):
        """
        Unit test for :py:meth:`Forward.second`.
        """
        findiff = Forward(function.f, h)

        for x in function.domain:
            assert findiff.second(x) == pytest.approx(function.forward2(h)(x)), x

    def test_nth(self, function: RealFunctionCase, h: float):
        """
        Unit test for :py:meth:`Forward.nth`.
        """
        findiff = Forward(function.f, h)

        for x in function.domain:
            assert findiff.nth(x, 1) == pytest.approx(findiff.first(x)), x
            assert findiff.nth(x, 2) == pytest.approx(findiff.second(x)), x


@pytest.mark.parametrize("h", HVALUES)
@pytest.mark.parametrize("function", functions.FUNCTIONS)
class TestBackward:
    """
    Unit tests for :py:class:`Backward`.
    """
    def test_first(self, function: RealFunctionCase, h: float):
        """
        Unit test for :py:meth:`Backward.first`.
        """
        findiff = Backward(function.f, h)

        for x in function.domain:
            assert findiff.first(x) == pytest.approx(function.backward1(h)(x)), x

    def test_second(self, function: RealFunctionCase, h: float):
        """
        Unit test for :py:meth:`Backward.second`.
        """
        findiff = Backward(function.f, h)

        for x in function.domain:
            assert findiff.second(x) == pytest.approx(function.backward2(h)(x)), x

    def test_nth(self, function: RealFunctionCase, h: float):
        """
        Unit test for :py:meth:`Backward.nth`.
        """
        findiff = Backward(function.f, h)

        for x in function.domain:
            assert findiff.nth(x, 1) == pytest.approx(findiff.first(x)), x
            assert findiff.nth(x, 2) == pytest.approx(findiff.second(x)), x


@pytest.mark.parametrize("h", HVALUES)
@pytest.mark.parametrize("function", functions.FUNCTIONS)
class TestCentral:
    """
    Unit tests for :py:class:`Central`.
    """
    def test_first(self, function: RealFunctionCase, h: float):
        """
        Unit test for :py:meth:`Central.first`.
        """
        findiff = Central(function.f, h)

        for x in function.domain:
            assert findiff.first(x) == pytest.approx(function.central1(h)(x)), x

    def test_second(self, function: RealFunctionCase, h: float):
        """
        Unit test for :py:meth:`Central.second`.
        """
        findiff = Central(function.f, h)

        for x in function.domain:
            assert findiff.second(x) == pytest.approx(function.central2(h)(x)), x

    def test_nth(self, function: RealFunctionCase, h: float):
        """
        Unit test for :py:meth:`Central.nth`.
        """
        findiff = Central(function.f, h)

        for x in function.domain:
            assert findiff.nth(x, 1) == pytest.approx(findiff.first(x)), x
            assert findiff.nth(x, 2) == pytest.approx(findiff.second(x)), x


@pytest.mark.parametrize("h", HVALUES)
@pytest.mark.parametrize("function", functions.FUNCTIONS)
class TestDifferenceQuotient:
    """
    Unit tests for :py:class:`DifferenceQuotient`.
    """
    def test_first(self, function: RealFunctionCase, h: float):
        """
        Unit test for :py:meth:`DifferenceQuotient.first`.
        """
        diffquot = DifferenceQuotient(function.f, h)

        for x in function.domain:
            assert diffquot.first(x) == pytest.approx(function.dquotient1(h)(x)), x

    def test_second(self, function: RealFunctionCase, h: float):
        """
        Unit test for :py:meth:`DifferenceQuotient.second`.
        """
        diffquot = DifferenceQuotient(function.f, h)

        for x in function.domain:
            assert diffquot.second(x) == pytest.approx(function.dquotient2(h)(x)), x

    def test_nth(self, function: RealFunctionCase, h: float):
        """
        Unit test for :py:meth:`DifferenceQuotient.nth`.
        """
        diffquot = DifferenceQuotient(function.f, h)

        for x in function.domain:
            assert diffquot.nth(x, 1) == pytest.approx(function.dquotient1(h)(x)), x
            assert diffquot.nth(x, 2) == pytest.approx(function.dquotient2(h)(x)), x
