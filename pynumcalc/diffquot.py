"""
"""

from .finitediff import (
    FiniteDifference, Forward, Backward, Central,
    PFiniteDifference, PForward, PBackward, PCentral
)
from .typedef import (
    RealFunction, RealFunctionN
)


class DifferenceQuotient:
    """
    """
    def __init__(self, f: RealFunction, h: float):
        self._f  = f
        self._h = h

        self._fdiff: FiniteDifference
        try:
            self._fdiff = Central(self.f, self.h)
        except ValueError:
            try:
                self._fdiff = Forward(self.f, self.h)
            except ValueError:
                self._fdiff = Backward(self.f, self.h)

    @property
    def f(self) -> RealFunction:
        """
        """
        return self._f
    
    @property
    def h(self) -> float:
        """
        """
        return self._h
    
    @h.setter
    def h(self, value: float) -> None:
        self._h = float(value)
        self.fdiff.h = self.h

    @property
    def fdiff(self) -> FiniteDifference:
        """
        """
        return self._fdiff

    def first(self, x: float) -> float:
        """
        """
        return self.fdiff.first(x) / self.h
    
    def second(self, x: float) -> float:
        """
        """
        return self.fdiff.second(x) / pow(self.h, 2)
    
    def nth(self, x: float, n: int) -> float:
        """
        """
        return self.fdiff.nth(x, n) / pow(self.h, n)


class PDifferenceQuotient:
    """
    """
    def __init__(self, f: RealFunctionN, dim: int, h: float):
        self._f  = f
        self._dim = dim
        self._h = h

        self._fdiff: PFiniteDifference
        try:
            self._fdiff = PCentral(self.f, self.h)
        except ValueError:
            try:
                self._fdiff = PForward(self.f, self.h)
            except ValueError:
                self._fdiff = PBackward(self.f, self.h)

    @property
    def f(self) -> RealFunction:
        """
        """
        return self._f
    
    @property
    def dim(self) -> int:
        """
        """
        return self._dim
    
    @property
    def h(self) -> float:
        """
        """
        return self._h
    
    @h.setter
    def h(self, value: float) -> None:
        self._h = float(value)
        self.fdiff.h = self.h

    @property
    def fdiff(self) -> PFiniteDifference:
        """
        """
        return self._fdiff

    def first(self, x: float) -> float:
        """
        """
        return self.fdiff.first(x) / self.h
    
    def second(self, x: float) -> float:
        """
        """
        return self.fdiff.second(x) / pow(self.h, 2)
    
    def nth(self, x: float, n: int) -> float:
        """
        """
        return self.fdiff.nth(x, n) / pow(self.h, n)
