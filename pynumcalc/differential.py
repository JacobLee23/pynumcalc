"""
Interfaces for differential calculus numerical computation.
"""

from numbers import Number
import typing

import numpy as np
import scipy.special

from .types import RealFunction, RealFunctionN


class FiniteDifference:
    r"""
    Computes the finite differences for a real-valued function ``f`` of ``dim`` dimensions defined
    by :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}` using step size ``h``. If ``dim`` is ``None``,
    then ``f`` is a real-valued function defined as :math:`f: \mathbb{R} \to \mathbb{R}`.

    :param f: A callable representation of a real-valued function
    :param h: The step size of the finite difference
    :param dim: The dimension of the domain of ``f``
    """
    dtype: type = np.float64

    _h = None

    @typing.overload
    def __init__(self, f: RealFunction, h: float, dim: int = None): ...

    @typing.overload
    def __init__(self, f: RealFunctionN, h: float, dim: int): ...

    def __init__(
        self, f: typing.Union[RealFunction, RealFunctionN], h: float,
        dim: typing.Optional[int] = None
    ):
        self._f = f
        self._dim = dim

        self.h = h

    @staticmethod
    def domain_membership(x: typing.Union[float, np.ndarray], dim: typing.Optional[int]) -> bool:
        r"""
        Tests the membership of an element in the domain of some real-valued function :math:`f`
        defined as :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}` based on the dimension of the test
        element and the dimension of the domain of :math:`f`.

        :param x: A test domain element
        :param dim: The dimension of the domain of the test function
        :return: Whether ``x`` is in the domain of the test function
        :raise TypeError: The type of ``x`` does not match the value of ``dim``
        :raise ValueError: The dimension of ``x`` does not equal ``dim``
        """
        if dim is None and isinstance(x, float):
            return True

        if dim is not None and isinstance(x, np.ndarray):

            if (dim,) != x.shape:
                raise ValueError(x)

            return True

        raise TypeError(x)

    @typing.overload
    @classmethod
    def _first(cls, f: RealFunction, h: float, x: float, dim: int = None) -> float: ...

    @typing.overload
    @classmethod
    def _first(cls, f: RealFunctionN, h: float, x: np.ndarray, dim: int) -> np.ndarray: ...

    @classmethod
    def _first(
        cls, f: typing.Union[RealFunction, RealFunctionN], h: float,
        x: typing.Union[float, np.ndarray], dim: typing.Optional[int] = None
    ) -> typing.Union[float, np.ndarray]:
        """
        Computes the first-order finite difference of a real-valued function ``f`` of ``dim``
        dimensions at domain element ``x`` using step size ``h``.

        :param f: A callable representation of a real-valued function
        :param h: The step size of the finite difference
        :param x: The domain element of ``f`` at which to calculate the finite difference
        :param dim: The dimension of the domain of ``f``
        :return: The first-order finite difference(s) of ``f`` at ``x`` using step size ``h``
        """
        raise NotImplementedError

    @typing.overload
    @classmethod
    def _second(cls, f: RealFunction, h: float, x: float, dim: int = None) -> float: ...

    @typing.overload
    @classmethod
    def _second(cls, f: RealFunctionN, h: float, x: np.ndarray, dim: int) -> np.ndarray: ...

    @classmethod
    def _second(
        cls, f: typing.Union[RealFunction, RealFunctionN], h: float,
        x: typing.Union[float, np.ndarray], dim: typing.Optional[int] = None
    ) -> typing.Union[float, np.ndarray]:
        """
        Computes the second-order finite difference of a real-valued funciton ``f`` of ``dim``
        dimensions at domain element ``x`` using step size ``h``.

        :param f: A callable representation of a real-valued function
        :param h: The step size of the finite difference
        :param x: The domain element of ``f`` at which to calculate the finite difference
        :param dim: The dimension of the domain of ``f``
        :return: The first-order finite difference of ``f`` at ``x`` using step size ``h``
        """
        raise NotImplementedError

    @typing.overload
    @classmethod
    def _nth(cls, f: RealFunction, h: float, x: float, n: int, dim: int = None) -> float: ...

    @typing.overload
    @classmethod
    def _nth(cls, f: RealFunctionN, h: float, x: np.ndarray, n: int, dim: int) -> np.ndarray: ...

    @typing.overload
    @classmethod
    def _nth(
        cls, f: typing.Union[RealFunction, RealFunctionN], h: float,
        x: typing.Union[float, np.ndarray], n: int, dim: typing.Optional[int] = None
    ) -> typing.Union[float, np.ndarray]:
        r"""
        Computes the ``n``\th-order finite difference of a real-valued function ``f`` of ``dim``
        dimensions at domain element ``x`` using step size ``h``.

        :param f: A callable representation of a real-valued function
        :param h: The step size of the finite difference
        :param x: The domain element of ``f`` at which to calculate the finite difference
        :param n: The order of the finite difference
        :param dim: The dimension of the domain of ``f``
        :return: The ``n``\th-order finite difference of ``f`` at ``x`` using step size ``h``
        """
        raise NotImplementedError

    @property
    def f(self) -> typing.Union[RealFunction, RealFunctionN]:
        r"""
        A real-valued function defined by :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}` if
        :py:attr:`dim` is not ``None``, otherwise :math:`f: \mathbb{R} \to \mathbb{R}`.
        """
        return self._f

    @property
    def h(self) -> float:
        """
        The h size used to compute finite differences.
        """
        return self._h

    @h.setter
    def h(self, value: Number) -> None:
        self._h = float(value)

    @property
    def dim(self) -> typing.Optional[int]:
        """
        The dimension of the domain of :py:attr:`f`. If ``None``, then :py:attr:`f` is a
        one-dimensional function.
        """
        return self._dim

    @dim.setter
    def dim(self, value: typing.Optional[int]) -> None:
        self._dim = value

    @typing.overload
    def first(self, x: Number) -> float: ...

    @typing.overload
    def first(self, x: np.ndarray) -> np.ndarray: ...

    def first(
        self, x: typing.Union[Number, np.ndarray]
    ) -> typing.Union[float, np.ndarray]:
        """
        Computes the first-order finite difference of real-valued function :py:attr:`f` at ``x``
        using step size :py:attr`h`.

        If :py:attr:`dim` is ``None``, then ``x`` is expected to be type ``float``; otherwise,
        ``x`` is expected to be type :class:`np.ndarray`.

        :param x: The domain element at which to calculate the finite difference
        :return: The first-order finite difference at ``x``
        """
        return self._first(
            self.f, self.h, x if isinstance(x, np.ndarray) else float(x), self.dim
        )

    @typing.overload
    def second(self, x: Number) -> float: ...

    @typing.overload
    def second(self, x: np.ndarray) -> np.ndarray: ...

    def second(
        self, x: typing.Union[Number, np.ndarray]
    ) -> typing.Union[float, np.ndarray]:
        """
        Computes the second-order finite difference of real-valued function :py:attr:`f` at ``x``
        using step size :py:attr:`h`.

        If :py:attr:`dim` is ``None``, then ``x`` is expected to be type ``float``; otherwise,
        ``x`` is expected to be type :class:`np.ndarray`.

        :param x: The domain element at which to calculate the finite difference
        :return: The second-order finite difference at ``x``
        """
        return self._second(
            self.f, self.h, x if isinstance(x, np.ndarray) else float(x), self.dim
        )

    @typing.overload
    def nth(self, x: Number, n: int) -> float: ...

    @typing.overload
    def nth(self, x: np.ndarray, n: int) -> np.ndarray: ...

    def nth(
        self, x: typing.Union[Number, np.ndarray], n: int
    ) -> typing.Union[float, np.ndarray]:
        r"""
        Computes the ``n``\th-order finite difference of real-valued function :py:attr:`f` at ``x``
        using step size :py:attr:`h`.

        If :py:attr:`dim` is ``None``, then ``x`` is expected to be type ``float``; otherwise,
        ``x`` is expected to be type :class:`np.ndarray`.

        :param x: The domain element at which to calculate the finite difference
        :param n: The order of the finite difference
        :return: The ``n``\th-order finite difference at ``x``
        """
        return self._nth(
            self.f, self.h, x if isinstance(x, np.ndarray) else float(x), n, self.dim
        )


class Forward(FiniteDifference):
    r"""
    Computes :math:`n`th-order forward finite differences for a real-valued function ``f`` of
    ``dim`` dimensions, defined by :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}` using step size
    ``h``.

    If ``dim`` is ``None``, then ``f`` is a real-valued function defined by
    :math:`f: \mathbb{R} \to \mathbb{R}`, and the :math:`n`th-order forward finite difference of
    :math:`f` is denoted by

    .. math::

        {\Delta}_{h}^{n} [f](x)

    Otherwise, the :math:`n`th-order forward finite difference of :math:`f` is denoted by

    .. math::

        {\Delta}_{h} [f](\vec{x}) = \begin{bmatrix}
            {\Delta}_{h} {[f]}_{x_1}(\vec{x}) \\
            \vdots \\
            {\Delta}_{h} {[f]}_{x_{\dim{\vec{x}}}}(\vec{x}) \\
        \end{bmatrix}
    """
    @classmethod
    def _first(
        cls, f: typing.Union[RealFunction, RealFunctionN], h: float,
        x: typing.Union[float, np.ndarray], dim: typing.Optional[int] = None
    ) -> typing.Union[float, np.ndarray]:
        r"""
        If ``dim`` is ``None``, which implies that ``f`` is defined by
        :math:`f: \mathbb{R} \to \mathbb{R}`, then the first-order forward finite difference of
        ``f`` using step size ``h`` at ``x`` is defined as

        .. math::

            {\Delta}_{h} [f](x) = f(x + h) - f(x)

        If ``dim`` is not ``None``, which implies that ``f`` is defined by
        :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}`, then the first-order forward finite difference
        of ``f`` using step size ``h`` at ``x`` is defined as

        .. math::

            {\Delta}_{h} {[f]}_{x_i}(\vec{x}) = f(
                \langle x_1, \dots, x_i + h, \dots, {x}_{\dim{\vec{x}}} \rangle
            ) - f(\vec{x})

        :param f: A callable representation of a real-valued function
        :param h: The step size of the finite difference
        :param x: The domain element of ``f`` at which to calculate the finite difference
        :param dim: The dimension of the domain of ``f``
        :return: The first-order finite difference of ``f`` at ``x`` using step size ``h``
        :raise TypeError: The type of ``x`` does not match the value of ``dim``
        """
        assert cls.domain_membership(x, dim)

        if dim is None:
            return f(x + h) - f(x)

        index = np.arange(dim)

        return np.fromiter(
            (
                f(np.where(index == d, x[d] + h, x)) - f(x) for d in range(dim)
            ), cls.dtype
        )

    @classmethod
    def _second(
        cls, f: typing.Union[RealFunction, RealFunctionN], h: float,
        x: typing.Union[float, np.ndarray], dim: typing.Optional[int] = None
    ) -> typing.Union[float, np.ndarray]:
        r"""
        If ``dim`` is ``None``, which implies that ``f`` is defined by
        :math:`f: \mathbb{R} \to \mathbb{R}`, then the second-order forward finite difference of
        ``f`` using step size ``h`` at ``x`` is defined as:

        .. math::

            {\Delta}_{h}^{2} [f](x) = f(x + 2h) - 2f(x + h) + f(x)

        If ``dim`` is not ``None``, which implies that ``f`` is defined by
        :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}`, then the second-order forward finite difference
        of ``f`` using step size ``h`` at ``x`` is defined as:

        .. math::

            {\Delta}_{h}^{2} {[f]}_{x_i}(\vec{x}) = f(
                \langle x_1, \dots, x_i + 2h, \dots, {x}_{\dim{\vec{x}}} \rangle
            ) - 2f(
                \langle x_1, \dots, x_i + h, \dots, {x}_{\dim{\vec{x}}} \rangle
            ) + f(\vec{x})

        :param f: A callable representation of a real-valued function
        :param h: The step size of the finite difference
        :param x: The domain element of ``f`` at which to calculate the finite difference
        :param dim: The dimension of the domain of ``f``
        :return: The first-order finite difference of ``f`` at ``x`` using step size ``h``
        :raise TypeError: The type of ``x`` does not match the value of ``dim``
        """
        assert cls.domain_membership(x, dim)

        if dim is None:
            return f(x + 2 * h) - 2 * f(x + h) + f(x)

        index = np.arange(dim)

        return np.fromiter(
            (
                f(
                    np.where(index == d, x[d] + 2 * h, x)
                ) - 2 * f(
                    np.where(index == d, x[d] + h, x)
                ) + f(x) for d in range(dim)
            ), cls.dtype
        )

    @classmethod
    def _nth(
        cls, f: typing.Union[RealFunction, RealFunctionN], h: float,
        x: typing.Union[float, np.ndarray], n: int, dim: typing.Optional[int] = None
    ) -> typing.Union[float, np.ndarray]:
        r"""
        If ``dim`` is ``None``, which implies that ``f`` is defined by
        :math:`f: \mathbb{R} \to \mathbb{R}`, then the ``n``\th-order forward finite difference of
        ``f`` at ``x`` using step size ``h`` is defined as:

        .. math::

            {\Delta}_{h}^{n} [f](x) = \sum_{k=0}^{n} {(-1)}^{n-k} {{n}\choose{k}} f(x + kh)

        If ``dim`` is not ``None``, which implies that ``f`` is defined by
        :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}`, then the ``n``\th-order forward finite
        difference of ``f`` at ``x`` using step size ``h`` is defined as:

        .. math::

            {\Delta}_{h}^{n} {[f]}_{x_i}(\vec{x}) = \sum_{k=0}^{n} {(-1)}^{n-k} {{n}\choose{k}} f(
                \langle x_1, \dots, x_i + kh, \dots, {x}_{\dim{\vec{x}}} \rangle
            )

        :param f: A callable representation of a real-valued function
        :param h: The step size of the finite difference
        :param x: The domain element of ``f`` at which to calculate the finite difference
        :param n: The order of the finite difference
        :param dim: The dimension of the domain of ``f``
        :return: The first-order finite difference of ``f`` at ``x`` using step size ``h``
        :raise TypeError: The type of ``x`` does not match the value of ``dim``
        """
        assert cls.domain_membership(x, dim)

        k = np.arange(0, n + 1, 1, cls.dtype)

        if dim is None:
            return (
                np.power(-1, n - k) * scipy.special.comb(n, k) * np.fromiter(
                    map(f, x + k * h), cls.dtype
                )
            ).sum()

        index = np.arange(dim)

        return np.fromiter(
            (
                (
                    np.power(-1, n - k) * scipy.special.comb(n, k) * f(
                        np.where(index == d, x[d] + k * h, x)
                    )
                ).sum() for d in range(dim)
            ), cls.dtype
        )


class Backward(FiniteDifference):
    r"""
    Computes :math:`n`th-order backward finite differences for a real-valued function ``f`` of
    ``dim`` dimensions, defined by :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}` using step size
    ``h``.

    If ``dim`` is ``None``, then ``f`` is a real-valued function defined by
    :math:`f: \mathbb{R} \to \mathbb{R}`, and the :math:`n`th-order backward finite difference of
    :math:`f` is denoted by

    .. math::

        {\nabla}_{h}^{n} [f](x)

    Otherwise, the :math:`n`th-order backward finite difference of :math:`f` is denoted by

    .. math::

        {\nabla}_{h} [f](\vec{x}) = \begin{bmatrix}
            {\nabla}_{h} {[f]}_{x_1}(\vec{x}) \\
            \vdots \\
            {\nabla}_{h} {[f]}_{x_{\dim{\vec{x}}}}(\vec{x}) \\
        \end{bmatrix}
    """
    @classmethod
    def _first(
        cls, f: typing.Union[RealFunction, RealFunctionN], h: float,
        x: typing.Union[float, np.ndarray], dim: typing.Optional[int] = None
    ) -> typing.Union[float, np.ndarray]:
        r"""
        If ``dim`` is ``None``, which implies that ``f`` is defined by
        :math:`f: \mathbb{R} \to \mathbb{R}`, then the first-order backward finite difference of
        ``f`` using step size ``h`` at ``x`` is defined as

        .. math::

            {\nabla}_{h} [f](x) = f(x) - f(x - h)

        If ``dim`` is not ``None``, which implies that ``f`` is defined by
        :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}`, then the first-order backward finite difference
        of ``f`` using step size ``h`` at ``x`` is defined as

        .. math::

            {\nabla}_{h} {[f]}_{x_i}(\vec{x}) = f(\vec{x}) - f(
                \langle x_1, \dots, x_i - h, \dots, {x}_{\dim{\vec{x}}} \rangle
            )

        :param f: A callable representation of a real-valued function
        :param h: The step size of the finite difference
        :param x: The domain element of ``f`` at which to calculate the finite difference
        :param dim: The dimension of the domain of ``f``
        :return: The first-order finite difference of ``f`` at ``x`` using step size ``h``
        :raise TypeError: The type of ``x`` does not match the value of ``dim``
        """
        assert cls.domain_membership(x, dim)

        if dim is None:
            return f(x) - f(x - h)

        index = np.arange(dim)

        return np.fromiter(
            (
                f(x) - f(
                    np.where(index == d, x[d] - h, x)
                ) for d in range(dim)
            ), cls.dtype
        )

    @classmethod
    def _second(
        cls, f: typing.Union[RealFunction, RealFunctionN], h: float,
        x: typing.Union[float, np.ndarray], dim: typing.Optional[int] = None
    ) -> typing.Union[float, np.ndarray]:
        r"""
        If ``dim`` is ``None``, which implies that ``f`` is defined by
        :math:`f: \mathbb{R} \to \mathbb{R}`, then the second-order forward finite difference of
        ``f`` using step size ``h`` at ``x`` is defined as:

        .. math::

            {\nabla}_{h}^{2} [f](x) = f(x) - 2f(x - h) + f(x - 2h)

        If ``dim`` is not ``None``, which implies that ``f`` is defined by
        :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}`, then the second-order forward finite difference
        of ``f`` using step size ``h`` at ``x`` is defined as:

        .. math::

            {\nabla}_{h}^{2} {[f]}_{x_i}(\vec{x}) = f(\vec{x}) - 2f(
                \langle x_1, \dots, x_i - h, \dots, {x}_{\dim{\vec{x}}} \rangle
            ) + f(
                \langle x_1, \dots, x_i - 2h, \dots, {x}_{\dim{\vec{x}}} \rangle
            )

        :parma f:
        :param h:
        :param x:
        :param dim:
        :return:
        :raise TypeError:
        """
        assert cls.domain_membership(x, dim)

        if dim is None:
            return f(x) - 2 * f(x - h) + f(x - 2 * h)

        index = np.arange(dim)

        return np.fromiter(
            (
                f(x) - 2 * f(
                    np.where(index == d, x[d] - h, x)
                ) + f(
                    np.where(index == d, x[d] - 2 * h, x)
                ) for d in range(dim)
            ), cls.dtype
        )

    @classmethod
    def _nth(
        cls, f: typing.Union[RealFunction, RealFunctionN], h: float,
        x: typing.Union[float, np.ndarray], n: int, dim: typing.Optional[int] = None
    ) -> typing.Union[float, np.ndarray]:
        r"""
        If ``dim`` is ``None``, which implies that ``f`` is defined by
        :math:`f: \mathbb{R} \to \mathbb{R}`, then the ``n``\th-order backward finite difference of
        ``f`` at ``x`` using step size ``h`` is defined as:

        .. math::

            {\nabla}_{h}^{n} [f](x) = \sum_{k=0}^{n} {(-1)}^{k} {{n}\choose{k}} f(x - kh)

        If ``dim`` is not ``None``, which implies that ``f`` is defined by
        :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}`, then the ``n``\th-order backward finite
        difference of ``f`` at ``x`` using step size ``h`` is defined as:

        .. math::

            {\nabla}_{h}^{n} {[f]}_{x_i}(\vec{x}) = \sum_{k=0}^{n} {(-1)}^{k} {{n}\choose{k}} f(
                \langle x_1, \dots, x_i - kh, \dots, {x}_{\dim{\vec{x}}} \rangle
            )

        :param f: A callable representation of a real-valued function
        :param h: The step size of the finite difference
        :param x: The domain element of ``f`` at which to calculate the finite difference
        :param n: The order of the finite difference
        :param dim: The dimension of the domain of ``f``
        :return: The first-order finite difference of ``f`` at ``x`` using step size ``h``
        :raise TypeError: The type of ``x`` does not match the value of ``dim``
        """
        assert cls.domain_membership(x, dim)

        k = np.arange(0, n + 1, 1, cls.dtype)

        if dim is None:
            return (
                np.power(-1, k) * scipy.special.comb(n, k) * np.fromiter(
                    map(f, x - k * h), cls.dtype
                )
            ).sum()

        index = np.arange(dim)

        return np.fromiter(
            (
                (
                    np.power(-1, k) * scipy.special.comb(n, k) * f(
                        np.where(index == d, x[d] - k * h, x)
                    )
                ).sum() for d in range(dim)
            ), cls.dtype
        )


class Central(FiniteDifference):
    r"""
    Computes :math:`n`th-order central finite differences for a real-valued function ``f`` of
    ``dim`` dimensions, defined by :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}` using step size
    ``h``.

    If ``dim`` is ``None``, then ``f`` is a real-valued function defined by
    :math:`f: \mathbb{R} \to \mathbb{R}`, and the :math:`n`th-order central finite difference of
    :math:`f` is denoted by

    .. math::

        {\delta}_{h}^{n} [f](x)

    Otherwise, the :math:`n`th-order backward finite difference of :math:`f` is denoted by

    .. math::

        {\delta}_{h} [f](\vec{x}) = \begin{bmatrix}
            {\delta}_{h} {[f]}_{x_1}(\vec{x}) \\
            \vdots \\
            {\delta}_{h} {[f]}_{x_{\dim{\vec{x}}}}(\vec{x}) \\
        \end{bmatrix}
    """
    @classmethod
    def _first(
        cls, f: typing.Union[RealFunction, RealFunctionN], h: float,
        x: typing.Union[float, np.ndarray], dim: typing.Optional[int] = None
    ) -> typing.Union[float, np.ndarray]:
        r"""
        If ``dim`` is ``None``, which implies that ``f`` is defined by
        :math:`f: \mathbb{R} \to \mathbb{R}`, then the first-order central finite difference of
        ``f`` using step size ``h`` at ``x`` is defined as

        .. math::

            {\delta}_{h} [f](x) = f(x + \frac{h}{2}) - f(x - \frac{h}{2})

        If ``dim`` is not ``None``, which implies that ``f`` is defined by
        :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}`, then the first-order central finite difference
        of ``f`` using step size ``h`` at ``x`` is defined as

        .. math::

            {\delta}_{h} {[f]}_{x_i}(\vec{x}) = f(
                \langle x_1, \dots, x_i + \frac{h}{2}, \dots, {x}_{\dim{\vec{x}}} \rangle
            ) - f(
                \langle x_1, \dots, x_i - \frac{h}{2}, \dots, {x}_{\dim{\vec{x}}} \rangle
            )

        :param f: A callable representation of a real-valued function
        :param h: The step size of the finite difference
        :param x: The domain element of ``f`` at which to calculate the finite difference
        :param dim: The dimension of the domain of ``f``
        :return: The first-order finite difference of ``f`` at ``x`` using step size ``h``
        :raise TypeError: The type of ``x`` does not match the value of ``dim``
        """
        assert cls.domain_membership(x, dim)

        if dim is None:
            return f(x + h / 2) - f(x - h / 2)

        index = np.arange(dim)

        return np.fromiter(
            (
                f(
                    np.where(index == d, x[d] + h / 2, x)
                ) - f(
                    np.where(index == d, x[d] - h / 2, x)
                ) for d in range(dim)
            ), cls.dtype
        )

    @classmethod
    def _second(
        cls, f: typing.Union[RealFunction, RealFunctionN], h: float,
        x: typing.Union[float, np.ndarray], dim: typing.Optional[int] = None
    ) -> typing.Union[float, np.ndarray]:
        r"""
        If ``dim`` is ``None``, which implies that ``f`` is defined by
        :math:`f: \mathbb{R} \to \mathbb{R}`, then the second-order central finite difference of
        ``f`` using step size ``h`` at ``x`` is defined as:

        .. math::

            {\delta}_{h}^{2} [f](x) = f(x + h) - 2f(x) + f(x - h)

        If ``dim`` is not ``None``, which implies that ``f`` is defined by
        :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}`, then the second-order central finite difference
        of ``f`` using step size ``h`` at ``x`` is defined as:

        .. math::

            {\delta}_{h}^{2} {[f]}_{x_i}(\vec{x}) = f(
                \langle x_1, \dots, x_i + h, \dots, {x}_{\dim{\vec{x}}} \rangle
            ) - 2f(\vec{x}) + f(
                \langle x_1, \dots, x_i - h, \dots, {x}_{\dim{\vec{x}}} \rangle
            )

        :param f: A callable representation of a real-valued function
        :param h: The step size of the finite difference
        :param x: The domain element of ``f`` at which to calculate the finite difference
        :param dim: The dimension of the domain of ``f``
        :return: The first-order finite difference of ``f`` at ``x`` using step size ``h``
        :raise TypeError: The type of ``x`` does not match the value of ``dim``
        """
        assert cls.domain_membership(x, dim)

        if dim is None:
            return f(x + h) - 2 * f(x) + f(x - h)

        index = np.arange(dim)

        return np.fromiter(
            (
                f(
                    np.where(index == d, x[d] + h, x)
                ) - 2 * f(x) + f(
                    np.where(index == d, x[d] - h, x)
                ) for d in range(dim)
            ), cls.dtype
        )

    @classmethod
    def _nth(
        cls, f: typing.Union[RealFunction, RealFunctionN], h: float,
        x: typing.Union[float, np.ndarray], n: int, dim: typing.Optional[int] = None
    ) -> typing.Union[float, np.ndarray]:
        r"""
        If ``dim`` is ``None``, which implies that ``f`` is defined by
        :math:`f: \mathbb{R} \to \mathbb{R}`, then the ``n``\th-order central finite difference of
        ``f`` at ``x`` using step size ``h`` is defined as:

        .. math::

            {\delta}_{h}^{n} [f](x) = \sum_{k=0}^{n} {(-1)}^{k} {{n}\choose{k}} f(x + (\frac{n}{2} - k)h)

        If ``dim`` is not ``None``, which implies that ``f`` is defined by
        :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}`, then the ``n``\th-order central finite
        difference of ``f`` at ``x`` using step size ``h`` is defined as:

        .. math::

            {\delta}_{h}^{n} {[f]}_{x_i}(\vec{x}) = \sum_{k=0}^{n} {(-1)}^{k} {{n}\choose{k}} f(
                \langle x_1, \dots, x_i + (\frac{n}{2} - k)h, \dots, {x}_{\dim{\vec{x}}} \rangle
            )

        :param f: A callable representation of a real-valued function
        :param h: The step size of the finite difference
        :param x: The domain element of ``f`` at which to calculate the finite difference
        :param n: The order of the finite difference
        :param dim: The dimension of the domain of ``f``
        :return: The first-order finite difference of ``f`` at ``x`` using step size ``h``
        :raise TypeError: The type of ``x`` does not match the value of ``dim``
        """
        assert cls.domain_membership(x, dim)

        k = np.arange(0, n + 1, 1, cls.dtype)

        if dim is None:
            return (
                np.power(-1, k) * scipy.special.comb(n, k) * np.fromiter(
                    map(f, x + (n / 2 - k) * h), k.dtype
                )
            ).sum()

        index = np.arange(dim)

        return np.fromiter(
            (
                (
                    np.power(-1, k) * scipy.special.comb(n, k) * f(
                        np.where(index == d, x[d] + (n / 2 - k) * h, x)
                    )
                ).sum() for d in range(dim)
            ), cls.dtype
        )


class DifferenceQuotient:
    r"""
    Computes the difference quotient for a real-valued function ``f`` of ``dim`` dimensions defined
    by :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}` using step size ``h``. If ``dim`` is ``None``,
    then ``f`` is a real-valued function defined as :math:`f: \mathbb{R} \to \mathbb{R}`.

    :param f: A callable representaiton of a real-valued function
    :param h: The step size of the difference quotient
    :param dim: The dimension of the domain of ``f``
    """
    @typing.overload
    def __init__(self, f: RealFunction, h: float, dim: int = None): ...

    @typing.overload
    def __init__(self, f: RealFunctionN, h: float, dim: int): ...

    def __init__(
        self, f: typing.Union[RealFunction, RealFunctionN], h: float,
        dim: typing.Optional[int] = None
    ):
        self._f = f
        self._h = h
        self._dim = dim

        self._forward = Forward(self.f, self.h, self.dim)
        self._backward = Backward(self.f, self.h, self.dim)
        self._central = Central(self.f, self.h, self.dim)

    @property
    def f(self) -> typing.Union[RealFunction, RealFunctionN]:
        r"""
        A real-valued function defined by :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}` if
        :py:attr:`dim` is not ``None``, otherwise :math:`f: \mathbb{R} \to \mathbb{R}`.
        """
        return self._f

    @property
    def h(self) -> float:
        """
        The h size used to compute the difference quotient.
        """
        return self._h

    @h.setter
    def h(self, value: float) -> None:
        self._h = value

    @property
    def dim(self) -> int:
        """
        The dimension of the domain of :py:attr:`f`. If ``None``, then :py:attr:`f` is a
        one-dimensional function.
        """
        return self._dim

    @property
    def forward(self) -> Forward:
        """
        """
        return self._forward

    @property
    def backward(self) -> Backward:
        """
        """
        return self._backward

    @property
    def central(self) -> Central:
        """
        """
        return self._central

    @typing.overload
    def first(self, x: float) -> float: ...

    @typing.overload
    def first(self, x: np.ndarray) -> np.ndarray: ...

    def first(self, x: typing.Union[float, np.ndarray]) -> typing.Union[float, np.ndarray]:
        """
        Computes the first-order difference quotient of real-valued function :py:attr:`f` at ``x``
        using step size :py:attr`h`.

        If :py:attr:`dim` is ``None``, then ``x`` is expected to be type ``float``; otherwise,
        ``x`` is expected to be type :class:`np.ndarray`.

        :param x: The domain element at which to calculate the difference quotient
        :return: The first-order difference quotient at ``x``
        """
        try:
            fdiff = self.central.first(x)
        except ValueError:
            try:
                fdiff = self.forward.first(x)
            except ValueError:
                fdiff = self.backward.first(x)

        return fdiff / self.h

    @typing.overload
    def second(self, x: float) -> float: ...

    @typing.overload
    def second(self, x: np.ndarray) -> np.ndarray: ...

    def second(self, x: typing.Union[float, np.ndarray]) -> typing.Union[float, np.ndarray]:
        """
        Computes the second-order difference quotient of real-valued function :py:attr:`f` at ``x``
        using step size :py:attr`h`.

        If :py:attr:`dim` is ``None``, then ``x`` is expected to be type ``float``; otherwise,
        ``x`` is expected to be type :class:`np.ndarray`.

        :param x: The domain element at which to calculate the difference quotient
        :return: The second-order difference quotient at ``x``
        """
        try:
            fdiff = self.central.second(x)
        except ValueError:
            try:
                fdiff = self.forward.second(x)
            except ValueError:
                fdiff = self.backward.second(x)

        return fdiff / np.power(self.h, 2)

    @typing.overload
    def nth(self, x: float, n: int) -> float: ...

    @typing.overload
    def nth(self, x: np.ndarray, n: int) -> np.ndarray: ...

    def nth(self, x: typing.Union[float, np.ndarray], n: int) -> typing.Union[float, np.ndarray]:
        r"""
        Computes the ``n``\th-order difference quotient of real-valued function :py:attr:`f` at
        ``x`` using step size :py:attr`h`.

        If :py:attr:`dim` is ``None``, then ``x`` is expected to be type ``float``; otherwise,
        ``x`` is expected to be type :class:`np.ndarray`.

        :param x: The domain element at which to calculate the difference quotient
        :param n: The order of the difference quotient
        :return: The ``n``\th-order difference quotient at ``x``
        """
        try:
            fdiff = self.central.nth(x, n)
        except ValueError:
            try:
                fdiff = self.forward.nth(x, n)
            except ValueError:
                fdiff = self.backward.nth(x, n)

        return fdiff / np.power(self.h, n)
