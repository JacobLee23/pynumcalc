"""
Contains type aliases for mathematical function definitions.

.. py:data:: RealFunction

.. py:data:: RealFunctionN
"""

import typing

import numpy as np


RealFunction = typing.Callable[[float], float]
RealFunctionN = typing.Callable[[np.ndarray], float]
