from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies a class's forward method to a Scalar(s) and outputs the resulting Scalar."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Adds the float 'a' to 'b'."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward for Add method."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the log of float 'a' and saves 'a' as context."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward for Log method."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# ---- Task 1.2.


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass for the Mul method."""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass for the Mul method."""
        (a, b) = ctx.saved_values
        gradident_a = b * d_output
        gradident_b = a * d_output
        return gradident_a, gradident_b


class Inv(ScalarFunction):
    """Calculates the reciprocal of 'x'"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for the Inv method."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass for the Inv method."""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for the Neg method."""
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass for the Neg method."""
        return -d_output


class Sigmoid(ScalarFunction):
    """Maps input 'x' to a value between 0 and 1 using the sigmoid activation function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for the Sigmoid method."""
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass for the Sigmoid method."""
        (a,) = ctx.saved_values
        gradient_a = operators.mul(operators.sigmoid(a), (1 - operators.sigmoid(a)))
        return gradient_a * d_output


class ReLU(ScalarFunction):
    """Maps input 'x' to a value between 0 and 1 using the rectified linear unit activation function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for the ReLU method."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass for the ReLU method."""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Calculates the exponential function of 'x'"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for the Exp method."""
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass for the Exp method."""
        (a,) = ctx.saved_values
        return d_output * operators.exp(a)


class LT(ScalarFunction):
    """Checks if 'x' is less than 'y'"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass for the LT method."""
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass for the LT method."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Checks if 'x' is equal to 'y'"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass for the EQ method."""
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass for the EQ method."""
        return 0.0, 0.0
