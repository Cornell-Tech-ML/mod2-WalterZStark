from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the variable.

        Args:
        ----
            x (Any): The derivative to be accumulated for this variable.

        Return:
        ------
            None

        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for the variable.

        Args:
        ----
            None

        Returns:
        -------
            int: A unique identifier for this variable.

        """
        ...

    def is_leaf(self) -> bool:
        """Checks if the variable is a leaf in the computation graph.

        Args:
        ----
            None

        Returns:
        -------
            bool: True if the variable is a leaf; otherwise, False.

        """
        ...

    def is_constant(self) -> bool:
        """Checks if the variable is constant.

        Args:
        ----
            None

        Returns:
        -------
            bool: True if the variable is constant; otherwise, False.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of this variable.

        Args:
        ----
            None

        Returns:
        -------
            Iterable[Variable]: An iterable containing the parent variables.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients for parent variables.

        Args:
        ----
            d_output (Any): The gradient of the output with respect to the loss.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple
            contains a parent variable and its corresponding gradient.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
    variable: The right-most variable.
    deriv: The derivative we want to propagate backward to the leaves.

    Returns:
    -------
    None: Updates the derivative values of each leaf through accumulate_derivative`.

    """
    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns saved 'values' to be used for backpropagation."""
        return self.saved_values
