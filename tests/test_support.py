import jax.numpy as jnp

from pref_bootstrap.utils.utils import numeric_grad


def grad_check(fn, grad_fn, point, *, atol=1e-8, rtol=1e-5, eps=1e-5):
    """Assert that gradient of `fn` at `point` matches `grad_fn`.

    Args:
        fn (jnp.ndarray -> float): function to differentiate.
        grad_fn (ndarray -> ndarray): putative gradient of fn (will be checked
            numerically).
        atol: absolute tolerance for allclose comparison.
        rtol: relative tolerance for allclose comparison.
        eps: epsilon to use for two-sided numeric differentiation."""
    grad_fn_val = grad_fn(point)
    num_grad_val = numeric_grad(fn, point, eps=eps)
    assert jnp.allclose(grad_fn_val, num_grad_val, atol=atol, rtol=rtol)
