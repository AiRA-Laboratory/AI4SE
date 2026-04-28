# Optional Track — JAX for Scientific AI

**TA:** Lê Nguyễn Ngọc Vũ · **Prerequisites:** Module 1 (PyTorch basics), Module 3 (PINN)

---

## Why JAX?

JAX is increasingly popular in scientific ML research because of:

- **`jit`:** JIT-compile any Python/NumPy function → near-C performance
- **`grad`:** Automatic differentiation for arbitrary Python functions, not just `nn.Module`
- **`vmap`:** Vectorize a function over a batch axis without rewriting it
- **`pmap`:** Parallelize across multiple devices (TPU/GPU) trivially

For scientific computing (PINN, operator learning, physics simulations), JAX often enables cleaner code than PyTorch because physics residuals are just Python functions — no `nn.Module` wrapper needed.

---

## Track Contents

| Notebook | Topic |
|----------|-------|
| [`01_jax_basics.ipynb`](notebooks/01_jax_basics.ipynb) | JAX arrays, `jit`, `grad`, `vmap` |
| [`02_flax_mlp.ipynb`](notebooks/02_flax_mlp.ipynb) | MLP training with Flax + Optax |
| [`03_jax_pinn.ipynb`](notebooks/03_jax_pinn.ipynb) | PINN for ODE reimplemented in JAX |

---

## Environment Setup

Install JAX alongside the base environment:

```bash
# CPU-only (simplest for tutorial)
pip install jax jaxlib

# GPU (CUDA 12)
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Neural network libraries
pip install flax optax
```

Verify:
```python
import jax
print(jax.__version__)
print(jax.devices())
```

---

## JAX vs PyTorch — Quick Reference

| Feature | PyTorch | JAX |
|---------|---------|-----|
| Arrays | `torch.Tensor` | `jnp.ndarray` (immutable) |
| Gradient | `loss.backward()` | `jax.grad(loss_fn)(params)` |
| JIT compile | `torch.compile` | `jax.jit(fn)` |
| Vectorize | Manual batch dim | `jax.vmap(fn)` |
| Neural nets | `nn.Module` | Flax `nn.Module` |
| Optimizer | `torch.optim` | `optax` |
| State | Mutable in-place | Immutable — explicit state |

**Key mindset shift:** JAX functions are **pure** (no side effects, no in-place ops). State (model parameters, optimizer state) is carried explicitly as pytree dicts, not hidden inside objects.

---

## Resources

- [JAX official quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
- [Ben Moseley JAX workshop](https://github.com/benmoseley/jax-tutorial)
- [Flax documentation](https://flax.readthedocs.io/)
- [Optax documentation](https://optax.readthedocs.io/)
- [JAX by example (PINN examples)](https://jax-fenics.readthedocs.io/)
