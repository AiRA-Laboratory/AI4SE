# JAX Optional Track FAQ

> Lê Nguyễn Ngọc Vũ: update after each session.

---

## Installation

**Q: `jax.devices()` shows only CPU even though I have a GPU.**

Check you installed the GPU wheel:
```bash
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Then verify: `print(jax.devices())` should show `[CudaDevice(id=0)]`.

---

**Q: `import flax` fails.**

```bash
pip install flax optax
```

---

## Core Transformations

**Q: My `jit`-compiled function gives different results from the plain version.**

Most likely you have a Python `if` on a traced (abstract) value. Replace:
```python
# BAD inside jit
if x > 0:
    return x
else:
    return -x
```
with:
```python
# GOOD
return jnp.where(x > 0, x, -x)
# or
return jax.lax.cond(x > 0, lambda: x, lambda: -x)
```

---

**Q: `jax.grad` gives an error: "grad requires a scalar output".**

`jax.grad` only differentiates scalar-output functions. If your function returns a vector, use `jax.jacobian` instead, or reduce to a scalar first (e.g., `jnp.mean`).

---

**Q: `print` inside `jit` only shows on first call. How do I debug?**

Use `jax.debug.print`:
```python
@jax.jit
def f(x):
    jax.debug.print("x = {x}", x=x)  # prints every call
    return x * 2
```

---

**Q: What does "ConcretizationTypeError" mean?**

You're using a JAX traced value in a Python operation that requires a concrete value (e.g., `len(x)`, Python `if` on `x`). Either:
1. Use JAX operations instead of Python equivalents
2. Mark the argument as `static` with `static_argnums`

---

## Flax and Optax

**Q: How do I save and load Flax model parameters?**

```python
import orbax.checkpoint as ocp

# Save
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/path/to/checkpoint', params)

# Load
params = checkpointer.restore('/path/to/checkpoint')
```

---

**Q: What is the difference between `nn.compact` and `setup` in Flax?**

- `@nn.compact`: define submodules inline inside `__call__` — cleaner for simple models
- `setup`: define submodules in a `setup()` method, then use in `__call__` — better for large models with shared submodules

---

**Q: How do I use dropout or batch norm in Flax (they need RNG/state)?**

```python
class MLPWithDropout(nn.Module):
    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Dense(64)(x)
        x = nn.Dropout(rate=0.3, deterministic=not training)(x)
        return nn.Dense(1)(x)

# At train time, pass an RNG key:
variables = model.init({'params': key, 'dropout': dropout_key}, x, training=True)
```

---

## JAX PINN

**Q: My `vmap(grad(...))` gives wrong shapes.**

When using `vmap` over `grad`, the function inside `grad` must take a *scalar* input (single collocation point), not a batch. Pattern:

```python
# Step 1: function for a single point
def u_single(params, t_scalar):
    return model.apply({'params': params}, t_scalar.reshape(1, 1))[0, 0]

# Step 2: derivative for a single point
du_dt_single = lambda params, t: jax.grad(u_single, argnums=1)(params, t)

# Step 3: vectorize over collocation points
du_dt_batch = jax.vmap(du_dt_single, in_axes=(None, 0))(params, t_colloc)
```

---

**Q: JAX PINN converges much faster than PyTorch PINN. Why?**

Two main reasons:
1. `jax.jit` eliminates Python overhead in the inner training loop
2. `vmap` over `grad` is more efficient than PyTorch's `create_graph=True` for large collocation batches

---

*Last updated: Week 0 · Add new issues below after each session.*
