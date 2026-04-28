# JAX for Scientific AI — Lecture Notes

**TA:** Lê Nguyễn Ngọc Vũ · **Optional Track**

---

## 1. JAX Core Transformations

JAX is built around four composable function transformations. Understanding these is the entire mental model.

### `jax.jit` — Just-In-Time Compilation

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sin(x) ** 2 + jnp.cos(x) ** 2

f_fast = jax.jit(f)

x = jnp.linspace(0, 2 * jnp.pi, 10000)
%timeit f(x)        # Python-traced, slow
%timeit f_fast(x)   # compiled, fast
```

**How it works:** On first call, JAX traces the function with abstract values and compiles it to XLA HLO. Subsequent calls skip Python entirely.

**Gotchas:**
- No Python side effects inside `jit` (no `print`, no list appends)
- Control flow on dynamic values needs `jax.lax.cond` / `jax.lax.fori_loop`
- Static arguments (e.g., `training=True`) must be declared with `static_argnums`

---

### `jax.grad` — Automatic Differentiation

```python
def loss(w, x, y):
    pred = w * x
    return jnp.mean((pred - y) ** 2)

# Gradient w.r.t. first argument (w) by default
grad_fn = jax.grad(loss)
dL_dw = grad_fn(w, x, y)

# Gradient w.r.t. multiple arguments
grad_fn = jax.grad(loss, argnums=(0,))      # w only
grad_fn = jax.grad(loss, argnums=(0, 1))    # w and x
```

**Higher-order derivatives:**
```python
# Second derivative: d²f/dx²
f = lambda x: x ** 3
df = jax.grad(f)
d2f = jax.grad(df)       # or jax.hessian(f) for the full Hessian
```

**Key difference from PyTorch:** `jax.grad` differentiates a *function*, not a computation graph. The function must return a scalar.

---

### `jax.vmap` — Vectorization

```python
def per_sample_loss(w, x_i, y_i):
    """Loss for a single (x, y) pair — no batch dimension."""
    pred = jnp.dot(w, x_i)
    return (pred - y_i) ** 2

# Vectorize over the second and third arguments (x and y)
batched_loss = jax.vmap(per_sample_loss, in_axes=(None, 0, 0))
losses = batched_loss(w, X_batch, y_batch)   # shape (batch_size,)
```

**Why this matters for scientific ML:**
- Write physics residuals for a *single collocation point*
- Use `vmap` to evaluate on the entire batch — clean, readable, fast
- No manual `reshape` or broadcasting tricks

---

### `jax.value_and_grad` — Efficient Combined Computation

```python
loss_and_grad = jax.value_and_grad(loss_fn)
loss_val, grads = loss_and_grad(params, batch)
```

Use this instead of calling `loss_fn` and `grad(loss_fn)` separately — avoids computing the forward pass twice.

---

## 2. Immutability and Pytrees

### No In-Place Operations

```python
# PyTorch: OK
x[0] = 5.0

# JAX: ERROR — arrays are immutable
x = x.at[0].set(5.0)   # returns new array
```

### Pytrees

JAX treats nested Python dicts/lists/tuples as *pytrees* — it automatically traverses them for `grad`, `jit`, `vmap`.

```python
# Parameters as a dict — JAX handles gradients automatically
params = {
    'W1': jnp.ones((3, 64)),
    'b1': jnp.zeros(64),
    'W2': jnp.ones((64, 1)),
    'b2': jnp.zeros(1),
}

# grad differentiates through the entire nested dict
grads = jax.grad(loss_fn)(params, x, y)
# grads has the same structure as params
```

---

## 3. Neural Networks with Flax + Optax

### Flax MLP

```python
import flax.linen as nn

class MLP(nn.Module):
    features: tuple = (64, 64, 1)

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
        return nn.Dense(self.features[-1])(x)
```

### Initialization

```python
model = MLP(features=(64, 64, 1))
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones((1, 1)))['params']
```

### Training Step with Optax

```python
import optax

optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, x, y):
    def loss_fn(params):
        pred = model.apply({'params': params}, x)
        return jnp.mean((pred - y) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
```

---

## 4. PINN in JAX

The key advantage of JAX for PINN: derivatives of the network w.r.t. inputs are just `jax.grad` calls on a plain function — no PyTorch `autograd.grad` boilerplate.

```python
def u(params, t):
    """Network output for a single scalar t."""
    return model.apply({'params': params}, t.reshape(1, 1))[0, 0]

# Derivative du/dt at a single point
du_dt = jax.grad(u, argnums=1)(params, t_scalar)

# Over a batch of collocation points using vmap
du_dt_batch = jax.vmap(jax.grad(u, argnums=1), in_axes=(None, 0))(params, t_colloc)
```

**Residual for ODE du/dt = -k·u:**
```python
def physics_residual(params, t, k):
    du = jax.grad(lambda t: u(params, t))(t)
    return du + k * u(params, t)

residual_batch = jax.vmap(physics_residual, in_axes=(None, 0, None))(params, t_colloc, k)
loss_physics = jnp.mean(residual_batch ** 2)
```

---

## 5. PyTorch vs JAX Side-by-Side (PINN Example)

| Step | PyTorch | JAX |
|------|---------|-----|
| Compute `du/dt` | `autograd.grad(u, t, create_graph=True)` | `jax.grad(u_fn)(t_scalar)` |
| Over batch | Manual loop or `vmap` workaround | `jax.vmap(jax.grad(u_fn))` |
| Training step | `loss.backward(); optim.step()` | `jax.value_and_grad(loss_fn)(params)` |
| JIT compile | `torch.compile(model)` | `@jax.jit` on any function |
| Second derivative | Nested `autograd.grad` | `jax.grad(jax.grad(f))` |

---

## Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| `print` inside `jit` | Prints only on first call | Use `jax.debug.print` |
| `nan` gradient | Appears with `jnp.where` | Use `jnp.where(cond, a, b)` not Python `if` |
| Shape error in `vmap` | Wrong `in_axes` | Print shapes before vmap |
| Slow first call | `jit` tracing | Expected; subsequent calls are fast |
| Parameter mutation | `AttributeError` | Arrays immutable — use `.at[].set()` |
