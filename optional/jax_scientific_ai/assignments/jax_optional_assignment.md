# Optional Assignment — JAX for Scientific AI

**TA:** Lê Nguyễn Ngọc Vũ · **This assignment is optional — for students who want to go deeper**

---

## Prerequisites

Complete `01_jax_basics.ipynb`, `02_flax_mlp.ipynb`, and `03_jax_pinn.ipynb` before starting.

---

## Part 1 — JAX Transformation Exercises (30 points)

### 1a. JIT speedup benchmark (10 pts)

Write a function `batch_physics_residual(X, t)` that computes the heat equation residual `u_t - alpha*u_xx` at a grid of `(x, t)` points using `jnp` operations (no neural network yet — just a reference function).

- Measure runtime with and without `jax.jit` on a grid of 10,000 points
- Report the speedup ratio
- Explain when JIT does NOT speed things up

### 1b. vmap derivative (10 pts)

For the function `f(x) = sin(2πx) * exp(-x)` on `x ∈ [0, 3]`:

1. Compute the analytic derivative by hand
2. Use `jax.grad` for a single point and verify
3. Use `jax.vmap(jax.grad(f))` to evaluate on 1000 points
4. Plot both curves and compute max absolute error

### 1c. Second derivative (10 pts)

For `f(x) = x³ - 2x² + x`:

1. Compute `f''(x)` analytically
2. Implement with `jax.grad(jax.grad(f))`
3. Implement with `jax.hessian(f)` (returns a matrix for scalar input)
4. Verify all three match

---

## Part 2 — Flax MLP vs PyTorch MLP (20 points)

Rerun `module_1_nn_cnn/notebooks/02_mlp_regression.ipynb` (PyTorch) and `optional/jax_scientific_ai/notebooks/02_flax_mlp.ipynb` (JAX) on the **same dataset with the same architecture**.

Fill in the comparison table:

| Metric | PyTorch | JAX/Flax |
|--------|---------|----------|
| Final val MSE | | |
| Training time (100 epochs) | | |
| Lines of code (training loop) | | |
| JIT speedup (1st vs subsequent epoch) | N/A | |

Write 3–5 lines: when would you choose JAX over PyTorch for a research project?

---

## Part 3 — JAX PINN Extension (50 points)

Extend `03_jax_pinn.ipynb` to solve the **harmonic oscillator ODE**:

```
d²u/dt² + ω²·u = 0,   u(0) = 1,   u'(0) = 0,   ω = 2π
Analytic: u(t) = cos(ωt)
```

### Requirements

- [ ] Model: Flax MLP with tanh activation, input=scalar `t`, output=scalar `u`
- [ ] Compute `d²u/dt²` using `jax.grad(jax.grad(u))`
- [ ] Physics residual: `u_tt + ω²u = 0` at collocation points
- [ ] IC losses: `u(0) - 1 = 0` and `u'(0) - 0 = 0` (two separate terms)
- [ ] Training: Adam, ≥ 15,000 steps
- [ ] Plot: PINN prediction vs `cos(2πt)` on `t ∈ [0, 2]`
- [ ] Report MSE vs analytic

### Comparison with PyTorch version

Write 3–5 lines comparing:
- Code clarity for second-order ODE (JAX vs PyTorch)
- Convergence speed (if you also ran the PyTorch version)
- Any difficulties you encountered

---

## Submission

- File name: `jax_optional_<your_name>.ipynb`
- Submit to: `projects/student_groups/<your_group>/` or as directed by TA

---

## Grading

| Part | Points |
|------|--------|
| Part 1: JAX transformations | 30 |
| Part 2: JAX vs PyTorch comparison | 20 |
| Part 3: JAX PINN for harmonic oscillator | 50 |
| **Total** | **100** |
