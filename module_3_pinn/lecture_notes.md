# Module 3 — Physics-Informed Neural Networks (PINN)

**TA:** TA-3 · **Weeks:** 5–6

---

## Week 5 — PINN for ODE

### Key Concepts

**Why Not Pure Supervised Learning?**

| | Supervised ML | PINN |
|--|---------------|------|
| Data needed | Large labeled dataset | Small or no labeled data |
| Physics constraint | Ignored | Enforced via residual loss |
| Generalization | Interpolation only | Respects governing equations |
| Inverse problems | Hard | Natural formulation |

**PINN Loss Components**

```
L_total = w_data · L_data + w_physics · L_residual + w_ic · L_ic + w_bc · L_bc
```

- `L_data`: MSE at measured/labeled points
- `L_residual`: physics equation residual at collocation points
- `L_ic`: initial condition violation
- `L_bc`: boundary condition violation

**Computing ODE Residual with Autodiff**

For ODE `du/dt = f(u, t)`:
```python
t = torch.linspace(0, 1, 100).reshape(-1, 1).requires_grad_(True)
u = model(t)

du_dt = torch.autograd.grad(u, t,
                             grad_outputs=torch.ones_like(u),
                             create_graph=True)[0]

residual = du_dt - f(u, t)
loss_physics = torch.mean(residual**2)
```

**Forward vs Inverse Problems**

| Problem type | Known | Unknown |
|-------------|-------|---------|
| Forward | PDE + IC/BC | Solution `u(x,t)` |
| Inverse | Partial measurements | PDE parameters (e.g., viscosity, diffusivity) |

### Common Issues (TA Reference)

| Issue | Symptom | Fix |
|-------|---------|-----|
| Supervised vs physics loss confusion | Student uses only data loss | Explicitly show both loss terms separately |
| Wrong IC/BC | Solution drifts from true answer | Double-check initial/boundary conditions |
| Poor convergence | Loss doesn't decrease after ~5000 steps | Try Adam then LBFGS; adjust weights |
| Gradient computation fails | `RuntimeError: grad can be implicitly created only for scalar outputs` | Use `grad_outputs=torch.ones_like(u)` |

---

## Week 6 — PINN for PDE

### Key Concepts

**Extending to PDE**

For PDE `∂u/∂t + N[u] = 0` where `N` is a differential operator:

```python
# Input: (x, t) pairs (collocation points)
xt = torch.cat([x, t], dim=1).requires_grad_(True)
u = model(xt)

# Compute partial derivatives
u_t = grad(u, t)
u_x = grad(u, x)
u_xx = grad(u_x, x)

# Example: heat equation u_t = alpha * u_xx
residual = u_t - alpha * u_xx
```

**Collocation Points**

Points in the domain where the physics loss is evaluated:
- More points → better physics enforcement → higher compute cost
- Use uniform, random, or adaptive sampling
- Rule of thumb: 1000–10000 collocation points for 1D/2D problems

**Common PDEs in the Course**

| Equation | Form |
|----------|------|
| Heat (1D) | `u_t = α u_xx` |
| Wave (1D) | `u_tt = c² u_xx` |
| Burgers (1D) | `u_t + u u_x = ν u_xx` |
| Poisson (2D) | `u_xx + u_yy = f(x,y)` |

**When NOT to Use PINN**

- High-dimensional problems (>3D) → training very hard
- Very stiff equations → numerical instability
- Need very high accuracy → traditional solvers are more reliable
- Need fast repeated solves → use operator learning (DeepONet, FNO)

### Common Issues (TA Reference)

| Issue | Symptom | Fix |
|-------|---------|-----|
| Input/output confusion | Wrong shape for `(x, t)` input | Always print shapes; use named variables |
| BC not enforced | Solution correct inside but wrong at boundary | Add `L_bc` term with higher weight |
| Exploding loss early | Loss jumps up at start | Reduce learning rate; normalize input domain |
| Poor convergence on PDE | PINN fails to match reference | Use LBFGS after Adam warm-up |

---

## Notebooks

| File | Description |
|------|-------------|
| `notebooks/01_pinn_ode.ipynb` | PINN for simple ODE (e.g., harmonic oscillator) |
| `notebooks/02_pinn_pde.ipynb` | PINN for heat equation or Burgers equation |

**Convention:**
- `*_clean.ipynb` = blank cells for students to fill
- `*_solution.ipynb` = TA solution with outputs

---

## Resources

- [ETH Zurich / CAMLab PINN materials](https://camlab.ethz.ch/)
- [Ben Moseley PINN tutorial](https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/)
- Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). Physics-informed neural networks. *Journal of Computational Physics*.
