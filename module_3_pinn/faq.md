# Module 3 FAQ — PINN

> TAs: update this file after each session with real issues encountered.

---

## General PINN

**Q: What is the difference between supervised loss and physics loss?**

- **Supervised loss (`L_data`):** MSE between model output and measured/labeled data points. Like regular regression.
- **Physics loss (`L_residual`):** MSE of the governing equation's residual at collocation points — no labels needed, only the equation itself.

The PINN total loss combines both:
```python
loss = w_data * loss_data + w_physics * loss_residual + w_ic * loss_ic
```

---

**Q: How do I compute `du/dt` using autodiff?**

```python
t = t_colloc.requires_grad_(True)
u = model(t)
du_dt = torch.autograd.grad(
    u, t,
    grad_outputs=torch.ones_like(u),
    create_graph=True   # needed to backprop through this gradient
)[0]
```

`create_graph=True` is needed so you can compute second derivatives (`u_tt`, `u_xx`).

---

**Q: My PINN converges to zero everywhere. What's wrong?**

The model is taking the trivial solution `u = 0` which satisfies the residual (often) but not the IC/BC. Fix:
1. Increase the weight on IC/BC losses
2. Pre-train on IC/BC first (a few hundred steps) before adding physics loss
3. Check that IC/BC loss is actually decreasing

---

**Q: Should I use Adam or LBFGS?**

Typical strategy:
1. Train with Adam for 5000–10000 steps to get a reasonable initialization
2. Switch to LBFGS for fine-tuning (needs smaller dataset but converges better for smooth functions)

```python
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=50)
```

---

**Q: How many collocation points do I need?**

Start with:
- 1D problems: 500–2000 points
- 2D problems: 2000–10000 points

Distribute points uniformly or use Latin Hypercube Sampling. For Burgers equation, add more points near steep gradients.

---

## ODE PINN

**Q: My ODE PINN gives the right shape but is off in scale. Why?**

Check your initial condition loss weight. If `w_ic` is too small, the model ignores the initial value. Try increasing it (e.g., `w_ic = 10` vs `w_physics = 1`).

---

**Q: What ODE should I use for the tutorial?**

Good simple choices:
- **Exponential decay:** `du/dt = -k·u`, `u(0) = 1` (analytic: `u = exp(-kt)`)
- **Harmonic oscillator:** `d²u/dt² + ω²u = 0`, `u(0) = 1, u'(0) = 0`
- **Logistic growth:** `du/dt = r·u·(1 - u/K)`

---

## PDE PINN

**Q: My heat equation PINN is not matching the reference. Common causes?**

1. **Wrong normalization:** normalize `x ∈ [0,1]` and `t ∈ [0,1]`
2. **BC not enforced:** add explicit BC loss term
3. **Too few collocation points:** increase to 5000+
4. **Wrong alpha value:** double-check the diffusivity constant

---

**Q: What happens if I change the number of collocation points?**

More points → better physics enforcement → longer training. Try plotting the prediction error vs. number of collocation points as an experiment.

---

*Last updated: Week 0 · Add new issues below after each session.*
