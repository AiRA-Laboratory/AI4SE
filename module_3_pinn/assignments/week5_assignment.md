# Week 5 Assignment — PINN for ODE

**Module:** PINN · **Due:** Before Week 6 Session A

---

## Task

Solve a simple ODE using a Physics-Informed Neural Network and compare with the analytic solution.

---

## Chosen ODE (pick one)

**Option A — Exponential decay:**
```
du/dt = -k·u,   u(0) = 1,   k = 1.0
Analytic solution: u(t) = exp(-t)
```

**Option B — Simple harmonic oscillator:**
```
d²u/dt² + ω²·u = 0,   u(0) = 1,   u'(0) = 0,   ω = 2π
Analytic solution: u(t) = cos(ωt)
```

**Option C — Logistic growth:**
```
du/dt = r·u·(1 - u/K),   u(0) = 0.1,   r = 1.0,   K = 1.0
```

---

## Steps

### 1. Implement the PINN

- [ ] Define MLP (≥3 layers, tanh activation recommended for smooth ODEs)
- [ ] Sample collocation points `t ∈ [0, T]`
- [ ] Implement physics residual using `torch.autograd.grad`
- [ ] Implement IC loss
- [ ] Total loss: `L = w_physics * L_residual + w_ic * L_ic`
- [ ] Train with Adam (≥5000 steps)

### 2. Visualize results

- Plot: PINN prediction vs analytic solution on `t ∈ [0, T]`
- Plot: physics residual loss and IC loss over training steps (separately)

### 3. Ablation experiment

Run training with **two different loss weight settings** and compare:

| Setting | w_physics | w_ic | Final MSE (vs analytic) |
|---------|-----------|------|------------------------|
| Setting A | 1.0 | 1.0 | |
| Setting B | 1.0 | 10.0 | |

### 4. Reflection (5–10 lines)

1. What role does the physics loss play? What happens if you set `w_physics = 0`?
2. How does the PINN compare to just interpolating the data?
3. What would change if you had no analytic solution to compare against?

---

## Submission

- File name: `week5_<your_name>.ipynb`
- Submit to: `projects/student_groups/<your_group>/` or as directed by TA

---

## Grading Rubric

| Criterion | Points |
|-----------|--------|
| Physics residual computed correctly | 30 |
| IC loss implemented | 15 |
| Comparison plot (PINN vs analytic) | 20 |
| Loss weight ablation | 20 |
| Reflection | 15 |
| **Total** | **100** |
