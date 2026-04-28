# Week 6 Assignment — PINN for PDE

**Module:** PINN · **Due:** Before Week 7 Session A

---

## Task

Solve a simple PDE using PINN. Vary the number of collocation points and record the effect.

---

## Chosen PDE (TA selects one for the class)

**Option A — 1D Heat Equation:**
```
∂u/∂t = α · ∂²u/∂x²,   x ∈ [0,1],   t ∈ [0,1]
BC: u(0,t) = 0,   u(1,t) = 0
IC: u(x,0) = sin(πx)
Analytic: u(x,t) = sin(πx) · exp(-α π² t),   α = 0.1
```

**Option B — 1D Burgers Equation:**
```
∂u/∂t + u · ∂u/∂x = ν · ∂²u/∂x²,   x ∈ [-1,1],   t ∈ [0,1]
BC: u(-1,t) = u(1,t) = 0
IC: u(x,0) = -sin(πx)
ν = 0.01/π
```

---

## Steps

### 1. Implement PDE PINN

- [ ] Model input: `[x, t]` concatenated
- [ ] Compute `u_t`, `u_x`, `u_xx` using autodiff
- [ ] Physics residual loss at collocation points
- [ ] BC loss (enforce boundary values)
- [ ] IC loss (enforce initial condition)
- [ ] Train with Adam ≥ 10000 steps

### 2. Visualize

- Heatmap: predicted `u(x,t)` over the full domain `(x,t)`
- If analytic solution available: heatmap of the absolute error
- Line plots: `u(x, t*)` at 3 fixed time slices vs reference

### 3. Collocation point experiment

Run with `N_colloc ∈ [100, 500, 2000, 5000]`. Record:

| N_colloc | Final physics loss | Max absolute error |
|----------|-------------------|--------------------|
| 100 | | |
| 500 | | |
| 2000 | | |
| 5000 | | |

Plot: Max absolute error vs N_colloc

### 4. Reflection (5–10 lines)

1. How does N_colloc affect accuracy? Is there a point of diminishing returns?
2. What are the main failure modes you observed?
3. Would you use PINN for a real problem in your lab? Why or why not?

---

## Submission

- File name: `week6_<your_name>.ipynb`
- Submit to: `projects/student_groups/<your_group>/` or as directed by TA

---

## Grading Rubric

| Criterion | Points |
|-----------|--------|
| PDE residual computed correctly (u_t, u_x, u_xx) | 25 |
| BC and IC losses enforced | 20 |
| Domain heatmap visualization | 20 |
| Collocation point experiment | 25 |
| Reflection | 10 |
| **Total** | **100** |
