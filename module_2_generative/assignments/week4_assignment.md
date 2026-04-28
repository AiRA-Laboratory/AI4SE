# Week 4 Assignment — Diffusion Toy Demo

**Module:** Diffusion · **Due:** Before Week 5 Session A

---

## Task

Run a toy diffusion model, understand the forward/reverse process, and compare with VAE.

---

## Steps

### 1. Run the diffusion tutorial notebook

Use `notebooks/02_diffusion_toy.ipynb` provided by TA-2.

Make sure you can:
- [ ] Run the full notebook end-to-end without errors
- [ ] Understand what each major code block does

### 2. Annotate the notebook

Add markdown cells explaining these components in your own words:

1. **Noise schedule:** What is `beta`, `alpha`, `alpha_bar`?
2. **Forward process:** What does `q_sample(x0, t)` do?
3. **Denoising model:** What does the network predict?
4. **Training loop:** What is the loss and how is it computed?
5. **Sampling loop:** How are new samples generated?

### 3. Experiment: vary number of timesteps T

Run training with `T ∈ [50, 200, 500]` and note the effect on:
- Sample quality (visually)
- Training time

| T | Sample quality | Training time |
|---|----------------|---------------|
| 50 | | |
| 200 | | |
| 500 | | |

### 4. Comparison: VAE vs Diffusion

Write a short comparison (3 bullet points each):

**VAE strengths:**
- 
- 
- 

**Diffusion strengths:**
- 
- 
- 

**When would you choose diffusion over VAE in a scientific context?**
(2–3 sentences)

### 5. Reflection (5–10 lines)

1. What was the hardest part of understanding diffusion?
2. What potential application in your lab research could benefit from diffusion?

---

## Submission

- File name: `week4_<your_name>.ipynb`
- Submit to: `projects/student_groups/<your_group>/` or as directed by TA

---

## Grading Rubric

| Criterion | Points |
|-----------|--------|
| Notebook runs end-to-end | 20 |
| Annotations (5 components) | 30 |
| Timestep T experiment | 20 |
| VAE vs Diffusion comparison | 20 |
| Reflection | 10 |
| **Total** | **100** |
