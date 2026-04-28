# Week 3 Assignment — Variational Autoencoder

**Module:** VAE · **Due:** Before Week 4 Session A

---

## Task

Train a VAE on a toy dataset, explore the latent space, and generate new samples.

---

## Steps

### 1. Dataset (pick one)

- **Option A — MNIST digits** (standard benchmark)
- **Option B — 2D toy distribution:** Two concentric rings or a 2D mixture of Gaussians
- **Option C — Your own:** Any tabular or image dataset from your research

### 2. Implement VAE

Your notebook must include:

- [ ] Encoder: `x → (μ, log σ²)`
- [ ] Reparameterization: `z = μ + ε·σ`, `ε ~ N(0,I)`
- [ ] Decoder: `z → x̂`
- [ ] Loss: `L = reconstruction_loss + β * KL_loss`
- [ ] Training loop with loss logging

### 3. Evaluate

- Plot reconstruction loss and KL loss separately over epochs
- Show original vs reconstructed samples (at least 5 pairs)

### 4. Explore latent space

- Visualize 2D latent space (use PCA or t-SNE if `latent_dim > 2`)
- Perform latent interpolation between two samples:
  ```python
  z_interp = torch.lerp(z1, z2, steps=torch.linspace(0, 1, 10))
  ```
  Show the reconstructed sequence

### 5. Generate new samples

- Sample `z ~ N(0, I)` and decode
- Show at least 16 generated samples

### 6. Reflection (5–10 lines)

1. What does the KL term do to the latent space?
2. Are your generated samples realistic? Why or why not?
3. How might you use this VAE in your own research?

---

## Submission

- File name: `week3_<your_name>.ipynb`
- Submit to: `projects/student_groups/<your_group>/` or as directed by TA

---

## Grading Rubric

| Criterion | Points |
|-----------|--------|
| VAE architecture correct | 25 |
| Loss curves (reconstruction + KL) | 15 |
| Latent space visualization | 20 |
| Interpolation demo | 20 |
| Generated samples | 10 |
| Reflection | 10 |
| **Total** | **100** |
