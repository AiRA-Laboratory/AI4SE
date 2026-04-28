# Module 2 — Generative AI: VAE and Diffusion

**TA:** TA-2 · **Weeks:** 3–4

---

## Week 3 — Variational Autoencoder (VAE)

### Key Concepts

**From Autoencoder to VAE**

| | Autoencoder | VAE |
|--|-------------|-----|
| Latent | Deterministic point | Distribution `q(z|x) = N(μ, σ²)` |
| Sampling | Not possible | Sample `z ~ N(μ, σ²)` |
| Loss | Reconstruction only | Reconstruction + KL divergence |

**Latent Space**
- Low-dimensional representation capturing essential structure
- VAE forces latent space to be roughly Gaussian → smooth interpolation
- Applications: design space exploration, anomaly detection, dimensionality reduction

**Loss Function**
```
L = E[||x - x̂||²]  +  β · KL[q(z|x) || N(0,I)]
     reconstruction      regularization
```
- KL term ensures latent space is well-organized
- β controls the tradeoff (β-VAE)

**Reparameterization Trick**
```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)   # sample noise
    return mu + eps * std          # differentiable sampling
```

### Common Issues (TA Reference)

| Issue | Symptom | Fix |
|-------|---------|-----|
| KL collapse | KL → 0, reconstruction poor | Warm up β gradually |
| Posterior collapse | Decoder ignores latent z | Use smaller β, check architecture |
| Blurry reconstruction | Output is average-looking | Normal for VAE; try higher latent dim |
| Numerical instability | `nan` in logvar | Clamp logvar: `logvar.clamp(-4, 4)` |

---

## Week 4 — Diffusion Basics

### Key Concepts

**Intuition**
- Forward process: progressively add Gaussian noise until data = pure noise
- Reverse process: learn to denoise step by step

**Forward Process** (fixed, no parameters)
```
q(xₜ | xₜ₋₁) = N(xₜ; √(1-βₜ) xₜ₋₁, βₜI)
```
- After enough steps, `xₜ ≈ N(0, I)`

**Reverse Process** (learned)
```
p_θ(xₜ₋₁ | xₜ) = N(xₜ₋₁; μ_θ(xₜ, t), σₜ²I)
```
- Neural network (U-Net style) predicts noise at each timestep

**Training Objective** (simplified)
```python
noise = torch.randn_like(x0)
x_noisy = sqrt_alpha_bar[t] * x0 + sqrt_one_minus_alpha_bar[t] * noise
predicted_noise = model(x_noisy, t)
loss = F.mse_loss(predicted_noise, noise)
```

**Diffusion vs VAE**

| | VAE | Diffusion |
|--|-----|-----------|
| Generation quality | Moderate | High |
| Speed | Fast | Slow (many steps) |
| Latent space | Explicit, interpretable | Implicit |
| Training | Stable | More complex |
| Scientific use | Design space, anomaly | Inverse design, data augmentation |

### Common Issues (TA Reference)

| Issue | Symptom | Fix |
|-------|---------|-----|
| Code too long for tutorial | Students get lost | Use minimal DDPM; strip to core loop |
| Timestep confusion | Wrong noise schedule | Print shapes at each step |
| Slow sampling | Generation takes too long | Reduce `T` (e.g., 100 instead of 1000) for demo |
| Poor quality samples | Blurry or incoherent | Check noise schedule `betas` |

---

## Notebooks

| File | Description |
|------|-------------|
| `notebooks/01_vae_basics.ipynb` | VAE from scratch: encoder, decoder, latent space |
| `notebooks/02_diffusion_toy.ipynb` | Minimal DDPM on toy 2D or 1D data |

**Convention:**
- `*_clean.ipynb` = blank cells for students to fill
- `*_solution.ipynb` = TA solution with outputs

---

## Resources

- [Stanford CS236 — Deep Generative Models](https://cs236g.stanford.edu/)
- [PyTorch VAE example](https://github.com/AntixK/PyTorch-VAE)
- [HuggingFace Diffusion Course](https://huggingface.co/learn/diffusion-course)
- [The Annotated Diffusion Model (blog)](https://huggingface.co/blog/annotated-diffusion)
- DeepLearning.AI diffusion short course
