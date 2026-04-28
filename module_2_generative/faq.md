# Module 2 FAQ — Generative AI (VAE + Diffusion)

> TAs: update this file after each session with real issues encountered.

---

## VAE

**Q: What is KL divergence and why does it matter?**

KL divergence measures how far the learned latent distribution `q(z|x)` is from a standard Gaussian `N(0,I)`. Without this term, the latent space has no structure and you can't sample from it to generate new data.

Mathematically:
```
KL[q(z|x) || N(0,I)] = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
```

---

**Q: My KL loss is always 0 (or very small). Is that normal?**

No — this is called "posterior collapse." The decoder learns to ignore the latent variable entirely.

Common fixes:
1. Warm up KL weight: start with `beta=0` and gradually increase to `beta=1` over training
2. Use a smaller decoder capacity
3. Use a larger latent dimension

---

**Q: My reconstructions are blurry. Is my VAE broken?**

Blurry reconstructions are expected from VAE. The MSE/BCE reconstruction loss averages over all plausible reconstructions. This is a fundamental limitation of VAE. For sharper outputs, use:
- Larger latent dimension
- Perceptual loss (VGG features)
- Or switch to diffusion model

---

**Q: I get `nan` during training. What's wrong?**

Most likely `logvar` is exploding. Clamp it:
```python
logvar = logvar.clamp(-4, 4)
```
Also check that input data is normalized to `[0,1]`.

---

**Q: How do I visualize the latent space?**

If latent dim = 2:
```python
z = model.encode(x)[0]  # get mu
plt.scatter(z[:,0].detach(), z[:,1].detach(), c=labels, cmap='tab10')
```

If latent dim > 2, project with PCA or t-SNE:
```python
from sklearn.decomposition import PCA
z_2d = PCA(n_components=2).fit_transform(z.detach().numpy())
```

---

## Diffusion

**Q: The diffusion notebook has too much code. Where should I focus?**

Focus on these 4 components:
1. **Noise schedule:** `betas`, `alphas`, `alpha_bar`
2. **Forward process:** `q_sample(x0, t)` — add noise
3. **Model:** U-Net or MLP that predicts noise given `(x_noisy, t)`
4. **Training loop:** sample `t`, compute noisy input, predict noise, MSE loss

---

**Q: What is a timestep embedding?**

The denoiser needs to know which noise level it's operating at. Timestep `t` is converted to a vector (sinusoidal or learned embedding) and added to the network. Without it, the model can't tell how much noise to remove.

---

**Q: Sampling is very slow. Can I speed it up?**

For the tutorial demo, use fewer timesteps (e.g., `T=100` instead of `T=1000`). In production, use DDIM sampler which achieves good quality in 20–50 steps.

---

**Q: How is diffusion different from VAE for scientific applications?**

| Use case | Better choice |
|----------|--------------|
| Latent space analysis / anomaly detection | VAE |
| High-fidelity sample generation | Diffusion |
| Fast encoding of data | VAE |
| Inverse design with complex distributions | Diffusion |
| Interpretable latent space | VAE |

---

*Last updated: Week 0 · Add new issues below after each session.*
