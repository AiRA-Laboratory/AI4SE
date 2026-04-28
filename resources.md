# Course Resources

---

## Module 1 — Neural Networks and CNN

### Lectures and Courses
- [MIT 6.S191 — Introduction to Deep Learning](http://introtodeeplearning.com/)
- [Stanford CS231n — Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)

### Key Papers
- LeCun et al. (1989). Backpropagation applied to handwritten zip code recognition.
- He et al. (2016). Deep residual learning for image recognition. *CVPR*.

### Tools
- `torchinfo` — model summary: `pip install torchinfo`
- `tensorboard` — training visualization

---

## Module 2 — Generative AI (VAE + Diffusion)

### Courses
- [Stanford CS236 — Deep Generative Models](https://deepgenerativemodels.github.io/)
- [HuggingFace Diffusion Course](https://huggingface.co/learn/diffusion-course)
- [DeepLearning.AI — How Diffusion Models Work](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/)

### Code References
- [PyTorch VAE implementations](https://github.com/AntixK/PyTorch-VAE)
- [The Annotated Diffusion Model (HuggingFace blog)](https://huggingface.co/blog/annotated-diffusion)
- [DDPM minimal implementation](https://github.com/lucidrains/denoising-diffusion-pytorch)

### Key Papers
- Kingma & Welling (2013). Auto-encoding variational Bayes. *ICLR 2014*.
- Ho et al. (2020). Denoising diffusion probabilistic models. *NeurIPS*.
- Song et al. (2021). Score-based generative modeling through stochastic differential equations. *ICLR*.

---

## Module 3 — PINN

### Courses and Tutorials
- [Ben Moseley — So, what is a physics-informed neural network?](https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/)
- [ETH Zurich CAMLab PINN materials](https://camlab.ethz.ch/teaching.html)
- [NVIDIA Modulus (production PINN framework)](https://developer.nvidia.com/modulus)

### Key Papers
- Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*.
- Karniadakis et al. (2021). Physics-informed machine learning. *Nature Reviews Physics*.

### Advanced Topics
- DeepONet: Lu et al. (2021). Learning nonlinear operators via DeepONet. *Nature Machine Intelligence*.
- FNO: Li et al. (2021). Fourier neural operator for parametric partial differential equations. *ICLR*.

---

## Module 4 — Reinforcement Learning

### Courses
- [UC Berkeley CS285 — Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/)
- [David Silver RL Course (UCL / DeepMind)](https://www.davidsilver.uk/teaching/)
- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)

### Code References
- [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [OpenAI Gymnasium](https://gymnasium.farama.org/)

### Key Papers
- Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
- Schulman et al. (2017). Proximal policy optimization algorithms. *arXiv*.
- Haarnoja et al. (2018). Soft actor-critic: Off-policy maximum entropy deep RL. *ICML*.

---

## Optional Advanced Tracks

### JAX for Scientific AI
- [Ben Moseley JAX Workshop](https://github.com/benmoseley/jax-tutorial)
- [Official JAX Quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
- [Flax (neural networks in JAX)](https://flax.readthedocs.io/)

### Advanced Diffusion
- [Improved DDPM (OpenAI)](https://arxiv.org/abs/2102.09672)
- [Latent Diffusion / Stable Diffusion](https://arxiv.org/abs/2112.10752)
- [DDIM fast sampling](https://arxiv.org/abs/2010.02502)

### Operator Learning
- [DeepONet tutorial](https://deepxde.readthedocs.io/en/latest/demos/operator/)
- [FNO implementation](https://github.com/neuraloperator/neuraloperator)

---

## General Deep Learning References

- Goodfellow, Bengio, Courville. *Deep Learning*. MIT Press. [Free online](https://www.deeplearningbook.org/)
- Bishop & Bishop. *Deep Learning: Foundations and Concepts*. Springer. (2024)
- [Papers With Code](https://paperswithcode.com/) — state-of-the-art benchmarks with code

---

## Environment and Tools

| Tool | Purpose | Install |
|------|---------|---------|
| PyTorch | Core framework | `conda install pytorch` |
| Gymnasium | RL environments | `pip install gymnasium` |
| Stable-Baselines3 | RL algorithms | `pip install stable-baselines3` |
| diffusers | HuggingFace diffusion | `pip install diffusers` |
| torchinfo | Model summary | `pip install torchinfo` |
| TensorBoard | Training visualization | `pip install tensorboard` |
| matplotlib / seaborn | Plotting | `pip install matplotlib seaborn` |
