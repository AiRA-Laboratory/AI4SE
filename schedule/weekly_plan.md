# Weekly Schedule — AI for Scientific and Engineering

> 10-week short course · 2 sessions per week · PyTorch-first

---

## Format

Each week has two sessions:

- **Session A — Lecture + Concept** (90–120 min): PI leads theory, motivation, and research connections
- **Session B — Hands-on + Code Walkthrough** (120 min): TA leads code walkthrough, guided practice

---

## Week 0 — Preparation and Onboarding

**Goals:** Align on course objectives · Install shared environment · Assign TAs · Verify notebooks run

| Task | Owner | Status |
|------|-------|--------|
| Finalize student roster | PI | ☐ |
| Assign TAs to modules | PI | ☐ |
| Create GitHub repo | PI / TA | ☐ |
| Prepare shared environment | All TAs | ☐ |
| Test all notebooks locally or on server | All TAs | ☐ |
| Distribute student background survey | PI | ☐ |

**Deliverables:**
- `README.md` ✓
- `environment.yml` / `requirements.txt` ✓
- `schedule/weekly_plan.md` ✓
- `resources.md` ✓

---

## Week 1 — Neural Networks and MLP

**Session A — Lecture (PI)**
- Supervised learning for scientific/engineering tasks
- Tensor, model, loss, optimizer
- What is MLP
- Standard training loop
- Overfitting and validation

**Session B — Hands-on (TA-1)**
- PyTorch tensors and autograd
- Write MLP for regression from scratch
- Train / evaluate / save model

**Assignment:** Train an MLP on a toy scientific dataset; plot train/val loss; write 5-line reflection

**Notebook:** `module_1_nn_cnn/notebooks/01_pytorch_basics.ipynb`, `02_mlp_regression.ipynb`

---

## Week 2 — CNN for Scientific and Engineering Data

**Session A — Lecture (PI)**
- Convolution, kernel, receptive field
- CNN beyond natural images
- CNN for spectrograms, field maps, sensor maps
- Classification vs regression with CNN

**Session B — Hands-on (TA-1)**
- CNN with image-like data
- DataLoader and augmentation
- Visualize filters and feature maps

**Assignment:** Train CNN on small dataset; compare MLP vs CNN; explain why CNN is or isn't better

**Notebook:** `module_1_nn_cnn/notebooks/03_cnn_engineering.ipynb`

---

## Week 3 — Variational Autoencoder (VAE)

**Session A — Lecture (PI)**
- From autoencoder to VAE
- Latent space intuition
- Reconstruction vs regularization
- VAE in generative design and representation learning

**Session B — Hands-on (TA-2)**
- Code basic VAE in PyTorch
- Encoder → mean/logvar → reparameterization → decoder
- Latent interpolation and new sample generation

**Assignment:** Train VAE on toy dataset; visualize latent space; generate new samples; comment on reconstruction quality

**Notebook:** `module_2_generative/notebooks/01_vae_basics.ipynb`

---

## Week 4 — Diffusion Basics

**Session A — Lecture (PI)**
- Why diffusion models are powerful
- Noise-to-data intuition
- Forward process, reverse process, denoiser learning
- Diffusion for inverse design and generation

**Session B — Hands-on (TA-2)**
- Toy diffusion implementation
- Noising/denoising demo
- Small-scale sample generation

**Assignment:** Run toy diffusion demo; compare VAE vs diffusion (3 points); reflect on strengths and challenges

**Notebook:** `module_2_generative/notebooks/02_diffusion_toy.ipynb`

---

## Week 5 — PINN for ODE

**Session A — Lecture (PI)**
- Why supervised learning alone is insufficient for ODE/PDE
- Residual-based loss
- Initial and boundary conditions
- Collocation points
- Forward vs inverse problems

**Session B — Hands-on (TA-3)**
- PINN for a simple ODE
- Autodiff to compute derivatives
- Loss = data + physics + IC/BC terms

**Assignment:** Solve a simple ODE with PINN; compare against analytic or reference solution; explain role of physics loss

**Notebook:** `module_3_pinn/notebooks/01_pinn_ode.ipynb`

---

## Week 6 — PINN for PDE

**Session A — Lecture (PI)**
- Extending from ODE to PDE
- Heat equation or Burgers equation
- Sampling collocation points
- Common failure modes
- When to use / not use PINN

**Session B — Hands-on (TA-3)**
- PDE PINN demo
- Residual computation for spatiotemporal input
- BC/IC enforcement and evaluation

**Assignment:** Run PDE PINN notebook; vary number of collocation points; record the effect

**Notebook:** `module_3_pinn/notebooks/02_pinn_pde.ipynb`

---

## Week 7 — Reinforcement Learning Basics

**Session A — Lecture (PI)**
- Supervised learning vs RL
- State, action, reward, policy
- MDP intuition
- Value-based vs policy-based methods
- RL in robotics, control, parameter tuning

**Session B — Hands-on (TA-4)**
- Gymnasium environment
- CartPole with DQN
- Experience replay and target network

**Assignment:** Train CartPole agent; plot reward curve; write 5-line reflection on how RL differs from supervised learning

**Notebook:** `module_4_rl/notebooks/01_rl_basics_dqn.ipynb`

---

## Week 8 — RL for Scientific/Engineering Problems

**Session A — Lecture (PI)**
- Policy gradient / actor-critic intuition
- Model-based vs model-free
- RL for control, optimization, experiment design
- How to choose a suitable RL problem

**Session B — Hands-on (TA-4)**
- Continuous-control or parameter-tuning toy problem
- Stable-Baselines3 or minimal notebook
- Analyze failure cases

**Assignment:** Choose a toy problem and write state/action/reward formulation; run one baseline RL demo; identify a potential lab application

**Notebook:** `module_4_rl/notebooks/02_rl_engineering.ipynb`

---

## Week 9 — Mini-Project Implementation

**Format:** Groups of 2–3 students

**Session A:** Scope alignment with PI; finalize method and dataset choice
**Session B:** Implementation sprint; TA office hours

**Deliverables by end of week:**
- Project title
- 1-paragraph problem statement
- Method summary
- Preliminary results
- Next steps

**Suggested project directions:**
- Surrogate modeling with MLP/CNN
- VAE for latent representation or anomaly detection
- Toy diffusion model for generation or inverse design
- PINN for a small ODE/PDE problem
- RL for a control or tuning toy system

---

## Week 10 — Final Presentation

**Deliverables:**
- Clean code repository (one subfolder per group in `projects/student_groups/`)
- Short report (2–4 pages)
- Slides (5–7 slides)
- 10-minute presentation + 5-minute Q&A

**Evaluation:**

| Criterion | Weight |
|-----------|--------|
| Correct understanding of the method | 30% |
| Working and readable code | 25% |
| Result analysis | 20% |
| Connection to scientific/engineering problem | 15% |
| Clear presentation | 10% |
