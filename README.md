# AI for Scientific and Engineering — Short Course

> **Internal lab training course** · 10 weeks · PyTorch-first · Hands-on + Mini-project

---

## Course Overview

This course is designed for lab members (senior undergraduates, master's, and PhD students) following the progression:

```
Function Approximation → Generative Modeling → Physics-Informed Learning → Decision & Control
```

**Core goals:**
- Understand the fundamentals of each method family
- Read and modify code independently
- Run guided tutorials with confidence
- Connect learned methods to real lab research problems

---

## Course Structure

| Week | Module | Focus | Output |
|------|--------|-------|--------|
| 0 | Preparation | Setup, orientation, environment | Repo cloned & notebooks running |
| 1 | NN/MLP | PyTorch basics, MLP, training loop | MLP regression |
| 2 | CNN | CNN, feature maps, engineering data | CNN mini task |
| 3 | VAE | AE/VAE, latent space | VAE notebook |
| 4 | Diffusion | Diffusion intuition + toy implementation | Diffusion toy demo |
| 5 | PINN-1 | ODE, residual loss, autodiff | PINN for ODE |
| 6 | PINN-2 | PDE, collocation, BC/IC | PINN for PDE |
| 7 | RL-1 | RL basics, MDP, DQN intuition | CartPole demo |
| 8 | RL-2 | Policy gradient / actor-critic + engineering view | RL mini experiment |
| 9 | Project | Implementation | Draft results |
| 10 | Project | Final presentation | Code + report + slides |

---

## Modules

| Folder | Content | TA |
|--------|---------|-----|
| [`module_1_nn_cnn/`](module_1_nn_cnn/) | PyTorch basics, MLP, CNN | TA-1 |
| [`module_2_generative/`](module_2_generative/) | VAE, Diffusion | TA-2 |
| [`module_3_pinn/`](module_3_pinn/) | PINN for ODE & PDE | TA-3 |
| [`module_4_rl/`](module_4_rl/) | Reinforcement Learning | TA-4 |
| [`projects/`](projects/) | Mini-project templates & student groups | All TAs |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<org>/ai-science-short-course.git
cd ai-science-short-course
```

### 2. Set up the environment

**Option A — conda (recommended):**

```bash
conda env create -f environment.yml
conda activate ai4science
```

**Option B — pip:**

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter

```bash
jupyter lab
```

Navigate to any module folder and open a notebook to start.

---

## Repository Layout

```
ai-science-short-course/
├── README.md                    ← you are here
├── requirements.txt             ← pip dependencies
├── environment.yml              ← conda environment
├── schedule/
│   └── weekly_plan.md           ← detailed weekly schedule
├── module_1_nn_cnn/
│   ├── lecture_notes.md
│   ├── slides/                  ← PI/TA slides (PDF or PPTX)
│   ├── notebooks/               ← Jupyter notebooks
│   ├── assignments/             ← weekly assignments
│   └── faq.md                   ← common issues & solutions
├── module_2_generative/
│   └── ...
├── module_3_pinn/
│   └── ...
├── module_4_rl/
│   └── ...
└── projects/
    ├── templates/               ← proposal & report templates
    └── student_groups/          ← one subfolder per group
```

---

## TA Responsibilities

Each module has one TA responsible for:

- Running all tutorials end-to-end before the session
- Preparing a **clean version** (no outputs) and a **solution version** of each notebook
- Walking through code line-by-line during hands-on sessions
- Supporting debugging and collecting FAQs after each session
- Reporting weak areas to the PI

**TA assignments:**

| TA | Module | Weeks |
|----|--------|-------|
| TA-1 | Neural Networks / CNN | 1–2 |
| TA-2 | Generative AI (VAE + Diffusion) | 3–4 |
| TA-3 | PINN | 5–6 |
| TA-4 | Reinforcement Learning | 7–8 |

---

## Student Responsibilities

- Review pre-study materials before each session
- Participate actively in hands-on labs
- Submit weekly mini assignments
- Join a mini-project group (2–3 people) in Week 9

---

## Mini-Project Evaluation

| Criterion | Weight |
|-----------|--------|
| Correct understanding of the method | 30% |
| Working and readable code | 25% |
| Result analysis | 20% |
| Connection to scientific/engineering problem | 15% |
| Clear presentation | 10% |

---

## External Resources

| Module | Key Resources |
|--------|--------------|
| NN/CNN | MIT 6.S191, Stanford CS231n, PyTorch official tutorials |
| VAE/Diffusion | Stanford CS236, PyTorch VAE example, HuggingFace Diffusion Course |
| PINN | ETH Zurich CAMLab, Ben Moseley PINN materials |
| RL | UC Berkeley CS285, PyTorch DQN tutorial, Stable-Baselines3 |

---

## Optional Advanced Tracks

- JAX for scientific AI
- Advanced diffusion models
- Advanced PINN / Operator Learning (DeepONet, FNO)

---

*For questions, contact the PI or the respective TA for each module.*
