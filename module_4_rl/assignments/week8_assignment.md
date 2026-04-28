# Week 8 Assignment — RL for Scientific/Engineering Problems

**Module:** RL · **Due:** Before Week 9 Session A

---

## Task

Formulate a toy engineering problem as an RL problem, run a baseline agent, and connect it to your lab research.

---

## Part 1: Problem Formulation

Choose **one** of the following toy problems OR propose your own (get TA approval):

**Option A — Thermostat control:**
- State: current temperature, target temperature
- Action: heater power {low, medium, high}
- Reward: -|current_temp - target_temp|

**Option B — Parameter tuning toy:**
- State: current loss value, gradient magnitude
- Action: learning rate adjustment {decrease, keep, increase}
- Reward: -loss

**Option C — 1D navigation:**
- State: position, velocity
- Action: force {left, none, right}
- Reward: -distance_to_goal

Fill in this formulation table for your chosen problem:

| Component | Your Definition |
|-----------|----------------|
| State `S` | |
| Action `A` | |
| Reward `R(s, a)` | |
| Episode termination condition | |
| Expected optimal behavior | |

### Part 2: Run a Baseline Agent

- Use Stable-Baselines3 PPO or the minimal notebook from `02_rl_engineering.ipynb`
- Train for at least 20,000 steps
- Plot: episode reward vs timestep
- Show 5 evaluation episodes with greedy policy

### Part 3: Analysis

- [ ] What is the most challenging part of your reward design?
- [ ] Does the agent converge? How can you tell?
- [ ] What is one limitation of your formulation?

### Part 4: Lab Connection (the most important part)

Write 1 paragraph (≥5 sentences):

**Identify a real problem in your lab research that could be formulated as an RL problem.** Address:
1. What is the state?
2. What is the action?
3. What is the reward?
4. What is the biggest challenge in applying RL to this problem?
5. Would you actually use RL or is there a simpler approach?

---

## Submission

- File name: `week8_<your_name>.ipynb`
- Submit to: `projects/student_groups/<your_group>/` or as directed by TA

---

## Grading Rubric

| Criterion | Points |
|-----------|--------|
| Problem formulation (S/A/R table) | 20 |
| Baseline agent trained and plotted | 25 |
| Analysis (3 questions) | 20 |
| Lab connection paragraph | 35 |
| **Total** | **100** |
