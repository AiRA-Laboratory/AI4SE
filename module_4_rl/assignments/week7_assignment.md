# Week 7 Assignment — DQN on CartPole

**Module:** RL · **Due:** Before Week 8 Session A

---

## Task

Train a DQN agent to solve the CartPole-v1 environment and analyze the learning curve.

---

## Steps

### 1. Implement or run DQN

Use `notebooks/01_rl_basics_dqn.ipynb`. Your notebook must include:

- [ ] Environment setup with `gymnasium.make("CartPole-v1")`
- [ ] Replay buffer (circular buffer storing `(s, a, r, s', done)`)
- [ ] Q-network and target network
- [ ] ε-greedy action selection with decay
- [ ] Training loop: sample batch → compute target → update Q-network
- [ ] Target network update every N steps

### 2. Plot learning curve

- Plot: episode reward vs episode number (smoothed with rolling mean window=20)
- Mark where ε falls below 0.1 on the plot

### 3. Ablation: target network update frequency

Run DQN with `target_update_freq ∈ [10, 100, 500]`. Plot all three reward curves on the same figure.

| target_update_freq | Episodes to reach reward > 150 |
|--------------------|-------------------------------|
| 10 | |
| 100 | |
| 500 | |

### 4. Demonstrate the trained agent

Show a short demo: print episode reward for 10 evaluation episodes with ε = 0 (greedy policy).

### 5. Reflection (5–10 lines)

1. How is the RL training loop different from the supervised learning loop?
2. What role does the replay buffer play?
3. What happens if you remove the target network?

---

## Submission

- File name: `week7_<your_name>.ipynb`
- Submit to: `projects/student_groups/<your_group>/` or as directed by TA

---

## Grading Rubric

| Criterion | Points |
|-----------|--------|
| DQN implementation (replay, Q-net, target-net) | 30 |
| Learning curve plot | 20 |
| Target update frequency ablation | 25 |
| Greedy evaluation demo | 10 |
| Reflection | 15 |
| **Total** | **100** |
