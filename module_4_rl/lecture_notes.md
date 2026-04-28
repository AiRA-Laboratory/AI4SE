# Module 4 — Reinforcement Learning

**TA:** TA-4 · **Weeks:** 7–8

---

## Week 7 — RL Basics and DQN

### Key Concepts

**RL vs Supervised Learning**

| | Supervised Learning | Reinforcement Learning |
|--|---------------------|----------------------|
| Data source | Labeled dataset (fixed) | Environment interactions (dynamic) |
| Feedback | Immediate label | Delayed reward signal |
| Objective | Minimize prediction error | Maximize cumulative reward |
| Applications | Classification, regression | Control, game playing, optimization |

**Markov Decision Process (MDP)**

- `S` — State space: all possible environment states
- `A` — Action space: all possible actions
- `R(s, a)` — Reward function
- `P(s'|s, a)` — Transition dynamics
- `γ` — Discount factor (how much to value future rewards)

**Policy**

- `π(a|s)` — Probability of taking action `a` in state `s`
- Goal: find `π*` that maximizes expected discounted return:
  `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...`

**Q-Function (Action-Value Function)**

```
Q*(s, a) = expected return starting from state s, taking action a, then following π*
```

**DQN (Deep Q-Network)**

```python
# Core training step
with torch.no_grad():
    target = reward + gamma * target_net(next_state).max(1)[0] * (1 - done)

q_values = policy_net(state).gather(1, action)
loss = F.mse_loss(q_values, target.unsqueeze(1))
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Key components:
- **Experience replay buffer:** stores `(s, a, r, s', done)` transitions
- **Target network:** separate copy of Q-network, updated every N steps
- **ε-greedy exploration:** with probability ε choose random action, else greedy

### Common Issues (TA Reference)

| Issue | Symptom | Fix |
|-------|---------|-----|
| Reward not stabilizing | Reward oscillates forever | Check target network update frequency |
| ε not decaying | Model always explores, never exploits | Verify ε decay schedule |
| Episode vs step confusion | Student logs per-step, should log per-episode | Track `episode_reward` separately |
| Replay buffer too small | Training unstable | Increase buffer to ≥10000 |

---

## Week 8 — RL for Scientific/Engineering Problems

### Key Concepts

**Policy Gradient (REINFORCE)**
```
∇J(θ) = E[G_t · ∇ log π_θ(a_t|s_t)]
```
- Sample trajectories → compute returns → update policy
- Pros: works with continuous action spaces
- Cons: high variance → use baselines (actor-critic)

**Actor-Critic**
- **Actor:** policy network `π_θ(a|s)` — decides action
- **Critic:** value network `V_φ(s)` — estimates expected return
- Advantage: `A(s,a) = Q(s,a) - V(s)` — reduces variance

**RL for Engineering Applications**

| Application | State | Action | Reward |
|-------------|-------|--------|--------|
| PID tuning | System response, error | PID gains | Negative tracking error |
| Experiment design | Current data, uncertainty | Next measurement location | Information gain |
| Material design | Current composition | Parameter adjustment | Property improvement |
| Robot control | Joint angles, velocities | Joint torques | Task completion |

**State/Action/Reward Design Tips**
- State should be Markovian (enough info to predict future)
- Action space: discrete is simpler; continuous needs policy gradient
- Reward: dense is easier to learn; sparse is harder but more natural
- Normalize state and reward to `[-1, 1]` or `[0, 1]`

**Stable-Baselines3 Quickstart**
```python
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)
```

### Common Issues (TA Reference)

| Issue | Symptom | Fix |
|-------|---------|-----|
| Reward not defined properly | Agent learns wrong behavior | Carefully design sparse vs dense reward |
| Unstable training | Reward crashes after improving | Use smaller learning rate; normalize observations |
| Environment not resetting | Episode never ends | Check `env.reset()` is called after `done=True` |
| Reward scale too large | Q-values explode | Clip reward or normalize |

---

## Notebooks

| File | Description |
|------|-------------|
| `notebooks/01_rl_basics_dqn.ipynb` | CartPole with DQN from scratch |
| `notebooks/02_rl_engineering.ipynb` | RL for a toy engineering problem |

**Convention:**
- `*_clean.ipynb` = blank cells for students to fill
- `*_solution.ipynb` = TA solution with outputs

---

## Resources

- [UC Berkeley CS285](http://rail.eecs.berkeley.edu/deeprlcourse/)
- [PyTorch DQN tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/)
- [David Silver RL course (UCL)](https://www.davidsilver.uk/teaching/)
- [OpenAI Gymnasium](https://gymnasium.farama.org/)
