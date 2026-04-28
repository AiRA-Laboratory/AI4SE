# Module 4 FAQ — Reinforcement Learning

> TAs: update this file after each session with real issues encountered.

---

## RL Fundamentals

**Q: What is the difference between an episode and a step?**

- **Step:** One interaction with the environment (`s → a → r, s'`)
- **Episode:** A full sequence of steps from reset until termination (`done=True`)

Log reward per episode (sum of all step rewards), not per step.

---

**Q: Why do we need a replay buffer?**

Without it, you train on each transition once and then discard it. Problems:
1. Highly correlated consecutive samples → unstable training
2. Rare but important experiences are forgotten

The replay buffer stores transitions and samples random mini-batches, breaking correlation and reusing data.

---

**Q: What is ε-greedy exploration? Why does ε decay?**

- With probability ε: take a random action (explore)
- With probability 1-ε: take the greedy action (exploit)

ε starts high (e.g., 1.0) so the agent explores broadly at the start. It decays toward a small value (e.g., 0.05) so the agent exploits learned knowledge later.

---

**Q: Why do we need a target network?**

Without it, both the prediction and the target change at every update → training target is a moving target → instability.

The target network is a copy of the Q-network that is updated every N steps (not every step), providing a stable training target.

---

**Q: My CartPole reward is 0 for hundreds of episodes then suddenly spikes. Is this normal?**

Yes. Early exploration gives random behavior (low reward). The agent needs enough experience to start exploiting a good policy. Keep training — it should stabilize.

---

## Environment and State Design

**Q: How do I know if my state is Markovian?**

The state should contain all information needed to predict the future. If you need memory of past states to make good decisions, your state is not Markovian. Add historical observations or use a recurrent policy.

---

**Q: My reward is very sparse (0 most of the time, 1 at success). The agent never learns. What to do?**

Options:
1. **Reward shaping:** Add intermediate dense rewards (e.g., distance to goal)
2. **Hindsight Experience Replay (HER):** Relabel failed episodes as if the achieved goal was the intended goal
3. **Curriculum learning:** Start with easier versions of the task
4. **Intrinsic motivation:** Add a curiosity bonus

---

**Q: When should I use DQN vs PPO vs SAC?**

| Algorithm | Action space | When to use |
|-----------|-------------|-------------|
| DQN | Discrete | Simple, discrete control |
| PPO | Discrete or Continuous | Robust general-purpose choice |
| SAC | Continuous | Sample-efficient continuous control |

For the tutorial, use DQN for CartPole (discrete). For Week 8 engineering demo, use PPO via Stable-Baselines3.

---

## Stable-Baselines3

**Q: How do I monitor training with SB3?**

```python
from stable_baselines3.common.callbacks import EvalCallback

eval_env = gym.make("CartPole-v1")
eval_callback = EvalCallback(eval_env, eval_freq=500,
                              best_model_save_path="./logs/",
                              log_path="./logs/")
model.learn(total_timesteps=50000, callback=eval_callback)
```

---

**Q: How do I load and test a trained SB3 model?**

```python
from stable_baselines3 import PPO
model = PPO.load("./logs/best_model")

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
```

---

*Last updated: Week 0 · Add new issues below after each session.*
