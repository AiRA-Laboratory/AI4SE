# Week 1 Assignment — MLP for Scientific Regression

**Module:** NN/MLP · **Due:** Before Week 2 Session A

---

## Task

Train an MLP to solve a simple scientific regression problem using PyTorch.

---

## Steps

### 1. Choose a dataset (pick one)

- **Option A — Toy physics:** Predict kinetic energy from mass and velocity
  ```python
  # Generate synthetic data
  import torch
  N = 1000
  mass = torch.rand(N, 1) * 10       # kg
  velocity = torch.rand(N, 1) * 20   # m/s
  KE = 0.5 * mass * velocity**2      # target
  ```

- **Option B — Provided dataset:** Use `data/week1_dataset.csv` (if provided by TA)

- **Option C — Your own:** Any scalar regression dataset from your research

### 2. Implement the training pipeline

Your notebook must contain:

- [ ] Data loading and normalization
- [ ] MLP model definition with at least 2 hidden layers
- [ ] Training loop (min. 100 epochs)
- [ ] Validation split (80/20 or 70/30)
- [ ] Plot of training and validation loss over epochs

### 3. Evaluate the model

- Compute and report MSE and MAE on the validation set
- Plot: predicted vs. true values (scatter plot)

### 4. Reflection (write 5–10 lines)

Answer these questions in a markdown cell at the end of your notebook:

1. Does your model overfit? How can you tell?
2. What happens when you change the learning rate?
3. What happens when you add or remove a hidden layer?

---

## Submission

- File name: `week1_<your_name>.ipynb`
- Submit to: `projects/student_groups/<your_group>/` or as directed by TA

---

## Grading Rubric

| Criterion | Points |
|-----------|--------|
| Correct data pipeline | 20 |
| MLP implemented correctly | 25 |
| Loss curves plotted | 20 |
| Evaluation metrics reported | 20 |
| Reflection quality | 15 |
| **Total** | **100** |
