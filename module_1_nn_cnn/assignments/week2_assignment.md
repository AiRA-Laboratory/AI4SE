# Week 2 Assignment — CNN for Engineering Data

**Module:** CNN · **Due:** Before Week 3 Session A

---

## Task

Train a CNN on image-like engineering data and compare it with your Week 1 MLP.

---

## Steps

### 1. Choose a dataset (pick one)

- **Option A — Provided:** Use `data/week2_dataset/` (if provided by TA)
- **Option B — MNIST as surrogate:** Treat digit images as "sensor maps"
  ```python
  import torchvision
  dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                        transform=torchvision.transforms.ToTensor())
  ```
- **Option C — Your own:** Any 2D scientific image dataset (thermal, spectrogram, etc.)

### 2. Implement the CNN pipeline

- [ ] DataLoader with train/validation split
- [ ] CNN with at least 2 conv blocks + classifier head
- [ ] Training loop with loss + accuracy (or MSE) logging
- [ ] Plot: train/val loss and accuracy curves

### 3. Compare with MLP

Run the same task with a flat MLP (no conv layers). Fill in the table:

| Model | Val Accuracy / Val MSE | # Parameters | Training Time |
|-------|----------------------|--------------|---------------|
| MLP | | | |
| CNN | | | |

### 4. Visualize

- Plot at least 5 correct and 5 incorrect predictions (with true label vs predicted label)
- Optional: visualize first-layer filters or feature maps

### 5. Reflection (5–10 lines)

1. Where does CNN outperform MLP, and why?
2. When might MLP be sufficient (or better)?
3. What would you change in your architecture?

---

## Submission

- File name: `week2_<your_name>.ipynb`
- Submit to: `projects/student_groups/<your_group>/` or as directed by TA

---

## Grading Rubric

| Criterion | Points |
|-----------|--------|
| Correct DataLoader setup | 15 |
| CNN architecture implemented | 25 |
| MLP vs CNN comparison table | 20 |
| Visualization | 20 |
| Reflection quality | 20 |
| **Total** | **100** |
