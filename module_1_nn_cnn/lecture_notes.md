# Module 1 — Neural Networks and CNN

**TA:** Nguyễn Đình Hải · **Weeks:** 1–2

---

## Week 1 — Neural Network Basics with PyTorch

### Key Concepts

**Supervised Learning for Scientific Tasks**
- Input: measurements, sensor data, simulation snapshots
- Output: physical quantities, labels, regression targets
- Loss: MSE for regression, cross-entropy for classification

**Tensors and Autograd**
- PyTorch tensor = NumPy array with GPU support + autograd graph
- `requires_grad=True` enables gradient tracking
- `.backward()` computes gradients via reverse-mode autodiff

**MLP Architecture**
```
Input → Linear → Activation → Linear → Activation → ... → Output
```
- `nn.Linear(in, out)` applies `y = xW^T + b`
- Common activations: `ReLU`, `Tanh`, `GELU`
- Last layer: no activation for regression; `Softmax` for classification

**Standard Training Loop**
```python
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x_batch)
    loss = criterion(output, y_batch)
    loss.backward()
    optimizer.step()
```

**Overfitting and Validation**
- Train loss ↓ while val loss ↑ = overfitting
- Remedies: dropout, weight decay (`optimizer weight_decay`), more data, simpler model
- Always evaluate with `model.eval()` and `torch.no_grad()`

### Common Issues (TA Reference)

| Issue | Symptom | Fix |
|-------|---------|-----|
| Shape mismatch | `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Check `print(tensor.shape)` at each layer |
| Wrong device | `RuntimeError: Expected all tensors to be on the same device` | `.to(device)` for model and data |
| Exploding loss | Loss = `nan` or very large | Gradient clipping; lower learning rate |
| `train`/`eval` confusion | Dropout active during evaluation | `model.eval()` + `torch.no_grad()` |

---

## Week 2 — CNN for Scientific and Engineering Data

### Key Concepts

**Convolution**
- Kernel slides over input → detects local patterns
- Parameters: `kernel_size`, `stride`, `padding`
- Fewer parameters than fully connected; weight sharing

**Feature Maps**
- Output of a conv layer = feature map
- Early layers: low-level features (edges, gradients)
- Later layers: high-level features (shapes, patterns)

**CNN Beyond Natural Images**
| Data type | How to feed to CNN |
|-----------|--------------------|
| Vibration spectrogram | 2D image (time × frequency) |
| Thermal map | 2D field |
| Time series | 1D conv or reshape to 2D |
| Sensor array | 2D spatial layout |

**Typical CNN Block**
```python
nn.Sequential(
    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
    nn.BatchNorm2d(out_ch),
    nn.ReLU(),
    nn.MaxPool2d(2),
)
```

**Classification vs Regression**
- Classification: `nn.CrossEntropyLoss` + no final activation
- Regression: `nn.MSELoss` + linear output head

### Common Issues (TA Reference)

| Issue | Symptom | Fix |
|-------|---------|-----|
| Wrong channel order | Shape `(B, H, W)` instead of `(B, C, H, W)` | `unsqueeze(1)` or check DataLoader |
| Normalization skipped | Slow convergence, large initial loss | Normalize to `[0,1]` or `mean/std` |
| Pool too aggressive | Feature map → 1×1 after pooling | Reduce pool or add adaptive pool |
| Wrong output size | Final linear dim mismatch | Use `torchinfo.summary(model, input_size=...)` |

---

## Notebooks

| File | Description |
|------|-------------|
| `notebooks/01_pytorch_basics.ipynb` | Tensors, autograd, GPU basics |
| `notebooks/02_mlp_regression.ipynb` | MLP for toy scientific regression |
| `notebooks/03_cnn_engineering.ipynb` | CNN for image-like engineering data |

**Convention:**
- `*_clean.ipynb` = blank cells for students to fill
- `*_solution.ipynb` = TA solution with outputs

---

## Resources

- [PyTorch official tutorials](https://pytorch.org/tutorials/)
- [MIT 6.S191](http://introtodeeplearning.com/)
- [Stanford CS231n](http://cs231n.stanford.edu/)
- `torchinfo` for model summary: `pip install torchinfo`
