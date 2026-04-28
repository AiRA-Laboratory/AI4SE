# Module 1 FAQ — NN / CNN

> TAs: update this file after each session with real issues encountered.

---

## PyTorch Basics

**Q: Why do I get `RuntimeError: mat1 and mat2 shapes cannot be multiplied`?**

Your tensor shape doesn't match the layer's expected input. Print the shape at each step:
```python
x = torch.randn(32, 784)
print(x.shape)  # torch.Size([32, 784])
out = nn.Linear(784, 128)(x)
```
The input to `nn.Linear(in, out)` must be `(batch, in)`.

---

**Q: My loss is `nan` after a few steps. What's wrong?**

Most common causes:
1. Learning rate too high → try `1e-3` or `1e-4`
2. Missing `zero_grad()` → gradients accumulate
3. Log of zero in loss (e.g., `log(0)` in cross-entropy) → check labels and predictions

---

**Q: Should I use `model.eval()` during validation?**

Yes, always. Without it:
- Dropout is still active (adds noise to predictions)
- BatchNorm uses batch statistics instead of running statistics

Always pair it with `torch.no_grad()` to save memory:
```python
model.eval()
with torch.no_grad():
    val_loss = criterion(model(x_val), y_val)
```

---

**Q: My model trains fine on CPU but crashes on GPU.**

Check that both the model and the data are on the same device:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x, y = x.to(device), y.to(device)
```

---

## CNN

**Q: What should my input tensor shape be for `nn.Conv2d`?**

`Conv2d` expects `(batch_size, channels, height, width)`.

If your data is grayscale `(B, H, W)`, add a channel dimension:
```python
x = x.unsqueeze(1)  # (B, 1, H, W)
```

---

**Q: How do I calculate the output size after conv + pool?**

Formula for one dimension:
```
out = floor((in + 2*padding - kernel_size) / stride) + 1
```

Or just use `torchinfo`:
```python
from torchinfo import summary
summary(model, input_size=(1, 1, 64, 64))
```

---

**Q: My CNN is overfitting quickly. What can I do?**

1. Add `nn.Dropout2d(p=0.3)` after activations
2. Add data augmentation (`torchvision.transforms`)
3. Add `nn.BatchNorm2d` after conv layers
4. Reduce model depth/width
5. Increase training data if possible

---

## Environment

**Q: `import torch` fails after `conda env create`.**

Make sure you activated the environment:
```bash
conda activate ai4science
python -c "import torch; print(torch.__version__)"
```

If PyTorch is not found, reinstall from [pytorch.org](https://pytorch.org) selecting your OS and CUDA version.

---

*Last updated: Week 0 · Add new issues below after each session.*
