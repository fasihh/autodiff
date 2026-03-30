# autodiff

A minimal NumPy/CuPy-based autodiff playground with:

- A scalar/tensor computation graph (`Node`)
- Reverse-mode automatic differentiation (`backward`)
- Basic neural-network modules (`Linear`, `RNN`, activations)
- A simple optimizer (`SGD`)
- A small batch loader (`DataLoader`)

This project is intentionally lightweight and educational, not a full deep-learning framework.

## Features

- Dynamic computation graph through operator overloading
- CPU support via NumPy and optional GPU support via CuPy
- Core tensor ops: `matmul`, `concat`, arithmetic, reductions
- Activations: `sigmoid`, `tanh`, `relu`, `softmax`
- Losses: `mse`, `cross_entropy`, `binary_cross_entropy`
- Modules and parameter collection with a tiny `Module` base class

## Project Structure

- `node.py`: `Node` class, ops, losses, autodiff engine, device helpers
- `module.py`: `Module`, `Linear`, `RNN`, and activation modules
- `optim.py`: `Optimizer` and `SGD`
- `data.py`: `DataLoader`
- `autodiff.ipynb`: notebook experimentation

## Requirements

- Python 3.12+ (uses the `type` alias statement syntax)
- NumPy
- Optional: CuPy for GPU mode

Install dependencies:

```bash
py -m pip install numpy matplotlib
```

Optional GPU dependency (choose the CuPy build that matches your CUDA version):

```bash
py -m pip install cupy-cuda12x
```

## Core Concepts

### 1. Build values with `Node`

`Node` wraps array/scalar values and tracks graph history for gradients.

```python
from node import Node
import numpy as np

x = Node("x", np.array([[2.0]]))
y = Node("y", np.array([[3.0]]))
z = x * y + x

z.backward()
print("dz/dx:", x.grad)  # y + 1
print("dz/dy:", y.grad)  # x
```

### 2. Train with optimizer

```python
from node import Node
from optim import SGD
import numpy as np

w = Node("w", np.random.randn())
b = Node("b", np.random.randn())

X = np.arange(1, 8)
y_true = np.array([9, 8, 10, 12, 11, 13, 14])

opt = SGD([w, b], lr=0.01)
for _ in range(200):
    y_pred = w * X + b
    loss = Node.mse(y_pred, y_true)

    opt.zero_grad()
    loss.backward()
    opt.step()

print("w:", w.value, "b:", b.value)
```

### 3. Use modules (`RNN`)

```python
import numpy as np
from module import RNN
from node import Node
from optim import SGD

np.random.seed(42)

model = RNN(input_size=5, hidden_size=8, output_size=1)
opt = SGD(model.parameters(), lr=0.01)

dataset = [
    (
        [
            [0.2, 1.1, 0.3, 0.7, 0.5],
            [0.9, 0.1, 0.4, 0.2, 0.8],
        ],
        [[1.0]],
    ),
    (
        [
            [0.7, 0.3, 0.8, 0.2, 0.1],
            [0.4, 0.9, 0.5, 0.6, 0.3],
        ],
        [[0.0]],
    ),
]

for epoch in range(20):
    total = 0.0
    for X, y in dataset:
        prob = model(X)
        loss = Node.binary_cross_entropy(prob, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += float(loss.value)

    if epoch % 5 == 0:
        print(f"epoch {epoch}: {total / len(dataset):.4f}")
```

## API Reference (Minimal)

### `Node` (in `node.py`)

Factory methods:

- `Node.zeros(label, shape, device="cpu")`
- `Node.ones(label, shape, device="cpu")`
- `Node.randn(label, shape, scale=1.0, device="cpu")`

Ops/activations:

- `Node.matmul(a, b)`
- `Node.concat(a, b, axis=1)`
- `Node.tanh(x)`, `Node.relu(x)`, `Node.sigmoid(x)`, `Node.softmax(x)`

Losses:

- `Node.mse(values, target)`
- `Node.cross_entropy(probs, target_indices)`
- `Node.binary_cross_entropy(prob, target)`

Autodiff:

- `node.backward()` computes gradients for all ancestors in the graph
- `node.zero_grad()` clears a node's gradient tensor

Device control:

- `node.to_cpu()`
- `node.to_gpu()` (requires CuPy)

### `Module` (in `module.py`)

- Subclass and implement `forward(...)`
- Call module directly: `out = module(x)`
- Learnable parameters are tracked automatically when assigned as `Node`
- Use `module.parameters()` to pass params into an optimizer

Implemented modules:

- `Linear(in_dim, out_dim)`
- `RNN(input_size, hidden_size, output_size)`
- `Sigmoid`, `Tanh`, `ReLU`, `Softmax`

### `SGD` (in `optim.py`)

- `SGD(params, lr=0.01)`
- `zero_grad()` then `step()` in each training iteration

### `DataLoader` (in `data.py`)

- `DataLoader(X, y, batch_size=32, shuffle=True, device="cpu")`
- Iterating yields `(X_batch, y_batch)`
- Supports `to_cpu()` and `to_gpu()`

## Shape Conventions

The code is mostly row-major for batched features.

- Linear input: `(batch, in_dim)`
- Linear weights: `(in_dim, out_dim)`
- Linear output: `(batch, out_dim)`
- RNN step input expected as one row vector per timestep (internally reshaped to `(1, input_size)`)

For binary classification with one output unit:

- model output shape is typically `(1, 1)`
- target can be provided as a scalar, list, or array that broadcasts correctly (for example `1`, `[[1.0]]`)

## Common Training Loop Pattern

```python
pred = model(X)
loss = some_loss(pred, target)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Troubleshooting

- `ValueError` from `matmul`:
  - Check your feature dimensions and whether inputs are row vectors.
- Device mismatch (`mismatch in devices`):
  - Ensure all nodes involved in one operation are on the same device.
- `RuntimeError("cuda environment missing")`:
  - Install CuPy or run on CPU.

## Limitations / Notes

- This project does not include advanced features like momentum, Adam, mixed precision, or model serialization.
- Numerical stability is basic in some loss/activation routines.
- Broadcasting is supported in many operations, but shape discipline is still important.

###### readme generated using github copilot