# NeuroVision Architecture

## Goal

Build an interactive educational platform that explains how neural networks work visually.

## Philosophy

Every animation should represent a real computation inside the neural network.

## Learning Flow

Image
↓
Preprocessing
↓
Tensor
↓
Convolution
↓
Feature Maps
↓
Activation
↓
Flatten
↓
Classifier
↓
Prediction
# 03 - Convolution and How CNNs Learn

## 📅 Date
08 August 2026

---

# Today's Goal

Understand:
- How convolution actually works
- Why CNNs use small kernels
- How filters learn during training
- Introduction to backpropagation and gradients

---

# 1. What is a Kernel (Filter)?

A kernel (or filter) is a small matrix (commonly 3×3) that slides over an image.

Example:

-1   0   1
-1   0   1
-1   0   1

The kernel looks at only a small part of the image at a time.

For every position:
1. Multiply corresponding values.
2. Add them together.
3. Produce one value in the output feature map.

This process is called **Convolution**.

---

# 2. Why use a 3×3 Kernel?

Instead of looking at the entire image, CNNs look at small local regions.

Advantages:

- Only 9 weights to learn (instead of hundreds).
- Can detect local patterns like edges and curves.
- Preserves spatial information by sliding across the image.
- The same filter can be reused everywhere in the image.

Modern CNNs usually prefer stacking multiple 3×3 convolutions instead of using one large kernel.

---

# 3. Filters Learn Different Features

Each filter starts with random values.

During training, filters gradually specialize.

Examples:

Filter 1 → Vertical edges

Filter 2 → Horizontal edges

Filter 3 → Curves

Filter 4 → Corners

...

No one programs these filters manually.

The network learns useful filters automatically.

---

# 4. CNN Learning Hierarchy

CNNs learn simple features first and combine them into more complex features.

Image

↓

Edges / Lines

↓

Curves / Corners

↓

Shapes

↓

Objects / Digits

This is called **Hierarchical Feature Learning**.

---

# 5. Training Cycle

Every neural network repeatedly performs four steps.

Image

↓

Prediction

↓

Calculate Loss

↓

Update Weights

↓

Repeat

The goal of training is to reduce the loss over time.

---

# 6. Loss Function

The loss function measures how wrong the prediction is.

A low loss means the prediction is close to the correct answer.

A high loss means the prediction is far from the correct answer.

Loss tells the network **how wrong it is**, but not **how to fix itself**.

---

# 7. Backpropagation

Backpropagation sends the error backward through the network.

Instead of changing every weight equally, it determines how much each weight contributed to the error.

Each weight receives its own correction.

Weights that contribute more receive larger updates.

Weights that contribute less receive smaller updates.

---

# 8. Gradient (Introduction)

The gradient tells us:

"If this weight changes slightly, how will the loss change?"

It tells the network:

- Increase this weight.
- Decrease this weight.
- Leave this weight almost unchanged.

Gradients provide the direction for improving the model.

# Key Takeaways

- A convolution filter is a sliding window.
- A filter learns useful patterns automatically.
- Multiple filters learn different features.
- CNNs learn from simple features to complex features.
- Loss measures how wrong the prediction is.
- Backpropagation distributes the error through the network.
- Gradients tell each weight how to change.
# Day 3 — CNN Activations & Forward Hooks

## What I learned

- A CNN doesn't just produce a final prediction. Intermediate layers produce activations/feature maps.
- PyTorch forward hooks let us inspect the output of a layer while the model is running.
- I added hooks to the `Conv2d` and `ReLU` layers of my CNN.
- The hooks store the layer outputs in an `activations` dictionary.
- `output.detach().cpu()` lets us store the activation without tracking gradients and move it to the CPU.

## What my CNN produced

For a 28×28 MNIST image:

```text
Conv1 → (1, 32, 26, 26)
ReLU  → (1, 32, 26, 26)

Conv2 → (1, 64, 24, 24)
ReLU  → (1, 64, 24, 24)

Conv3 → (1, 64, 22, 22)
ReLU  → (1, 64, 22, 22)
