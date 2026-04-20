# 📊 Input Specific Neural Networks (ISNN) — Implementation Report

## 🧠 Overview

This assignment implements **Input Specific Neural Networks (ISNNs)** based on the research paper:
*“Input Specific Neural Networks” (2025)*.

The goal was to:

* Understand the ISNN architecture
* Implement **ISNN-1 and ISNN-2**
* Train models on toy datasets
* Compare performance using loss curves and behavioral plots
* Implement **manual backpropagation using NumPy**

---

## ⚙️ Architectures Implemented

### 🔹 ISNN-1

* Multi-branch architecture:

  * `x` → convex branch
  * `y` → convex + monotonic
  * `t` → monotonic
  * `z` → unconstrained
* Branch outputs are combined once and passed through final layers

### 🔹 ISNN-2

* Extension of ISNN-1
* Introduces **skip connections**
* Inputs are reused in deeper layers
* Provides improved learning capacity and generalization

---

## 🧪 Dataset

We generated a synthetic dataset using the function:

[
f(x, y, t, z) =
e^{-0.5x}

* \log(1 + e^{0.4y})
* \tanh(t)
* \sin(z)

- 0.4
  ]

### 📌 Data Details:

* **Training set**: 500 samples, range [0, 4]
* **Test set**: 5000 samples, range [0, 6]
* Inputs: `[x, y, t, z]`
* Output: scalar value

---

## 🏋️ Training Setup

* Framework: **PyTorch**
* Optimizer: Adam
* Loss Function: Mean Squared Error (MSE)
* Epochs: 1000
* Hidden dimensions:

  * ISNN-1: 10 neurons/layer
  * ISNN-2: 15 neurons/layer

---

## 📈 Results

### 🔹 Loss Curves

* Both ISNN-1 and ISNN-2 successfully minimized training and test loss
* ISNN-2 generally achieved:

  * Faster convergence
  * Lower test loss

### 🔹 Behavioral Analysis

We evaluated model behavior by setting:

```
x = y = t = z ∈ [0, 6]
```

#### Observations:

* Models performed well within training range `[0, 4]`
* ISNN architectures showed strong **extrapolation ability**
* Predictions closely followed the true function trend outside training data

---

## 💀 NumPy Implementation (Manual Backpropagation)

A full neural network was implemented using **NumPy only**, including:

### ✔ Forward Pass

* Matrix multiplication
* Custom activation functions

### ✔ Backpropagation

* Manual gradient computation using chain rule
* No automatic differentiation used

### ✔ Training Loop

* Weight updates using gradient descent

---

## 📊 NumPy Results

* Model successfully trained using manual gradients
* Loss decreased steadily over epochs
* Behavioral plots showed reasonable approximation of the true function

---

## 📁 Project Structure

```
├── generate_data.py        # Dataset generation
├── isnn1_pytorch.py       # ISNN-1 (PyTorch)
├── isnn2_pytorch.py       # ISNN-2 (PyTorch)
├── isnn_numpy.py          # Manual implementation (NumPy)
├── train.csv / test.csv   # Dataset (optional CSV format)
├── plots/                 # Generated graphs
└── README.md              # Report
```

---

## 📦 Outputs

The following were generated:

* ✔ Training loss graphs
* ✔ Test loss graphs
* ✔ Behavioral plots (model vs true function)
* ✔ Dataset files (.npy / .csv)

---

## 🧠 Key Learnings

* ISNNs allow **structured constraints** on neural networks
* Architectural constraints improve **generalization and extrapolation**
* Manual backpropagation provides deeper understanding of:

  * Gradient flow
  * Weight updates
* Skip connections (ISNN-2) improve performance significantly

---

## 🚀 Conclusion

This project demonstrates:

* Successful implementation of research-level neural network architectures
* Effective comparison between constrained and unconstrained learning
* Practical understanding of both **automatic and manual differentiation**

ISNNs are powerful for problems where **input-output relationships must follow known constraints**.

---

## 👨‍💻 Author

**Emaan Ali**

---

## ⭐ Notes

* All implementations were built from scratch for learning purposes
* NumPy version avoids any deep learning frameworks
* Results align with trends reported in the original paper

---

