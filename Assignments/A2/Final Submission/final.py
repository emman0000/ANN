"""
=============================================================================
INPUT SPECIFIC NEURAL NETWORKS (ISNN) - Complete Implementation
=============================================================================
Implements:
  - Data generation for Toy Problem 1 (additive) and Toy Problem 2 (multiplicative)
  - ISNN-1 and ISNN-2 in PyTorch (with constraint enforcement)
  - ISNN-1 and ISNN-2 in NumPy (manual forward pass + manual backpropagation)
  - FFNN baseline in both PyTorch and NumPy
  - Training loops (10 random initialisations)
  - All plots: loss curves (Fig 3, Fig 5 style) and behavioural response (Fig 4, Fig 6 style)
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time, warnings
warnings.filterwarnings('ignore')

#  PyTorch ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    print("PyTorch available:", torch.__version__)
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch NOT available – only NumPy path will run")

os.makedirs("plots", exist_ok=True)
os.makedirs("datasets", exist_ok=True)

# =============================================================================
# 1.  LATIN HYPERCUBE SAMPLING
# =============================================================================

def latin_hypercube_sampling(n_samples, n_dims, low=0.0, high=4.0, seed=None):
    """Simple LHS: stratify each dimension, then randomly permute."""
    rng = np.random.default_rng(seed)
    result = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        u    = rng.uniform(size=n_samples)
        result[:, d] = (perm + u) / n_samples
    return low + result * (high - low)

# =============================================================================
# 2.  DATASET GENERATION
# =============================================================================

def generate_dataset1(n_train=500, n_test=5000, train_range=(0.0, 4.0),
                      test_range=(0.0, 6.0), seed=42):
    """
    Additive split function (Eq. 12):
        f = exp(-0.5x) + log(1+exp(0.4y)) + tanh(t) + sin(z) - 0.4
    Convex in x; convex + monotone increasing in y; monotone in t; arbitrary in z.
    """
    X_train = latin_hypercube_sampling(n_train, 4, *train_range, seed=seed)
    X_test  = latin_hypercube_sampling(n_test,  4, *test_range,  seed=seed+1)

    def f(X):
        x, y, t, z = X[:,0], X[:,1], X[:,2], X[:,3]
        return (np.exp(-0.5*x)
                + np.log(1 + np.exp(0.4*y))
                + np.tanh(t)
                + np.sin(z)
                - 0.4)

    y_train = f(X_train).reshape(-1, 1)
    y_test  = f(X_test).reshape(-1, 1)

    np.save("datasets/dataset1_X_train.npy", X_train)
    np.save("datasets/dataset1_y_train.npy", y_train)
    np.save("datasets/dataset1_X_test.npy",  X_test)
    np.save("datasets/dataset1_y_test.npy",  y_test)

    print(f"Dataset 1 | train: {X_train.shape}, test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def generate_dataset2(n_train=500, n_test=5000, train_range=(0.0, 4.0),
                      test_range=(0.0, 10.0), seed=42):
    """
    Multiplicative split function (Eq. 13/14):
        g = fx * fy * fz * ft
    Convex in x; convex + monotone in y; monotone in t; arbitrary in z.
    """
    X_train = latin_hypercube_sampling(n_train, 4, *train_range, seed=seed)
    X_test  = latin_hypercube_sampling(n_test,  4, *test_range,  seed=seed+1)

    def g(X):
        x, y, t, z = X[:,0], X[:,1], X[:,2], X[:,3]
        fx = np.exp(-0.3*x)
        fy = (0.15*y)**2
        ft = np.tanh(0.3*t)
        fz = 0.2*np.sin(0.5*z + 2) + 0.5
        return fx * fy * ft * fz

    y_train = g(X_train).reshape(-1, 1)
    y_test  = g(X_test).reshape(-1, 1)

    np.save("datasets/dataset2_X_train.npy", X_train)
    np.save("datasets/dataset2_y_train.npy", y_train)
    np.save("datasets/dataset2_X_test.npy",  X_test)
    np.save("datasets/dataset2_y_test.npy",  y_test)

    print(f"Dataset 2 | train: {X_train.shape}, test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


# =============================================================================
# 3.  ACTIVATION FUNCTIONS  (NumPy)
# =============================================================================

def softplus(x):       return np.log1p(np.exp(np.clip(x, -500, 500)))
def softplus_d(x):     return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))   # sigmoid
def softplus_dd(x):
    s = softplus_d(x)
    return s * (1.0 - s)

def sigmoid(x):        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_d(x):
    s = sigmoid(x)
    return s * (1.0 - s)

# σ_mc = softplus  (convex, monotone non-decreasing)
# σ_m  = σ_a = sigmoid (monotone non-decreasing)


# =============================================================================
# 4.  NumPy MANUAL NEURAL NETWORK BASE CLASS
# =============================================================================

class NPLayer:
    """Single dense layer (NumPy)."""
    def __init__(self, in_dim, out_dim, non_negative=False, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, scale, (out_dim, in_dim))
        self.b = np.zeros(out_dim)
        self.non_negative = non_negative
        if non_negative:
            self.W = np.abs(self.W)
        # cache for backprop
        self.x_in = None

    def forward(self, x):
        self.x_in = x.copy()
        W = np.abs(self.W) if self.non_negative else self.W
        return x @ W.T + self.b   # (N, out_dim)

    def backward(self, dout):
        """
        dout : gradient w.r.t. layer output  (N, out_dim)
        returns gradient w.r.t. layer input   (N, in_dim)
        Also stores dW, db for parameter update.
        """
        W = np.abs(self.W) if self.non_negative else self.W
        self.dW = dout.T @ self.x_in          # (out_dim, in_dim)
        self.db = dout.sum(axis=0)             # (out_dim,)
        dx = dout @ W                          # (N, in_dim)
        if self.non_negative:
            # chain rule through abs: sign of W
            sign_W = np.sign(self.W)
            sign_W[sign_W == 0] = 1
            self.dW = self.dW * sign_W
        return dx

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db
        if self.non_negative:
            self.W = np.abs(self.W)   # project back


# =============================================================================
# 5.  NumPy FFNN
# =============================================================================

class NP_FFNN:
    """
    Simple feed-forward network (NumPy).
    Architecture: 4 → 30 → 30 → 1   (sigmoid activations, unconstrained)
    """
    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.l1 = NPLayer(4,  30, rng=rng)
        self.l2 = NPLayer(30, 30, rng=rng)
        self.l3 = NPLayer(30,  1, rng=rng)
        self.cache = {}

    def forward(self, X):
        z1 = self.l1.forward(X)
        a1 = sigmoid(z1);  self.cache['a1'] = a1; self.cache['z1'] = z1
        z2 = self.l2.forward(a1)
        a2 = sigmoid(z2);  self.cache['a2'] = a2; self.cache['z2'] = z2
        z3 = self.l3.forward(a2)
        return z3  # linear output

    def backward(self, X, y_pred, y_true):
        N   = X.shape[0]
        dL  = 2.0 * (y_pred - y_true) / N          # MSE gradient
        dz3 = dL
        da2 = self.l3.backward(dz3)
        dz2 = da2 * sigmoid_d(self.cache['z2'])
        da1 = self.l2.backward(dz2)
        dz1 = da1 * sigmoid_d(self.cache['z1'])
        self.l1.backward(dz1)

    def update(self, lr):
        for l in [self.l1, self.l2, self.l3]:
            l.update(lr)

    def predict(self, X):
        return self.forward(X)

    def mse(self, X, y):
        return float(np.mean((self.predict(X) - y)**2))


# =============================================================================
# 6.  NumPy ISNN-1
# =============================================================================

class NP_ISNN1:
    """
    ISNN-1 (NumPy manual backprop).
    Input branches:
      x0 → convex (σ_mc, non-negative W from layer 1+)
      y0 → convex + monotone (σ_mc, non-negative W)
      t0 → monotone (σ_m, non-negative W)
      z0 → arbitrary (σ_a, unconstrained)
    Hidden layers: 2 per branch, 10 neurons each.
    """
    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        H = 10
        # y-branch (2 hidden layers)
        self.Wyy0 = NPLayer(1, H, non_negative=True,  rng=rng)
        self.Wyy1 = NPLayer(H, H, non_negative=True,  rng=rng)
        # z-branch
        self.Wzz0 = NPLayer(1, H, non_negative=False, rng=rng)
        self.Wzz1 = NPLayer(H, H, non_negative=False, rng=rng)
        # t-branch
        self.Wtt0 = NPLayer(1, H, non_negative=True,  rng=rng)
        self.Wtt1 = NPLayer(H, H, non_negative=True,  rng=rng)
        # x-branch (first layer takes x0 + branch outputs; subsequent layers non-negative)
        self.Wxx0 = NPLayer(1, H, non_negative=False, rng=rng)  # W0 can be negative
        self.Wxy  = NPLayer(H, H, non_negative=True,  rng=rng)
        self.Wxz  = NPLayer(H, H, non_negative=False, rng=rng)
        self.Wxt  = NPLayer(H, H, non_negative=True,  rng=rng)
        self.Wxx1 = NPLayer(H, H, non_negative=True,  rng=rng)
        self.Wout = NPLayer(H,  1, non_negative=True,  rng=rng)
        self.cache = {}

    def forward(self, X):
        x0 = X[:, 0:1]; y0 = X[:, 1:2]
        t0 = X[:, 2:3]; z0 = X[:, 3:4]

        # y-branch
        gy0 = self.Wyy0.forward(y0)
        y1  = softplus(gy0);   self.cache['gy0']=gy0; self.cache['y1']=y1
        gy1 = self.Wyy1.forward(y1)
        y2  = softplus(gy1);   self.cache['gy1']=gy1; self.cache['y2']=y2

        # z-branch
        jz0 = self.Wzz0.forward(z0)
        z1  = sigmoid(jz0);    self.cache['jz0']=jz0; self.cache['z1']=z1
        jz1 = self.Wzz1.forward(z1)
        z2  = sigmoid(jz1);    self.cache['jz1']=jz1; self.cache['z2']=z2

        # t-branch
        ht0 = self.Wtt0.forward(t0)
        t1  = sigmoid(ht0);    self.cache['ht0']=ht0; self.cache['t1']=t1
        ht1 = self.Wtt1.forward(t1)
        t2  = sigmoid(ht1);    self.cache['ht1']=ht1; self.cache['t2']=t2

        # x-branch layer 1 (combines all)
        f0 = (self.Wxx0.forward(x0)
              + self.Wxy.forward(y2)
              + self.Wxz.forward(z2)
              + self.Wxt.forward(t2))
        x1 = softplus(f0);     self.cache['f0']=f0; self.cache['x1']=x1

        # x-branch layer 2 (non-negative)
        f1 = self.Wxx1.forward(x1)
        x2 = softplus(f1);     self.cache['f1']=f1; self.cache['x2']=x2

        # output
        fout = self.Wout.forward(x2)
        out  = softplus(fout); self.cache['fout']=fout
        return out

    def backward(self, X, y_pred, y_true):
        N   = X.shape[0]
        x0 = X[:, 0:1]; y0 = X[:, 1:2]

        dL   = 2.0 * (y_pred - y_true) / N

        # output layer
        d_fout = dL * softplus_d(self.cache['fout'])
        d_x2   = self.Wout.backward(d_fout)

        # x-branch layer 2
        d_f1 = d_x2 * softplus_d(self.cache['f1'])
        d_x1 = self.Wxx1.backward(d_f1)

        # x-branch layer 1 – gradients flow to sub-branches
        d_f0 = d_x1 * softplus_d(self.cache['f0'])
        _ = self.Wxx0.backward(d_f0)   # x0 path
        d_y2 = self.Wxy.backward(d_f0)
        d_z2 = self.Wxz.backward(d_f0)
        d_t2 = self.Wxt.backward(d_f0)

        # y-branch (backwards)
        d_gy1 = d_y2 * softplus_d(self.cache['gy1'])
        d_y1  = self.Wyy1.backward(d_gy1)
        d_gy0 = d_y1 * softplus_d(self.cache['gy0'])
        self.Wyy0.backward(d_gy0)

        # z-branch
        d_jz1 = d_z2 * sigmoid_d(self.cache['jz1'])
        d_z1  = self.Wzz1.backward(d_jz1)
        d_jz0 = d_z1 * sigmoid_d(self.cache['jz0'])
        self.Wzz0.backward(d_jz0)

        # t-branch
        d_ht1 = d_t2 * sigmoid_d(self.cache['ht1'])
        d_t1  = self.Wtt1.backward(d_ht1)
        d_ht0 = d_t1 * sigmoid_d(self.cache['ht0'])
        self.Wtt0.backward(d_ht0)

    def update(self, lr):
        for layer in [self.Wyy0, self.Wyy1, self.Wzz0, self.Wzz1,
                      self.Wtt0, self.Wtt1, self.Wxx0, self.Wxy,
                      self.Wxz, self.Wxt, self.Wxx1, self.Wout]:
            layer.update(lr)

    def predict(self, X): return self.forward(X)
    def mse(self, X, y):  return float(np.mean((self.predict(X) - y)**2))


# =============================================================================
# 7.  NumPy ISNN-2
# =============================================================================

class NP_ISNN2:
    """
    ISNN-2 (NumPy manual backprop).
    All branches share depth H.  Skip connections from x0 to every x-layer.
    """
    def __init__(self, H=15, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.H = H
        # shared branches (1 hidden layer each before merging)
        self.Wyy = NPLayer(1, H, non_negative=True,  rng=rng)
        self.Wzz = NPLayer(1, H, non_negative=False, rng=rng)
        self.Wtt = NPLayer(1, H, non_negative=True,  rng=rng)
        # x-branch layer 0 (all inputs, unconstrained first W)
        self.Wxx0  = NPLayer(1, H, non_negative=False, rng=rng)
        self.Wxy0  = NPLayer(1, H, non_negative=True,  rng=rng)
        self.Wxz0  = NPLayer(1, H, non_negative=False, rng=rng)
        self.Wxt0  = NPLayer(1, H, non_negative=True,  rng=rng)
        self.b0    = np.zeros(H)
        # x-branch layer 1 (non-negative W[xx], skip from x0 via W[xx0] free)
        self.Wxx1  = NPLayer(H, H, non_negative=True,  rng=rng)
        self.Wxx01 = NPLayer(1, H, non_negative=False, rng=rng)  # skip
        self.Wxy1  = NPLayer(H, H, non_negative=True,  rng=rng)
        self.Wxz1  = NPLayer(H, H, non_negative=False, rng=rng)
        self.Wxt1  = NPLayer(H, H, non_negative=True,  rng=rng)
        self.b1    = np.zeros(H)
        # output
        self.Wout  = NPLayer(H,  1, non_negative=True,  rng=rng)
        self.cache = {}

    def forward(self, X):
        x0 = X[:, 0:1]; y0 = X[:, 1:2]
        t0 = X[:, 2:3]; z0 = X[:, 3:4]

        # parallel branches (one layer each)
        gy = self.Wyy.forward(y0)
        y1 = softplus(gy);  self.cache['gy']=gy;  self.cache['y1']=y1

        jz = self.Wzz.forward(z0)
        z1 = sigmoid(jz);   self.cache['jz']=jz;  self.cache['z1']=z1

        ht = self.Wtt.forward(t0)
        t1 = sigmoid(ht);   self.cache['ht']=ht;  self.cache['t1']=t1

        # x-branch layer 0
        f0 = (self.Wxx0.forward(x0)
              + self.Wxy0.forward(y0)
              + self.Wxz0.forward(z0)
              + self.Wxt0.forward(t0)
              + self.b0)
        x1 = softplus(f0);  self.cache['f0']=f0; self.cache['x1']=x1

        # x-branch layer 1 (skip connection from x0)
        f1 = (self.Wxx1.forward(x1)
              + self.Wxx01.forward(x0)
              + self.Wxy1.forward(y1)
              + self.Wxz1.forward(z1)
              + self.Wxt1.forward(t1)
              + self.b1)
        x2 = softplus(f1);  self.cache['f1']=f1; self.cache['x2']=x2

        fout = self.Wout.forward(x2)
        out  = softplus(fout); self.cache['fout']=fout
        return out

    def backward(self, X, y_pred, y_true):
        N   = X.shape[0]
        x0 = X[:, 0:1]; y0 = X[:, 1:2]

        dL    = 2.0*(y_pred - y_true)/N
        d_fout = dL * softplus_d(self.cache['fout'])
        d_x2   = self.Wout.backward(d_fout)

        # layer 1
        d_f1   = d_x2 * softplus_d(self.cache['f1'])
        d_x1   = self.Wxx1.backward(d_f1)
        _      = self.Wxx01.backward(d_f1)   # skip
        d_y1   = self.Wxy1.backward(d_f1)
        d_z1   = self.Wxz1.backward(d_f1)
        d_t1   = self.Wxt1.backward(d_f1)
        # biases
        self.db1 = d_f1.sum(axis=0)

        # layer 0
        d_f0  = d_x1 * softplus_d(self.cache['f0'])
        _ = self.Wxx0.backward(d_f0)
        _ = self.Wxy0.backward(d_f0)
        _ = self.Wxz0.backward(d_f0)
        _ = self.Wxt0.backward(d_f0)
        self.db0 = d_f0.sum(axis=0)

        # parallel branches
        d_gy = d_y1 * softplus_d(self.cache['gy'])
        self.Wyy.backward(d_gy)

        d_jz = d_z1 * sigmoid_d(self.cache['jz'])
        self.Wzz.backward(d_jz)

        d_ht = d_t1 * sigmoid_d(self.cache['ht'])
        self.Wtt.backward(d_ht)

    def update(self, lr):
        for layer in [self.Wyy, self.Wzz, self.Wtt,
                      self.Wxx0, self.Wxy0, self.Wxz0, self.Wxt0,
                      self.Wxx1, self.Wxx01, self.Wxy1, self.Wxz1, self.Wxt1,
                      self.Wout]:
            layer.update(lr)
        # update shared biases
        lr_b = lr
        self.b0 -= lr_b * getattr(self, 'db0', 0)
        self.b1 -= lr_b * getattr(self, 'db1', 0)

    def predict(self, X): return self.forward(X)
    def mse(self, X, y):  return float(np.mean((self.predict(X) - y)**2))


# =============================================================================
# 8.  NumPy TRAINING LOOP
# =============================================================================

def train_numpy(model_class, X_train, y_train, X_test, y_test,
                n_runs=10, n_epochs=30000, lr=1e-3, batch_size=None,
                model_kwargs=None):
    """
    Train a NumPy model n_runs times.
    Returns arrays of shape (n_runs, n_epochs) for train/test loss.
    """
    if model_kwargs is None:
        model_kwargs = {}
    all_train = np.zeros((n_runs, n_epochs))
    all_test  = np.zeros((n_runs, n_epochs))
    N = X_train.shape[0]

    for run in range(n_runs):
        rng   = np.random.default_rng(run * 1234)
        model = model_class(rng=rng, **model_kwargs)
        tl = vl = 999.0
        for ep in range(n_epochs):
            y_pred = model.forward(X_train)
            model.backward(X_train, y_pred, y_train)
            model.update(lr)
            if ep % 50 == 0 or ep == n_epochs - 1:
                tl = float(np.mean((model.predict(X_train) - y_train)**2))
                vl = float(np.mean((model.predict(X_test)  - y_test )**2))
            all_train[run, ep] = tl
            all_test[run,  ep] = vl
        print(f"  Run {run+1}/{n_runs}  train={all_train[run,-1]:.4e}  test={all_test[run,-1]:.4e}")
    return all_train, all_test


# =============================================================================
# 9.  PyTorch MODELS  (if available)
# =============================================================================

if TORCH_AVAILABLE:

    class ConstrainedLinear(nn.Module):
        """Linear layer with optional non-negativity constraint on weights."""
        def __init__(self, in_f, out_f, non_negative=False):
            super().__init__()
            self.non_negative = non_negative
            self.linear = nn.Linear(in_f, out_f)
            nn.init.kaiming_normal_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
            if non_negative:
                with torch.no_grad():
                    self.linear.weight.data = self.linear.weight.data.abs()

        def forward(self, x):
            if self.non_negative:
                weight = self.linear.weight.abs()
                return nn.functional.linear(x, weight, self.linear.bias)
            return self.linear(x)

    # σ_mc = softplus (convex, monotone non-decreasing)
    # σ_m = σ_a = sigmoid

    class PT_FFNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 30), nn.Sigmoid(),
                nn.Linear(30, 30), nn.Sigmoid(),
                nn.Linear(30, 1)
            )
        def forward(self, x): return self.net(x)

    class PT_ISNN1(nn.Module):
        """PyTorch ISNN-1 with constraint enforcement."""
        def __init__(self, H=10):
            super().__init__()
            # y-branch
            self.Wyy0 = ConstrainedLinear(1, H, non_negative=True)
            self.Wyy1 = ConstrainedLinear(H, H, non_negative=True)
            # z-branch (arbitrary)
            self.Wzz0 = ConstrainedLinear(1, H, non_negative=False)
            self.Wzz1 = ConstrainedLinear(H, H, non_negative=False)
            # t-branch (monotone)
            self.Wtt0 = ConstrainedLinear(1, H, non_negative=True)
            self.Wtt1 = ConstrainedLinear(H, H, non_negative=True)
            # x-branch (first W free; subsequent non-negative)
            self.Wxx0 = ConstrainedLinear(1, H, non_negative=False)
            self.Wxy  = ConstrainedLinear(H, H, non_negative=True)
            self.Wxz  = ConstrainedLinear(H, H, non_negative=False)
            self.Wxt  = ConstrainedLinear(H, H, non_negative=True)
            self.Wxx1 = ConstrainedLinear(H, H, non_negative=True)
            self.Wout = ConstrainedLinear(H,  1, non_negative=True)

        def forward(self, X):
            x0, y0, t0, z0 = X[:,0:1], X[:,1:2], X[:,2:3], X[:,3:4]
            sp = torch.nn.functional.softplus
            sg = torch.sigmoid

            y2 = sp(self.Wyy1(sp(self.Wyy0(y0))))
            z2 = sg(self.Wzz1(sg(self.Wzz0(z0))))
            t2 = sg(self.Wtt1(sg(self.Wtt0(t0))))

            f0 = self.Wxx0(x0) + self.Wxy(y2) + self.Wxz(z2) + self.Wxt(t2)
            x1 = sp(f0)
            x2 = sp(self.Wxx1(x1))
            return sp(self.Wout(x2))

    class PT_ISNN2(nn.Module):
        """PyTorch ISNN-2 with skip connections."""
        def __init__(self, H=15):
            super().__init__()
            self.Wyy  = ConstrainedLinear(1, H, non_negative=True)
            self.Wzz  = ConstrainedLinear(1, H, non_negative=False)
            self.Wtt  = ConstrainedLinear(1, H, non_negative=True)
            self.Wxx0 = ConstrainedLinear(1, H, non_negative=False)
            self.Wxy0 = ConstrainedLinear(1, H, non_negative=True)
            self.Wxz0 = ConstrainedLinear(1, H, non_negative=False)
            self.Wxt0 = ConstrainedLinear(1, H, non_negative=True)
            self.Wxx1  = ConstrainedLinear(H, H, non_negative=True)
            self.Wxx01 = ConstrainedLinear(1, H, non_negative=False)  # skip
            self.Wxy1  = ConstrainedLinear(H, H, non_negative=True)
            self.Wxz1  = ConstrainedLinear(H, H, non_negative=False)
            self.Wxt1  = ConstrainedLinear(H, H, non_negative=True)
            self.Wout  = ConstrainedLinear(H,  1, non_negative=True)

        def forward(self, X):
            x0, y0, t0, z0 = X[:,0:1], X[:,1:2], X[:,2:3], X[:,3:4]
            sp = torch.nn.functional.softplus
            sg = torch.sigmoid

            y1 = sp(self.Wyy(y0))
            z1 = sg(self.Wzz(z0))
            t1 = sg(self.Wtt(t0))

            f0 = self.Wxx0(x0) + self.Wxy0(y0) + self.Wxz0(z0) + self.Wxt0(t0)
            x1 = sp(f0)
            f1 = self.Wxx1(x1) + self.Wxx01(x0) + self.Wxy1(y1) + self.Wxz1(z1) + self.Wxt1(t1)
            x2 = sp(f1)
            return sp(self.Wout(x2))

    def train_pytorch(ModelClass, X_train_np, y_train_np, X_test_np, y_test_np,
                      n_runs=10, n_epochs=30000, lr=1e-3, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        X_tr = torch.FloatTensor(X_train_np)
        y_tr = torch.FloatTensor(y_train_np)
        X_te = torch.FloatTensor(X_test_np)
        y_te = torch.FloatTensor(y_test_np)

        all_train = np.zeros((n_runs, n_epochs))
        all_test  = np.zeros((n_runs, n_epochs))

        for run in range(n_runs):
            torch.manual_seed(run * 1234)
            model     = ModelClass(**model_kwargs)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            loss_fn   = nn.MSELoss()

            for ep in range(n_epochs):
                model.train()
                optimizer.zero_grad()
                pred = model(X_tr)
                loss = loss_fn(pred, y_tr)
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    tl = loss_fn(model(X_tr), y_tr).item()
                    vl = loss_fn(model(X_te), y_te).item()
                all_train[run, ep] = tl
                all_test[run,  ep] = vl

            print(f"  Run {run+1}/{n_runs}  train={all_train[run,-1]:.4e}  test={all_test[run,-1]:.4e}")
        return all_train, all_test


# =============================================================================
# 10.  PLOTTING HELPERS
# =============================================================================

COLORS = {'FFNN': '#d62728', 'ISNN-1': '#1f77b4', 'ISNN-2': '#2ca02c'}

def plot_loss_curves(results_dict, title, filename, n_epochs=30000):
    """Figure 3 / Figure 5 style: training and test loss curves."""
    epochs = np.arange(1, n_epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key in zip(axes, ['train', 'test']):
        for name, (train_arr, test_arr) in results_dict.items():
            arr   = train_arr if key == 'train' else test_arr
            mu    = arr.mean(axis=0)
            sigma = arr.std(axis=0)
            ax.semilogy(epochs, mu,  color=COLORS.get(name, 'grey'), label=name)
            ax.fill_between(epochs, np.clip(mu-sigma, 1e-10, None),
                            mu+sigma, alpha=0.25, color=COLORS.get(name, 'grey'))
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(f"({'a' if key=='train' else 'b'}) {key.capitalize()} Loss")
        ax.legend(); ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"plots/{filename}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plots/{filename}")


def compute_diagonal_response(model_np, true_fn, diag_range):
    """Evaluate model and true function along the diagonal x=y=t=z."""
    vals = np.linspace(diag_range[0], diag_range[1], 200)
    X    = np.stack([vals]*4, axis=1)
    pred = model_np.predict(X).flatten()
    true = true_fn(X).flatten()
    return vals, pred, true


def plot_behavioral_response(models_dict, true_fn, train_range, test_range,
                              title, filename, n_runs=10, n_epochs=30000,
                              dataset_fn=None, lr=1e-3):
    """
    Figure 4 / Figure 6 style: interpolated vs extrapolated predictions.
    For each model we re-train n_runs times and plot mean ± std on the diagonal.
    """
    X_tr, y_tr, X_te, y_te = dataset_fn()
    diag_full  = np.linspace(test_range[0],  test_range[1],  200)
    diag_train = np.linspace(train_range[0], train_range[1], 200)

    X_full  = np.stack([diag_full]*4,  axis=1)
    X_train_diag = np.stack([diag_train]*4, axis=1)
    true_full  = true_fn(X_full).flatten()

    fig, axes = plt.subplots(1, len(models_dict), figsize=(6*len(models_dict), 5))
    if len(models_dict) == 1:
        axes = [axes]

    for ax, (name, ModelClass) in zip(axes, models_dict.items()):
        preds = []
        for run in range(n_runs):
            rng   = np.random.default_rng(run * 999)
            model = ModelClass(rng=rng)
            for ep in range(n_epochs):
                yp = model.forward(X_tr)
                model.backward(X_tr, yp, y_tr)
                model.update(lr)
            pred = model.predict(X_full).flatten()
            preds.append(pred)

        preds = np.array(preds)
        mu    = preds.mean(axis=0)
        sigma = preds.std(axis=0)

        # split into interpolated (within train range) and extrapolated
        split = int(200 * (train_range[1] - test_range[0]) /
                    (test_range[1] - test_range[0]))

        ax.plot(diag_full, true_full, 'k--', label='True response', linewidth=1.5)
        ax.plot(diag_full[:split+1], mu[:split+1], color=COLORS.get(name,'blue'),
                label='Interpolated response')
        ax.fill_between(diag_full[:split+1],
                        mu[:split+1]-sigma[:split+1],
                        mu[:split+1]+sigma[:split+1], alpha=0.3,
                        color=COLORS.get(name,'blue'))
        if split < 200:
            ax.plot(diag_full[split:], mu[split:], color='#ff7f0e',
                    label='Extrapolated response')
            ax.fill_between(diag_full[split:],
                            mu[split:]-sigma[split:],
                            mu[split:]+sigma[split:], alpha=0.3,
                            color='#ff7f0e')
        ax.set_xlabel("x = y = t = z"); ax.set_ylabel("f")
        ax.set_title(f"({chr(97 + list(models_dict.keys()).index(name))}) {name}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"plots/{filename}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plots/{filename}")


# =============================================================================
# 11.  MAIN
# =============================================================================

def main():
    print("\n" + "="*65)
    print("  INPUT SPECIFIC NEURAL NETWORKS – Full Pipeline")
    print("="*65)

    # ── reduce epochs for practicality (paper uses 30 000) ──────────────────
    N_EPOCHS = 2000   # increase to 30000 for full reproduction
    LR       = 1e-3
    N_RUNS   = 3      # paper uses 10; reduce for speed

    # ─────────────────────────────────────────────────────────────────────────
    # DATASET GENERATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1] Generating datasets ...")
    X1_tr, y1_tr, X1_te, y1_te = generate_dataset1()
    X2_tr, y2_tr, X2_te, y2_te = generate_dataset2()

    def true_fn1(X):
        x, y, t, z = X[:,0], X[:,1], X[:,2], X[:,3]
        return (np.exp(-0.5*x)
                + np.log(1+np.exp(0.4*y))
                + np.tanh(t)
                + np.sin(z) - 0.4).reshape(-1, 1)

    def true_fn2(X):
        x, y, t, z = X[:,0], X[:,1], X[:,2], X[:,3]
        fx = np.exp(-0.3*x); fy = (0.15*y)**2
        ft = np.tanh(0.3*t); fz = 0.2*np.sin(0.5*z+2)+0.5
        return (fx*fy*ft*fz).reshape(-1, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # NUMPY TRAINING – DATASET 1
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[2] NumPy training – Dataset 1  ({N_EPOCHS} epochs, {N_RUNS} runs) ...")
    results_np_d1 = {}

    print("  FFNN ...")
    tr, te = train_numpy(NP_FFNN,  X1_tr, y1_tr, X1_te, y1_te,
                         n_runs=N_RUNS, n_epochs=N_EPOCHS, lr=LR)
    results_np_d1['FFNN'] = (tr, te)

    print("  ISNN-1 ...")
    tr, te = train_numpy(NP_ISNN1, X1_tr, y1_tr, X1_te, y1_te,
                         n_runs=N_RUNS, n_epochs=N_EPOCHS, lr=LR)
    results_np_d1['ISNN-1'] = (tr, te)

    print("  ISNN-2 ...")
    tr, te = train_numpy(NP_ISNN2, X1_tr, y1_tr, X1_te, y1_te,
                         n_runs=N_RUNS, n_epochs=N_EPOCHS, lr=LR)
    results_np_d1['ISNN-2'] = (tr, te)

    plot_loss_curves(results_np_d1,
                     "Dataset 1 (Additive) – NumPy Manual Backprop",
                     "fig3_numpy_dataset1_loss.png", n_epochs=N_EPOCHS)

    # ─────────────────────────────────────────────────────────────────────────
    # NUMPY TRAINING – DATASET 2
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[3] NumPy training – Dataset 2  ({N_EPOCHS} epochs, {N_RUNS} runs) ...")
    results_np_d2 = {}

    print("  FFNN ...")
    tr, te = train_numpy(NP_FFNN,  X2_tr, y2_tr, X2_te, y2_te,
                         n_runs=N_RUNS, n_epochs=N_EPOCHS, lr=LR)
    results_np_d2['FFNN'] = (tr, te)

    print("  ISNN-1 ...")
    tr, te = train_numpy(NP_ISNN1, X2_tr, y2_tr, X2_te, y2_te,
                         n_runs=N_RUNS, n_epochs=N_EPOCHS, lr=LR)
    results_np_d2['ISNN-1'] = (tr, te)

    print("  ISNN-2 ...")
    tr, te = train_numpy(NP_ISNN2, X2_tr, y2_tr, X2_te, y2_te,
                         n_runs=N_RUNS, n_epochs=N_EPOCHS, lr=LR)
    results_np_d2['ISNN-2'] = (tr, te)

    plot_loss_curves(results_np_d2,
                     "Dataset 2 (Multiplicative) – NumPy Manual Backprop",
                     "fig5_numpy_dataset2_loss.png", n_epochs=N_EPOCHS)

    # ─────────────────────────────────────────────────────────────────────────
    # BEHAVIORAL RESPONSE PLOTS (NumPy, reduced runs for speed)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4] Behavioral response plots (NumPy) ...")

    # Dataset 1
    BR_EPOCHS = 1000
    models_to_plot = {'FFNN': NP_FFNN, 'ISNN-1': NP_ISNN1, 'ISNN-2': NP_ISNN2}

    print("  Dataset 1 behavioral response ...")
    X_tr1, y_tr1, X_te1, y_te1 = X1_tr, y1_tr, X1_te, y1_te

    diag_full = np.linspace(0, 6, 200)
    X_diag    = np.stack([diag_full]*4, axis=1)
    true_diag = true_fn1(X_diag).flatten()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, Cls) in zip(axes, models_to_plot.items()):
        preds = []
        for run in range(N_RUNS):
            rng = np.random.default_rng(run*77)
            m   = Cls(rng=rng)
            for _ in range(BR_EPOCHS):
                yp = m.forward(X_tr1)
                m.backward(X_tr1, yp, y_tr1)
                m.update(LR)
            preds.append(m.predict(X_diag).flatten())
        preds = np.array(preds)
        mu = preds.mean(0); sig = preds.std(0)
        split = int(200 * 4/6)
        ax.plot(diag_full, true_diag, 'k--', label='True', lw=1.5)
        ax.plot(diag_full[:split], mu[:split], color=COLORS[name], label='Interpolated')
        ax.fill_between(diag_full[:split], mu[:split]-sig[:split], mu[:split]+sig[:split],
                        alpha=0.3, color=COLORS[name])
        ax.plot(diag_full[split:], mu[split:], color='#ff7f0e', label='Extrapolated')
        ax.fill_between(diag_full[split:], mu[split:]-sig[split:], mu[split:]+sig[split:],
                        alpha=0.3, color='#ff7f0e')
        ax.set_title(f"({chr(97+list(models_to_plot.keys()).index(name))}) {name}")
        ax.set_xlabel("x=y=t=z"); ax.set_ylabel("f"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Dataset 1 Behavioral Response (NumPy)", fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/fig4_numpy_dataset1_behavior.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved plots/fig4_numpy_dataset1_behavior.png")

    # Dataset 2
    print("  Dataset 2 behavioral response ...")
    diag_full2 = np.linspace(0, 10, 200)
    X_diag2    = np.stack([diag_full2]*4, axis=1)
    true_diag2 = true_fn2(X_diag2).flatten()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, Cls) in zip(axes, models_to_plot.items()):
        preds = []
        for run in range(N_RUNS):
            rng = np.random.default_rng(run*88)
            m   = Cls(rng=rng)
            for _ in range(BR_EPOCHS):
                yp = m.forward(X2_tr)
                m.backward(X2_tr, yp, y2_tr)
                m.update(LR)
            preds.append(m.predict(X_diag2).flatten())
        preds = np.array(preds)
        mu = preds.mean(0); sig = preds.std(0)
        split = int(200 * 4/10)
        ax.plot(diag_full2, true_diag2, 'k--', label='True', lw=1.5)
        ax.plot(diag_full2[:split], mu[:split], color=COLORS[name], label='Interpolated')
        ax.fill_between(diag_full2[:split], mu[:split]-sig[:split], mu[:split]+sig[:split],
                        alpha=0.3, color=COLORS[name])
        ax.plot(diag_full2[split:], mu[split:], color='#ff7f0e', label='Extrapolated')
        ax.fill_between(diag_full2[split:], mu[split:]-sig[split:], mu[split:]+sig[split:],
                        alpha=0.3, color='#ff7f0e')
        ax.set_title(f"({chr(97+list(models_to_plot.keys()).index(name))}) {name}")
        ax.set_xlabel("x=y=t=z"); ax.set_ylabel("g"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Dataset 2 Behavioral Response (NumPy)", fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/fig6_numpy_dataset2_behavior.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved plots/fig6_numpy_dataset2_behavior.png")

    # ─────────────────────────────────────────────────────────────────────────
    # PyTorch TRAINING
    # ─────────────────────────────────────────────────────────────────────────
    if TORCH_AVAILABLE:
        print(f"\n[5] PyTorch training – Dataset 1  ({N_EPOCHS} epochs, {N_RUNS} runs) ...")
        results_pt_d1 = {}

        print("  FFNN ...")
        tr, te = train_pytorch(PT_FFNN, X1_tr, y1_tr, X1_te, y1_te,
                                n_runs=N_RUNS, n_epochs=N_EPOCHS, lr=LR)
        results_pt_d1['FFNN'] = (tr, te)

        print("  ISNN-1 ...")
        tr, te = train_pytorch(PT_ISNN1, X1_tr, y1_tr, X1_te, y1_te,
                                n_runs=N_RUNS, n_epochs=N_EPOCHS, lr=LR)
        results_pt_d1['ISNN-1'] = (tr, te)

        print("  ISNN-2 ...")
        tr, te = train_pytorch(PT_ISNN2, X1_tr, y1_tr, X1_te, y1_te,
                                n_runs=N_RUNS, n_epochs=N_EPOCHS, lr=LR)
        results_pt_d1['ISNN-2'] = (tr, te)

        plot_loss_curves(results_pt_d1,
                         "Dataset 1 (Additive) – PyTorch",
                         "fig3_pytorch_dataset1_loss.png", n_epochs=N_EPOCHS)

        print(f"\n[6] PyTorch training – Dataset 2  ({N_EPOCHS} epochs, {N_RUNS} runs) ...")
        results_pt_d2 = {}

        print("  FFNN ...")
        tr, te = train_pytorch(PT_FFNN, X2_tr, y2_tr, X2_te, y2_te,
                                n_runs=N_RUNS, n_epochs=N_EPOCHS, lr=LR)
        results_pt_d2['FFNN'] = (tr, te)

        print("  ISNN-1 ...")
        tr, te = train_pytorch(PT_ISNN1, X2_tr, y2_tr, X2_te, y2_te,
                                n_runs=N_RUNS, n_epochs=N_EPOCHS, lr=LR)
        results_pt_d2['ISNN-1'] = (tr, te)

        print("  ISNN-2 ...")
        tr, te = train_pytorch(PT_ISNN2, X2_tr, y2_tr, X2_te, y2_te,
                                n_runs=N_RUNS, n_epochs=N_EPOCHS, lr=LR)
        results_pt_d2['ISNN-2'] = (tr, te)

        plot_loss_curves(results_pt_d2,
                         "Dataset 2 (Multiplicative) – PyTorch",
                         "fig5_pytorch_dataset2_loss.png", n_epochs=N_EPOCHS)

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY TABLE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  FINAL RESULTS SUMMARY  (mean test MSE ± std, last epoch)")
    print("="*65)
    print(f"\n{'Model':<10} {'Dataset':<12} {'Backend':<10} {'Test MSE (mean)':<18} {'Std'}")
    print("-"*65)

    for ds_name, res in [("Dataset1", results_np_d1), ("Dataset2", results_np_d2)]:
        for mname, (tr, te) in res.items():
            mu  = te[:, -1].mean()
            std = te[:, -1].std()
            print(f"{mname:<10} {ds_name:<12} {'NumPy':<10} {mu:.4e}           {std:.2e}")

    if TORCH_AVAILABLE:
        for ds_name, res in [("Dataset1", results_pt_d1), ("Dataset2", results_pt_d2)]:
            for mname, (tr, te) in res.items():
                mu  = te[:, -1].mean()
                std = te[:, -1].std()
                print(f"{mname:<10} {ds_name:<12} {'PyTorch':<10} {mu:.4e}           {std:.2e}")

    print("\n[Done] All plots saved to ./plots/")
    print("[Done] Datasets saved to ./datasets/")


if __name__ == "__main__":
    main()