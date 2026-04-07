#!/usr/bin/env python3
"""Code-cell dump of codes/discrete_diffusion.ipynb.

This script mirrors the notebook computations and additionally saves the
forward/backward process plots as PDF files next to this file.
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')

FIG_DIR = Path(__file__).resolve().parent

# %% [cell 0]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

n = 30
T = 100
N = 2500
sigma = .75

# %% [cell 1]

# Forward kernel P (Gaussian convolution, row-normalized)
i = np.arange(n)[:, None]
j = np.arange(n)[None, :]
K = np.exp(-((i - j) ** 2) / (2.0 * sigma ** 2))
P = K / K.sum(axis=1, keepdims=True)

# Initial histogram h^0
spikes = (np.array([0.1, 0.6, 0.8]) * n).astype(int)
amps = np.array([1.0, 1.4, 0.6], dtype=np.float64)
amps = amps / amps.sum()

h0 = np.zeros(n, dtype=np.float64)
h0[spikes] = amps

# %% [cell 2]
def sample_categorical_batch(prob):
    cdf = np.cumsum(prob, axis=1)
    u = np.random.rand(prob.shape[0], 1)
    return (u > cdf).sum(axis=1)


def plot_trajectories(Z, title, reverse_time=False, num_show=30, save_path=None):
    TT = Z.shape[1] - 1
    if reverse_time:
        x = np.arange(TT, -1, -1)   # t=T ... 0
    else:
        x = np.arange(0, TT + 1)     # t=0 ... T

    idx = np.linspace(0, Z.shape[0] - 1, min(num_show, Z.shape[0]), dtype=int)
    colors = plt.cm.hsv(np.linspace(0, 1, len(idx), endpoint=False))

    plt.figure(figsize=(10, 4.5))
    for c, k in zip(colors, idx):
        plt.plot(x, Z[k], color=c, alpha=0.4, linewidth=1.2)
    if reverse_time:
        plt.xlim(TT, 0)
    plt.xlabel('time')
    plt.ylabel('state index')
    plt.title(title + f' (showing {len(idx)} trajectories)')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def _fixed_steps_forward(TT, n):
    t = np.array([0, 1, 5, 15, TT], dtype=int)
    return np.unique(np.clip(t, 0, TT))


def _fixed_steps_backward(TT, n):
    t = np.array([TT, 15, 5, 1, 0], dtype=int)
    return np.unique(np.clip(t, 0, TT))[::-1]


def plot_histograms(H, title, reverse_time=False, save_path=None):
    nn, Tp1 = H.shape
    TT = Tp1 - 1

    fig, axes = plt.subplots(5, 1, figsize=(10, 4.5), sharex=True)
    blue = np.array([0.0, 0.0, 1.0])
    red = np.array([1.0, 0.0, 0.0])

    if reverse_time:
        t_steps = _fixed_steps_backward(TT, n)
        s_steps = TT - t_steps
        alphas = np.linspace(0.0, 1.0, len(t_steps))  # blue -> red
        colors = [(1-a)*blue + a*red for a in alphas]
        for ax, s, t, c in zip(axes, s_steps, t_steps, colors):
            ax.bar(np.arange(nn), H[:, s], color=c)
            ax.set_ylabel(f't={t}')
    else:
        t_steps = _fixed_steps_forward(TT, n)
        alphas = np.linspace(1.0, 0.0, len(t_steps))  # red -> blue
        colors = [(1-a)*blue + a*red for a in alphas]
        for ax, t, c in zip(axes, t_steps, colors):
            ax.bar(np.arange(nn), H[:, t], color=c)
            ax.set_ylabel(f't={t}')

    axes[-1].set_xlabel('state index')
    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    plt.show()

# %% [cell 3]

X = np.zeros((N, T + 1), dtype=np.int64)
H = np.zeros((n, T + 1), dtype=np.float64)

X[:, 0] = np.random.choice(n, size=N, p=h0)
H[:, 0] = h0

for t in range(T):
    cur = X[:, t]
    X[:, t + 1] = sample_categorical_batch(P[cur, :])
    H[:, t + 1] = P.T @ H[:, t]

# %% [cell 4]

plot_trajectories(X, 'Forward trajectories', reverse_time=False, num_show=30, save_path=FIG_DIR / 'forward_trajectories.pdf')
plot_histograms(H, 'Forward histograms', reverse_time=False, save_path=FIG_DIR / 'forward_histograms.pdf')

# %% [cell 5]

Q = np.zeros((T, n, n), dtype=np.float64)

for t in range(T):
    D_t = np.diag(H[:, t])
    inv_h_next = np.where(H[:, t + 1] > 0, 1.0 / H[:, t + 1], 0.0)
    D_next_pinv = np.diag(inv_h_next)

    Q_t = D_next_pinv @ P.T @ D_t

    rs = Q_t.sum(axis=1, keepdims=True)
    zero_rows = (rs[:, 0] == 0)
    Q_t[~zero_rows] = Q_t[~zero_rows] / rs[~zero_rows]
    Q_t[zero_rows] = 1.0 / n

    Q[t] = Q_t

# %% [cell 6]

i_mid = n // 2
t_show_disp = np.array([0, 1, 2, n], dtype=int)
t_show = np.clip(t_show_disp, 0, T - 1)

red = np.array([1.0, 0.0, 0.0])
blue = np.array([0.0, 0.0, 1.0])
alphas = np.linspace(0.0, 1.0, len(t_show))
colors = [(1.0 - a) * red + a * blue for a in alphas]

eps_log = 1e-12
x = np.arange(n)
plt.figure(figsize=(10, 4))
for td, t, c in zip(t_show_disp, t_show, colors):
    plt.plot(x, np.log(Q[t, i_mid, :] + eps_log), color=c, linewidth=2.0, label=f't={td}')
plt.xlabel('state index j')
plt.ylabel(r'$\log Q^t_{i,j}$')
plt.title(rf'Central row of reverse kernel (log-scale) at $i={i_mid}$')
plt.legend()
plt.tight_layout()
plt.show()

# %% [cell 7]

Y = np.zeros((N, T + 1), dtype=np.int64)
Y[:, 0] = np.random.choice(n, size=N, p=H[:, T])

for s in range(T):
    t = T - 1 - s
    cur = Y[:, s]
    Y[:, s + 1] = sample_categorical_batch(Q[t, cur, :])

G = np.zeros((n, T + 1), dtype=np.float64)
for s in range(T + 1):
    cnt = np.bincount(Y[:, s], minlength=n)
    G[:, s] = cnt / cnt.sum()

G_exact = np.zeros((n, T + 1), dtype=np.float64)
G_exact[:, 0] = H[:, T]
for s in range(T):
    t = T - 1 - s
    G_exact[:, s + 1] = Q[t].T @ G_exact[:, s]

err_exact = max(np.abs(G_exact[:, s] - H[:, T - s]).sum() for s in range(T + 1))
err_mc_t0 = np.abs(G[:, T] - H[:, 0]).sum()

print(f'deterministic max L1 error (index check): {err_exact:.3e}')
print(f'empirical L1 at recovered h^0: {err_mc_t0:.4f}')

# %% [cell 8]

plot_trajectories(Y, 'Backward trajectories (t=T to t=0)', reverse_time=True, num_show=30, save_path=FIG_DIR / 'backward_trajectories.pdf')
plot_histograms(G, 'Backward empirical histograms (t=T to t=0)', reverse_time=True, save_path=FIG_DIR / 'backward_histograms.pdf')

# %% [cell 9]
class ReverseNet(nn.Module):
    def __init__(self, n, hidden):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, n)

    def forward(self, x_norm, t_norm):
        # Geometric setting: encode state index x as a scalar in [0,1],
        # exactly like normalized time t; network input is 2D: (x_norm, t_norm).
        x = torch.cat([x_norm, t_norm], dim=1)
        x = F.relu(self.fc1(x))
        probs = torch.softmax(self.fc2(x), dim=1)
        return probs

# %% [cell 10]

x_prev = X[:, :-1].reshape(-1)
x_curr = X[:, 1:].reshape(-1)
t_idx = np.tile(np.arange(1, T + 1), N)

x_prev_t = torch.tensor(x_prev, dtype=torch.long)
x_curr_t = torch.tensor((x_curr / (n - 1)).reshape(-1, 1), dtype=torch.float32)
t_norm_t = torch.tensor((t_idx / T).reshape(-1, 1), dtype=torch.float32)

p_hidden = 4 * n
model = ReverseNet(n=n, hidden=p_hidden)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

batch_size = 2048
epochs = 20
M = x_prev_t.shape[0]

for epoch in range(1, epochs + 1):
    perm = torch.randperm(M)
    running = 0.0
    for start in range(0, M, batch_size):
        idx = perm[start:start + batch_size]
        probs = model(x_curr_t[idx], t_norm_t[idx])
        picked = probs[torch.arange(idx.numel()), x_prev_t[idx]].clamp_min(1e-12)
        loss = -torch.log(picked).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item() * idx.numel()

    if epoch % 5 == 0 or epoch == 1:
        print(f'epoch {epoch:02d} | NLL {running / M:.4f}')

# %% [cell 11]

with torch.no_grad():
    t_grid = np.repeat(np.arange(1, T + 1), n)
    i_grid = np.tile(np.arange(n), T)

    probs = model(
        torch.tensor((i_grid / (n - 1)).reshape(-1, 1), dtype=torch.float32),
        torch.tensor((t_grid / T).reshape(-1, 1), dtype=torch.float32),
    )
    Q_hat = probs.numpy().reshape(T, n, n)

e = 1e-12
kl = (Q * (np.log(Q + e) - np.log(Q_hat + e))).sum(axis=2).mean()
l1 = np.abs(Q - Q_hat).sum(axis=2).mean()

print(f'mean KL(Q || Q_hat) = {kl:.6f}')
print(f'mean L1(Q, Q_hat)   = {l1:.6f}')

# %% [cell 12]

i_mid = n // 2
t_disp = np.array([0, 1, 2, n], dtype=int)
t_idx = np.clip(t_disp, 0, T - 1)

blue = np.array([0.0, 0.0, 1.0])
red = np.array([1.0, 0.0, 0.0])
alphas = np.linspace(0.0, 1.0, len(t_idx))
colors = [(1.0 - a) * blue + a * red for a in alphas]

eps_log = 1e-12
x = np.arange(n)
plt.figure(figsize=(10, 4))
for td, t, c in zip(t_disp, t_idx, colors):
    plt.plot(x, np.log(Q[t, i_mid, :] + eps_log), color=c, linestyle='-', linewidth=2, label=f'true t={td}')
    plt.plot(x, np.log(Q_hat[t, i_mid, :] + eps_log), color=c, linestyle='--', linewidth=2, label=f'learned t={td}')

plt.xlabel('state index j')
plt.ylabel(r'$\log Q^t_{i,j}$')
plt.title(rf'Central row comparison (log-scale) at $i={i_mid}$')
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.show()

# %% [cell 13]


# %% [cell 14]


# %% [cell 15]

