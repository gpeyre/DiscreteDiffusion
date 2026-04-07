#!/usr/bin/env python3
"""Generate figures with the exact same logic as the notebook.

Outputs:
- forward_trajectories.pdf
- forward_histograms.pdf
- backward_trajectories.pdf
- backward_histograms.pdf
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    plt.close()


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
        colors = [(1-a) * blue + a * red for a in alphas]
        for ax, s, t, c in zip(axes, s_steps, t_steps, colors):
            ax.bar(np.arange(nn), H[:, s], color=c)
            ax.set_ylabel(f't={t}')
    else:
        t_steps = _fixed_steps_forward(TT, n)
        alphas = np.linspace(1.0, 0.0, len(t_steps))  # red -> blue
        colors = [(1-a) * blue + a * red for a in alphas]
        for ax, t, c in zip(axes, t_steps, colors):
            ax.bar(np.arange(nn), H[:, t], color=c)
            ax.set_ylabel(f't={t}')

    axes[-1].set_xlabel('state index')
    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    plt.close(fig)


if __name__ == '__main__':
    # Same seeds and parameters as notebook.
    np.random.seed(0)
    n = 30
    T = 100
    N = 2500
    sigma = 0.75

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

    # Forward simulation
    X = np.zeros((N, T + 1), dtype=np.int64)
    H = np.zeros((n, T + 1), dtype=np.float64)
    X[:, 0] = np.random.choice(n, size=N, p=h0)
    H[:, 0] = h0
    for t in range(T):
        X[:, t + 1] = sample_categorical_batch(P[X[:, t], :])
        H[:, t + 1] = P.T @ H[:, t]

    # Reverse kernels Q^t between t+1 and t
    Q = np.zeros((T, n, n), dtype=np.float64)
    for t in range(T):
        d_t = H[:, t]
        d_tp1 = H[:, t + 1]
        inv_d_tp1 = np.zeros_like(d_tp1)
        np.divide(1.0, d_tp1, out=inv_d_tp1, where=d_tp1 > 0)
        Q[t] = np.diag(inv_d_tp1) @ P.T @ np.diag(d_t)
        row_sums = Q[t].sum(axis=1, keepdims=True)
        nz = row_sums[:, 0] > 0
        Q[t, nz, :] /= row_sums[nz]

    # Backward simulation (empirical), same as notebook
    Y = np.zeros((N, T + 1), dtype=np.int64)
    Y[:, 0] = np.random.choice(n, size=N, p=H[:, T])
    for s in range(T):
        t = T - 1 - s
        Y[:, s + 1] = sample_categorical_batch(Q[t, Y[:, s], :])

    G = np.zeros((n, T + 1), dtype=np.float64)
    for s in range(T + 1):
        c = np.bincount(Y[:, s], minlength=n)
        G[:, s] = c / c.sum()

    out = Path(__file__).resolve().parent
    plot_trajectories(X, 'Forward trajectories', reverse_time=False, num_show=30, save_path=out / 'forward_trajectories.pdf')
    plot_histograms(H, 'Forward histograms', reverse_time=False, save_path=out / 'forward_histograms.pdf')
    plot_trajectories(Y, 'Backward trajectories (t=T to t=0)', reverse_time=True, num_show=30, save_path=out / 'backward_trajectories.pdf')
    plot_histograms(G, 'Backward empirical histograms (t=T to t=0)', reverse_time=True, save_path=out / 'backward_histograms.pdf')

    print('Generated: forward_trajectories.pdf, forward_histograms.pdf, backward_trajectories.pdf, backward_histograms.pdf')
