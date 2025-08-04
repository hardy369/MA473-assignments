import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 5
T = 1
mu = 0.06
sigma = 0.3
lmbda = 5
n_paths = 10
n_steps = 1000
dt = T / n_steps
time = np.linspace(0, T, n_steps + 1)

np.random.seed(0)
paths = []

for _ in range(n_paths):
    S = [S0]
    for _ in range(n_steps):
        dW = np.random.normal(0, np.sqrt(dt))
        jump = np.random.poisson(lmbda * dt)
        q = 1 + abs(np.random.normal(0, 1)) / 10 if jump > 0 else 1
        dS = mu * S[-1] * dt + sigma * S[-1] * dW + (q - 1) * S[-1] * jump
        S.append(S[-1] + dS)
    paths.append(S)

# Plot
plt.figure(figsize=(10, 6))
for path in paths:
    plt.plot(time, path, lw=1)
plt.title('Jump Diffusion Process (10 Sample Paths)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()
# Time-dependent functions
def mu_t(t): return 0.0325 + (-0.25) * t
def sigma_t(t): return 0.012 + 0.0138 * t - 0.00125 * t**2

paths_td = []

for _ in range(n_paths):
    S = [S0]
    for i in range(n_steps):
        t = time[i]
        mu = mu_t(t)
        sigma = sigma_t(t)
        dW = np.random.normal(0, np.sqrt(dt))
        jump = np.random.poisson(lmbda * dt)
        q = 1 + abs(np.random.normal(0, 1)) / 10 if jump > 0 else 1
        dS = mu * S[-1] * dt + sigma * S[-1] * dW + (q - 1) * S[-1] * jump
        S.append(S[-1] + dS)
    paths_td.append(S)

# Plot
plt.figure(figsize=(10, 6))
for path in paths_td:
    plt.plot(time, path, lw=1)
plt.title('Jump Diffusion with Time-Dependent μ(t) and σ(t)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()
