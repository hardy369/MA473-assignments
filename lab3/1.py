import numpy as np
import pandas as pd
from scipy.stats import norm


def bs_price(S, K, tau, r, sigma, kind='call'):
    if tau <= 0:
        return max(S - K, 0) if kind == 'call' else max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    if kind == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r*tau) * norm.cdf(d2)
    else:
        return K * np.exp(-r*tau) * norm.cdf(-d2) - S * norm.cdf(-d1)

def fd_solver(method, S0, K, T, r, sigma, kind='call', times=None, M=150, N=800):
    if times is None:
        times = [0, 0.25, 0.5, 0.75, 0.95]
    Smax = 3*K  

    # Stability  for explicit method
    if method == 'explicit':
        dt_stable = 0.9 / (sigma**2 * M**2 + r * M)
        if T / N > dt_stable:
            N = int(T / dt_stable) + 1

    dS = Smax / M
    dt = T / N
    S = np.linspace(0, Smax, M+1)
    V = np.maximum(S - K, 0) if kind == 'call' else np.maximum(K - S, 0)
    i = np.arange(1, M)
    alpha = 0.5 * dt * (sigma**2 * i**2 - r * i)
    beta = 1 - dt * (sigma**2 * i**2 + r)
    gamma = 0.5 * dt * (sigma**2 * i**2 + r * i)
    results = {}

    step_targets = {int(round((T - t) / dt)): t for t in times}

    if method == 'explicit':
        for n in range(N+1):
            if n in step_targets:
                results[step_targets[n]] = np.interp(S0, S, V)
            V[1:M] = alpha * V[0:M-1] + beta * V[1:M] + gamma * V[2:M+1]
            if kind == 'call':
                V[0] = 0
                V[M] = Smax - K * np.exp(-r * dt * n)
            else:
                V[0] = K * np.exp(-r * dt * n)
                V[M] = 0

    elif method == 'implicit':
        A = np.zeros((M-1, M-1))
        for j in range(M-1):
            A[j, j] = 1 + dt * (sigma**2 * (j+1)**2 + r)
            if j > 0:
                A[j, j-1] = -0.5 * dt * (sigma**2 * (j+1)**2 - r*(j+1))
            if j < M-2:
                A[j, j+1] = -0.5 * dt * (sigma**2 * (j+1)**2 + r*(j+1))
        for n in range(N+1):
            if n in step_targets:
                results[step_targets[n]] = np.interp(S0, S, V)
            b = V[1:M].copy()
            V[1:M] = np.linalg.solve(A, b)

    elif method == 'crank-nicolson':
        A = np.zeros((M-1, M-1))
        B = np.zeros((M-1, M-1))
        for j in range(M-1):
            a = -0.25 * dt * (sigma**2 * (j+1)**2 - r*(j+1))
            b_mid = 1 + 0.5 * dt * (sigma**2 * (j+1)**2 + r)
            c = -0.25 * dt * (sigma**2 * (j+1)**2 + r*(j+1))
            A[j, j] = b_mid
            B[j, j] = 2 - b_mid
            if j > 0:
                A[j, j-1] = a
                B[j, j-1] = -a
            if j < M-2:
                A[j, j+1] = c
                B[j, j+1] = -c
        for n in range(N+1):
            if n in step_targets:
                results[step_targets[n]] = np.interp(S0, S, V)
            b = B @ V[1:M]
            V[1:M] = np.linalg.solve(A, b)

    return results

S0, K, T, r, sigma = 50, 50, 1, 0.08, 0.3
methods = ['explicit', 'implicit', 'crank-nicolson']
times = [0, 0.25, 0.50, 0.75, 0.95]

call_results = []
put_results = []

for method in methods:
    call_prices = fd_solver(method, S0, K, T, r, sigma, 'call', times)
    put_prices = fd_solver(method, S0, K, T, r, sigma, 'put', times)
    for t in times:
        call_results.append([method, t, call_prices[t]])
        put_results.append([method, t, put_prices[t]])

# Closed form
for t in times:
    tau = T - t
    call_results.append(['closed-form', t, bs_price(S0, K, tau, r, sigma, 'call')])
    put_results.append(['closed-form', t, bs_price(S0, K, tau, r, sigma, 'put')])

# Tables
df_call = pd.DataFrame(call_results, columns=['Method', 'Time', 'Price'])
df_put = pd.DataFrame(put_results, columns=['Method', 'Time', 'Price'])

print("\nCALL OPTION PRICES")
print(df_call.pivot(index='Time', columns='Method', values='Price').round(4))

print("\nPUT OPTION PRICES")
print(df_put.pivot(index='Time', columns='Method', values='Price').round(4))
