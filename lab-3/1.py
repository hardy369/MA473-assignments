import numpy as np
import pandas as pd
from scipy.stats import norm

def bs_price(S, K, tau, r, sigma, kind='call'):
    if tau <= 0:
        return max(S - K, 0) if kind == 'call' else max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    if kind == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)
    return K*np.exp(-r*tau)*norm.cdf(-d2) - S*norm.cdf(-d1)

def fd_theta_solver(theta, S0, K, T, r, sigma, kind='call', times=None, M=100, N=1000):
    times = times or [0, 0.25, 0.5, 0.75, 0.95]
    Smax, dS, dt = 3*K, 3*K/M, T/N
    S = np.linspace(0, Smax, M+1)
    V = np.maximum(S-K, 0) if kind=='call' else np.maximum(K-S, 0)
    i = np.arange(1, M)
    a = -0.5*dt*(sigma**2*i**2 - r*i)
    b = 1 + dt*(sigma**2*i**2 + r)
    c = -0.5*dt*(sigma**2*i**2 + r*i)
    A = np.diag(1+theta*(b-1)) + np.diag(theta*a[1:],-1) + np.diag(theta*c[:-1],1)
    B = np.diag(1-(1-theta)*(b-1)) - np.diag((1-theta)*a[1:],-1) - np.diag((1-theta)*c[:-1],1)

    results, step_targets = {}, {int(round((T-t)/dt)):t for t in times}
    for n in range(N+1):
        if n in step_targets: results[step_targets[n]] = np.interp(S0, S, V)
        V[1:M] = np.linalg.solve(A, B@V[1:M])
        V[0], V[M] = (0, Smax-K*np.exp(-r*dt*n)) if kind=='call' else (K*np.exp(-r*dt*n), 0)
    return results
# Parameters

S0, K, T, r, sigma = 50, 50, 1, 0.08, 0.3
thetas = {0:"explicit", 1:"implicit", 0.5:"crank-nicolson"}
times = [0, 0.25, 0.5, 0.75, 0.95]

def results(kind):
    rows = []
    for th, name in thetas.items():
        prices = fd_theta_solver(th, S0, K, T, r, sigma, kind, times)
        rows += [[name, t, prices[t]] for t in times]
    rows += [['closed-form', t, bs_price(S0, K, T-t, r, sigma, kind)] for t in times]
    return pd.DataFrame(rows, columns=['Method','Time','Price'])

print("\nCALL OPTION PRICES")
print(results('call').pivot(index='Time', columns='Method', values='Price').round(4))

print("\nPUT OPTION PRICES")
print(results('put').pivot(index='Time', columns='Method', values='Price').round(4))
