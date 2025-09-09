import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
S0 = 50
K = 50
T = 1
r = 0.08
sigma = 0.3

# Black-Scholes European put (used in approximations)
def euro_put_bs(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# (1) Interpolation-based approximation
def american_put_interp(S, K, T, r, sigma):
    P_euro = euro_put_bs(S, K, T, r, sigma)
    
    premium = 0.02 * K * np.exp(-r*T) * (1 - np.exp(-0.5*S/K))
    return np.maximum(P_euro + premium, K - S)

# (2) Quadratic approximation
def american_put_quadratic(S, K, T, r, sigma):
    P_euro = euro_put_bs(S, K, T, r, sigma)
   
    m = np.maximum((K - S)/K, 0)
    premium = 0.5 * m**2 * K * np.exp(-r*T)
    return np.maximum(P_euro + premium, K - S)


S_grid = np.linspace(1, 100, 100)
T_grid = np.linspace(0.01, T, 50)

P_interp = np.zeros((len(T_grid), len(S_grid)))
P_quad = np.zeros((len(T_grid), len(S_grid)))

for i, t in enumerate(T_grid):
    for j, s in enumerate(S_grid):
        P_interp[i, j] = american_put_interp(s, K, t, r, sigma)
        P_quad[i, j] = american_put_quadratic(s, K, t, r, sigma)


fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
T_mesh, S_mesh = np.meshgrid(T_grid, S_grid)
ax1.plot_surface(S_mesh, T_mesh, P_interp.T, cmap="viridis")
ax1.set_title("American Put (Interpolation Approx)")
ax1.set_xlabel("Stock Price")
ax1.set_ylabel("Time to Maturity")
ax1.set_zlabel("Option Price")

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(S_mesh, T_mesh, P_quad.T, cmap="plasma")
ax2.set_title("American Put (Quadratic Approx)")
ax2.set_xlabel("Stock Price")
ax2.set_ylabel("Time to Maturity")
ax2.set_zlabel("Option Price")

plt.show()
