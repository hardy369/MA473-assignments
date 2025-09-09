import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def price_asian_option_cn(S0, T, r, sigma, R_max, M, N):

    dt = T / M
    dr = R_max / N
    R = np.linspace(0, R_max, N + 1)
    t = np.linspace(0, T, M + 1)
    
    
    H_surface = np.zeros((N + 1, M + 1))

    H = np.maximum(1 - R / T, 0)
    H_surface[:, -1] = H
  
    i = np.arange(1, N)
    alpha = 0.25 * dt * (sigma**2 * R[i]**2 / dr**2 - (1 - r*R[i]) / dr)
    beta = -0.5 * dt * (sigma**2 * R[i]**2 / dr**2 + 1/dt)
    gamma = 0.25 * dt * (sigma**2 * R[i]**2 / dr**2 + (1 - r*R[i]) / dr)

    LHS = np.diag(1 - beta[0:N-1]) + np.diag(-gamma[0:N-2], k=1) + np.diag(-alpha[1:N-1], k=-1)
    
    RHS = np.diag(1 + beta[0:N-1]) + np.diag(gamma[0:N-2], k=1) + np.diag(alpha[1:N-1], k=-1)


    for j in range(M - 1, -1, -1):
   
        b = RHS @ H[1:N]
        H_interior = np.linalg.solve(LHS, b)
        H[1:N] = H_interior

        H[0] = 2 * H[1] - H[2] # d^2H/dR^2 = 0 at R=0 (Neumann)
        H[N] = 0 # H(R_max, t) = 0 (Dirichlet)

        H_surface[:, j] = H


    option_price = S0 * H[0]
    
    return option_price, H_surface, R, t


S0 = 100.0
T = 1.0


M = 200 
N = 200 
R_max = 4.0 * T 


r_values = [0.05, 0.09, 0.15]
sigma_values = [0.1, 0.2, 0.3]

results = np.zeros((len(r_values), len(sigma_values)))


for i, r in enumerate(r_values):
    for j, sigma in enumerate(sigma_values):
        price, _, _, _ = price_asian_option_cn(S0, T, r, sigma, R_max, M, N)
        results[i, j] = price


results_df = pd.DataFrame(results, index=[f"r = {r}" for r in r_values], columns=[f"sigma = {s}" for s in sigma_values])

print("--- Asian Call Option Prices using Crank-Nicolson ---")
print(results_df)
print("\n")



r_plot = 0.05   # from r in [0.02, 0.08]
sigma_plot = 0.3 # from sigma in [0.2, 0.4]


_, H_surface_plot, R_grid, t_grid = price_asian_option_cn(S0, T, r_plot, sigma_plot, R_max, M, N)


R_mesh, T_mesh = np.meshgrid(R_grid, t_grid)


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')


surf = ax.plot_surface(R_mesh.T, T_mesh.T, H_surface_plot, cmap='viridis', edgecolor='none')

ax.set_title(f'Solution Surface H(R, t) for r={r_plot}, sigma={sigma_plot}', fontsize=16)
ax.set_xlabel('State Variable R = I/S', fontsize=12)
ax.set_ylabel('Time t', fontsize=12)
ax.set_zlabel('H(R, t)', fontsize=12)
ax.view_init(elev=30, azim=-120) # Adjust viewing angle
fig.colorbar(surf, shrink=0.5, aspect=5)

print("--- Generating 3D Plot ---")
plt.show()