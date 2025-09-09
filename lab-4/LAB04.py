import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Parameters
S0 = 50       # initial stock price
K = 50        # strike
T = 1.0       # maturity (years)
r = 0.08      # risk-free rate
sigma = 0.30  # volatility

# Grid setup
M = 200     # stock price steps
N = 200     # time steps
Smax = 5*K  # max stock price
dS = Smax / M
dt = T / N

S = np.linspace(0, Smax, M+1)    # stock grid
t = np.linspace(0, T, N+1)       # time grid

# Option value grid
V = np.zeros((M+1, N+1))

# Terminal payoff at maturity (t=T)
V[:, -1] = np.maximum(K - S, 0)

# Coefficients for CN scheme
j = np.arange(1, M)
a = 0.25*dt*(sigma**2*j**2 - r*j)
b = -dt*0.5*(sigma**2*j**2 + r)
c = 0.25*dt*(sigma**2*j**2 + r*j)

# Build matrices
A = np.diag(1-b) + np.diag(-a[1:], -1) + np.diag(-c[:-1], 1)
B = np.diag(1+b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)

# Early exercise boundary storage
exercise_boundary = []

# Solve backwards in time
for n in reversed(range(N)):
    # Right-hand side
    rhs = B @ V[1:M, n+1]

    # Boundary conditions
    rhs[0]  -= a[0]  * K  # at S=0, option ~ K
    rhs[-1] -= c[-1] * 0  # at S=Smax, option ~ 0

    # Initial guess
    x = V[1:M, n+1].copy()

    # PSOR iteration
    payoff = np.maximum(K - S[1:M], 0)
    omega = 1.2
    tol = 1e-6
    max_iter = 10000

    for it in range(max_iter):
        x_old = x.copy()
        for i in range(M-1):
            # Update using Gauss-Seidel with relaxation
            y = (rhs[i] 
                 - ( -a[i]*x[i-1] if i>0 else 0 )
                 - ( -c[i]*x[i+1] if i<M-2 else 0 )) / A[i,i]

            x[i] = max(payoff[i], x[i] + omega*(y - x[i]))
        if np.linalg.norm(x - x_old, np.inf) < tol:
            break

    V[1:M, n] = x
    V[0, n] = K   # boundary at S=0
    V[M, n] = 0   # boundary at S=Smax

    # Store early exercise boundary: lowest S where continuation > payoff
    idx = np.where(x > payoff)[0]
    if len(idx) > 0:
        exercise_boundary.append(S[idx[0]+1])  # +1 offset
    else:
        exercise_boundary.append(0)

exercise_boundary = exercise_boundary[::-1]  # reverse to match time

# --- Results ---
price_american = np.interp(S0, S, V[:,0])
print(f"American Put Price at S0={S0}: {price_american:.4f}")

# --- Plots ---
# 1. Option vs Stock Price
plt.figure()
for nplot in [0, int(N/4), int(N/2), N]:
    plt.plot(S, V[:,nplot], label=f"t={t[nplot]:.2f}")
plt.xlabel("Stock Price S")
plt.ylabel("Option Value V")
plt.title("American Put vs Stock Price")
plt.legend()
plt.show()

# 2. Option vs Time (for selected S values)
plt.figure()
for s_val in [30, 40, 50, 60]:
    idx = np.searchsorted(S, s_val)
    plt.plot(t, V[idx,:], label=f"S={s_val}")
plt.xlabel("Time t")
plt.ylabel("Option Value V")
plt.title("American Put vs Time")
plt.legend()
plt.show()

# 3. Early exercise boundary
plt.figure()
plt.plot(t[:-1], exercise_boundary)
plt.xlabel("Time t")
plt.ylabel("Early Exercise Boundary S*(t)")
plt.title("Early Exercise Boundary (American Put)")
plt.show()

# 3d plot
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
Tgrid, Sgrid = np.meshgrid(t, S)
ax.plot_surface(Tgrid, Sgrid, V, cmap='viridis')
ax.set_xlabel("Time t")
ax.set_ylabel("Stock Price S")
ax.set_zlabel("Option Value V")
ax.set_title("American Put Option Surface (Finite Difference)")
plt.show()

