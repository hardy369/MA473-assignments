import numpy as np
import matplotlib.pyplot as plt

# Input parameters
S0 = 50
K = 50
T = 1
r = 0.08
sigma = 0.30

def get_ud(model, dt):
    if model == 'A':
        u = np.exp(sigma * np.sqrt(dt))
        d = np.exp(-sigma * np.sqrt(dt))
    elif model == 'B':
        beta = 0.5 * (np.exp(-r * dt) + np.exp((r + sigma**2) * dt))
        u = beta + np.sqrt(beta**2 - 1)
        d = beta - np.sqrt(beta**2 - 1)
    else:
        raise ValueError("Invalid model: Choose 'A' or 'B'")
    return u, d

def get_p(u, d, dt):
    return (np.exp(r * dt) - d) / (u - d)

def binomial_tree_option(S0, K, T, r, sigma, M, option_type='european', model='A'):
    dt = T / M
    u, d = get_ud(model, dt)
    p = get_p(u, d, dt)
    discount = np.exp(-r * dt)

    # Stock prices at maturity
    stock_tree = np.zeros((M + 1, M + 1))
    stock_tree[0, 0] = S0
    for i in range(1, M + 1):
        stock_tree[i, 0] = stock_tree[i - 1, 0] * d
        for j in range(1, i + 1):
            stock_tree[i, j] = stock_tree[i - 1, j - 1] * u

    # Option values at maturity
    option_tree = np.zeros((M + 1, M + 1))
    for j in range(M + 1):
        if option_type == 'call':
            option_tree[M, j] = max(0, stock_tree[M, j] - K)
        else:
            option_tree[M, j] = max(0, K - stock_tree[M, j])

    # Backward induction
    for i in range(M - 1, -1, -1):
        for j in range(i + 1):
            if option_type == 'american_put':
                exercise = max(0, K - stock_tree[i, j])
                hold = discount * (p * option_tree[i + 1, j + 1] + (1 - p) * option_tree[i + 1, j])
                option_tree[i, j] = max(exercise, hold)
            else:
                option_tree[i, j] = discount * (p * option_tree[i + 1, j + 1] + (1 - p) * option_tree[i + 1, j])

    return option_tree, stock_tree

def tabulate_values(M, model):
    times = [0.0, 0.25, 0.5, 0.75, 0.95]
    indices = [int(t * M) for t in times]
    print("\nTabulated Option Values (M = 20, Model = {}):".format(model))
    print("Time\tCall\tPut\tAmerican Put")
    for i in indices:
        call_tree, _ = binomial_tree_option(S0, K, T, r, sigma, M, 'call', model)
        put_tree, _ = binomial_tree_option(S0, K, T, r, sigma, M, 'put', model)
        amer_tree, _ = binomial_tree_option(S0, K, T, r, sigma, M, 'american_put', model)
        print(f"{i/M:.2f}\t{call_tree[i,0]:.4f}\t{put_tree[i,0]:.4f}\t{amer_tree[i,0]:.4f}")

def plot_options_vs_time(M, model):
    call_tree, _ = binomial_tree_option(S0, K, T, r, sigma, M, 'call', model)
    put_tree, _ = binomial_tree_option(S0, K, T, r, sigma, M, 'put', model)
    amer_tree, _ = binomial_tree_option(S0, K, T, r, sigma, M, 'american_put', model)

    times = np.linspace(0, T, M + 1)
    call_vals = [call_tree[i, 0] for i in range(M + 1)]
    put_vals = [put_tree[i, 0] for i in range(M + 1)]
    amer_vals = [amer_tree[i, 0] for i in range(M + 1)]

    plt.figure(figsize=(10, 6))
    plt.plot(times, call_vals, label='European Call')
    plt.plot(times, put_vals, label='European Put')
    plt.plot(times, amer_vals, label='American Put')
    plt.xlabel('Time')
    plt.ylabel('Option Value')
    plt.title(f'Option Values vs Time (M={M}, Model={model})')
    plt.legend()
    plt.grid()
    plt.show()

def plot_all_options_vs_stock(M, model):
    # Get the final (maturity) values
    call_tree, stock_tree = binomial_tree_option(S0, K, T, r, sigma, M, 'call', model)
    put_tree, _ = binomial_tree_option(S0, K, T, r, sigma, M, 'put', model)
    amer_tree, _ = binomial_tree_option(S0, K, T, r, sigma, M, 'american_put', model)

    stock_prices = stock_tree[M]
    call_prices = call_tree[M]
    put_prices = put_tree[M]
    amer_prices = amer_tree[M]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(stock_prices, call_prices, marker='o', label='European Call')
    plt.plot(stock_prices, put_prices, marker='s', label='European Put')
    plt.plot(stock_prices, amer_prices, marker='^', label='American Put')

    plt.xlabel('Stock Price at T')
    plt.ylabel('Option Value at T')
    plt.title(f'Option Value vs Stock Price at Maturity (M={M}, Model={model})')
    plt.legend()
    plt.grid(True)
    plt.show()



# --- Main execution for part (A) ---
print("European & American Option Pricing\n")
for M in [5, 10, 20]:
    for model in ['A', 'B']:
        print(f"\nM = {M}, Model = {model}")
        call_tree, _ = binomial_tree_option(S0, K, T, r, sigma, M, 'call', model)
        put_tree, _ = binomial_tree_option(S0, K, T, r, sigma, M, 'put', model)
        amer_tree, _ = binomial_tree_option(S0, K, T, r, sigma, M, 'american_put', model)
        print(f"Call Option Value at t=0: {call_tree[0, 0]:.4f}")
        print(f"Put Option Value at t=0: {put_tree[0, 0]:.4f}")
        print(f"American Put Option Value at t=0: {amer_tree[0, 0]:.4f}")

# --- Part (B) ---
tabulate_values(20, model='A')
tabulate_values(20, model='B')
# --- Part (C) ---
plot_options_vs_time(M=20, model='A')
plot_options_vs_time(M=20, model='B')
plot_all_options_vs_stock(M=20, model='A')
plot_all_options_vs_stock(M=20, model='B')
