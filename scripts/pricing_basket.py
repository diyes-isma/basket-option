import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim


def monte_carlo_basket_option(S0, K, T, r, sigma, rho, num_simulations=10000, num_steps=100):
    """Monte Carlo simulation for a basket option price under Black-Scholes assumptions."""
    dt = T / num_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    # Simulate correlated Brownian motions for two assets: the assets must be correlated.
    # If the correlation is negative: that would increase the vol. Otherwise, it decrease the vol.
    
    z1 = np.random.normal(size=(num_simulations, num_steps))
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=(num_simulations, num_steps))
    
    # Simulate price paths
    S1 = S0[0] * np.exp(np.cumsum(drift[0] + diffusion[0] * z1, axis=1))
    S2 = S0[1] * np.exp(np.cumsum(drift[1] + diffusion[1] * z2, axis=1))
    
    # Calculate basket price (weighted average of assets)
    basket = 0.5 * S1[:, -1] + 0.5 * S2[:, -1]
    
    # Calculate option payoff and discount it back to present value
    payoff = np.maximum(basket - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoff)
    
    return option_price