import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set your preferred style
params = {
    "ytick.color": "black",
    "xtick.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "font.family": "serif",
    "font.serif": "DejaVu Serif",
    "figure.facecolor": "white",
    "axes.facecolor": "white"
}
plt.rcParams.update(params)


def target_distribution(x, mu=0, sigma=1):
    """Gaussian target distribution we want to sample from"""
    return norm.pdf(x, loc=mu, scale=sigma)


def metropolis_algorithm(n_samples, initial_value, proposal_width=1, target_mu=0, target_sigma=1):
    """
    Metropolis algorithm for sampling from a Gaussian distribution
    """
    samples = [initial_value]
    accepted = 0

    for _ in range(n_samples):
        current = samples[-1]
        proposed = np.random.normal(current, proposal_width)

        prob_current = target_distribution(current, target_mu, target_sigma)
        prob_proposed = target_distribution(proposed, target_mu, target_sigma)

        acceptance_prob = min(1, prob_proposed / prob_current)

        if np.random.rand() < acceptance_prob:
            samples.append(proposed)
            accepted += 1
        else:
            samples.append(current)

    return np.array(samples), accepted / n_samples


# Parameters
n_samples = 10000
initial_value = 5
proposal_width = 2.4  # Optimal for Gaussian targets
target_mu, target_sigma = 0, 1

# Run MCMC
samples, acceptance_rate = metropolis_algorithm(
    n_samples, initial_value, proposal_width, target_mu, target_sigma
)

# Discard burn-in
burn_in = int(0.2 * n_samples)
post_burn_samples = samples[burn_in:]

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Trace plot
ax1.plot(samples, color='#1f77b4', alpha=0.8)
ax1.axvline(burn_in, color='#d62728', linestyle='--', linewidth=1.5)
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Parameter Value', fontsize=12)
ax1.set_title('Markov Chain Trace', fontsize=14, pad=10)
ax1.grid(True, linestyle=':', alpha=0.7)

# Histogram and KDE
n_bins = 50
counts, bins, patches = ax2.hist(post_burn_samples, bins=n_bins, density=True,
                                 color='#1f77b4', alpha=0.6, edgecolor='black', linewidth=0.5)

# Add KDE
from scipy.stats import gaussian_kde

kde = gaussian_kde(post_burn_samples)
x_vals = np.linspace(-4, 4, 500)
ax2.plot(x_vals, kde(x_vals), color='#2ca02c', linewidth=2.5, label='Sample KDE')

# True distribution
true_dist = norm.pdf(x_vals, target_mu, target_sigma)
ax2.plot(x_vals, true_dist, color='#d62728', linestyle='--', linewidth=2.5, label='True Distribution')

ax2.set_xlabel('Parameter Value', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Posterior Distribution', fontsize=14, pad=10)
ax2.legend(frameon=True, edgecolor='black')
ax2.grid(True, linestyle=':', alpha=0.7)

# Add info box
info_text = (f'Acceptance rate: {acceptance_rate:.1%}\n'
             f'Sample mean: {np.mean(post_burn_samples):.3f}\n'
             f'Sample std: {np.std(post_burn_samples):.3f}')
props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9)
ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes,
         verticalalignment='top', bbox=props, fontsize=10)

plt.tight_layout()
plt.savefig('mcmc_gaussian_black_theme.png', dpi=300, bbox_inches='tight')
plt.show()