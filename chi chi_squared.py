import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest
from statsmodels.tsa.stattools import acf

# Enhanced Metropolis algorithm that tracks moments
def metropolis_with_moments(n_samples, initial_value, proposal_width=1, target_mu=0, target_sigma=1):
    samples = [initial_value]
    accepted = 0
    moments = {
        'X': [],
        'X^2': [],
        'X^4': [],
        'running_mean': []
    }

    for i in range(n_samples):
        current = samples[-1]
        proposed = np.random.normal(current, proposal_width)

        prob_current = norm.pdf(current, target_mu, target_sigma)
        prob_proposed = norm.pdf(proposed, target_mu, target_sigma)

        acceptance_prob = min(1, prob_proposed / prob_current)

        if np.random.rand() < acceptance_prob:
            samples.append(proposed)
            accepted += 1
        else:
            samples.append(current)

        # Track moments every 10 samples for efficiency
        if i % 10 == 0:
            current_samples = np.array(samples)
            moments['X'].append(np.mean(current_samples))
            moments['X^2'].append(np.mean(current_samples**2))
            moments['X^4'].append(np.mean(current_samples**4))
            moments['running_mean'].append(np.mean(current_samples))

    return np.array(samples), accepted/n_samples, moments

# Analytical moments for N(μ,σ²)
def analytical_moments(mu, sigma):
    return {
        'X': mu,
        'X^2': mu**2 + sigma**2,
        'X^4': mu**4 + 6*mu**2*sigma**2 + 3*sigma**4,
        'Var(X)': sigma**2
    }

# Statistical analysis functions
def effective_sample_size(samples, max_lag=100):
    autocorr = acf(samples, nlags=max_lag, fft=True)
    return len(samples) / (1 + 2 * np.sum(autocorr[1:]))

def batch_means_standard_error(samples, batch_size=100):
    n_batches = len(samples) // batch_size
    batch_means = [np.mean(samples[i*batch_size:(i+1)*batch_size]) for i in range(n_batches)]
    return np.std(batch_means) / np.sqrt(n_batches)

# Run simulation and analysis
n_samples = 50000
initial_value = 3.0
proposal_width = 2.4
target_mu, target_sigma = 0, 1

samples, acceptance_rate, moments = metropolis_with_moments(
    n_samples, initial_value, proposal_width, target_mu, target_sigma
)

# Discard burn-in
burn_in = int(0.2 * n_samples)
post_burn_samples = samples[burn_in:]

# Calculate analytical results
true_moments = analytical_moments(target_mu, target_sigma)

# Sample estimates
sample_estimates = {
    'X': np.mean(post_burn_samples),
    'X^2': np.mean(post_burn_samples**2),
    'X^4': np.mean(post_burn_samples**4),
    'Var(X)': np.var(post_burn_samples)
}

# Calculate errors
errors = {key: sample_estimates[key] - true_moments.get(key, np.nan)
          for key in sample_estimates}

# Statistical diagnostics
ess = effective_sample_size(post_burn_samples)
batch_se = batch_means_standard_error(post_burn_samples)
ks_stat, ks_pvalue = kstest((post_burn_samples - target_mu)/target_sigma, 'norm')

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3)

# Trace plot
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(samples[:5000], alpha=0.7)
ax1.axhline(target_mu, color='r', linestyle='--')
ax1.set_title(f'Trace Plot (First 5000 samples, Acceptance Rate: {acceptance_rate:.1%})')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Parameter Value')

# Moment convergence
ax2 = fig.add_subplot(gs[1, 0])
for moment in ['X', 'X^2', 'X^4']:
    ax2.plot(moments[moment], label=moment)
ax2.axhline(true_moments['X'], color='r', linestyle='--')
ax2.axhline(true_moments['X^2'], color='g', linestyle='--')
ax2.axhline(true_moments['X^4'], color='b', linestyle='--')
ax2.set_title('Moment Convergence')
ax2.legend()

# Distribution comparison
ax3 = fig.add_subplot(gs[1, 1])
x_vals = np.linspace(-4, 4, 200)
ax3.hist(post_burn_samples, bins=50, density=True, alpha=0.6, label='Samples')
ax3.plot(x_vals, norm.pdf(x_vals, target_mu, target_sigma), 'r--', label='True')
ax3.set_title('Distribution Comparison')
ax3.legend()

# Error analysis
ax4 = fig.add_subplot(gs[1, 2])
errors_relative = {k: np.abs(v/true_moments[k]) if k in true_moments else np.nan
                  for k, v in errors.items()}
ax4.bar(errors_relative.keys(), errors_relative.values())
ax4.set_title('Relative Errors')
ax4.set_yscale('log')

# Statistical diagnostics table
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')
diagnostics_text = (
    f'Effective Sample Size: {ess:.0f}\n'
    f'Batch Means SE of Mean: {batch_se:.4f}\n'
    f'KS Test p-value: {ks_pvalue:.3f}\n\n'
    'Moment | Sample Estimate | True Value | Error\n'
    '--------------------------------------------\n'
    f"X      | {sample_estimates['X']:.4f}        | {true_moments['X']:.4f}    | {errors['X']:.4f}\n"
    f"X²     | {sample_estimates['X^2']:.4f}      | {true_moments['X^2']:.4f}  | {errors['X^2']:.4f}\n"
    f"X⁴     | {sample_estimates['X^4']:.4f}      | {true_moments['X^4']:.4f}  | {errors['X^4']:.4f}\n"
    f"Var(X) | {sample_estimates['Var(X)']:.4f}      | {true_moments['Var(X)']:.4f}  | {errors['Var(X)']:.4f}"
)
ax5.text(0.1, 0.5, diagnostics_text, fontfamily='monospace', fontsize=12)

plt.tight_layout()
plt.savefig('moment_analysis.png', dpi=300, bbox_inches='tight')
plt.show()