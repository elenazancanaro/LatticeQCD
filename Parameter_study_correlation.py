import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import seaborn as sns

# Set up the plotting style (using your original params)
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


def run_parameter_study():
    """Study the effect of different parameters on MCMC performance"""
    # Base parameters
    base_params = {
        'n_samples': 10000,
        'initial_value': 5.0,
        'proposal_width': 2.4,
        'burn_in_ratio': 0.2,
        'target_mu': 0,
        'target_sigma': 1
    }

    # Parameter ranges to test
    param_variations = {
        'initial_value': [-10, -5, 0, 5, 10],  # Cold to hot starts
        'proposal_width': [0.1, 0.5, 1.0, 2.4, 5.0, 10.0],
        'burn_in_ratio': [0.0, 0.1, 0.2, 0.3, 0.5],
        'n_samples': [100, 1000, 5000, 10000, 20000]
    }

    # Results storage
    results = {param: {'values': [], 'acceptance': [], 'error_mu': [], 'error_sigma': []}
               for param in param_variations}

    # Run studies for each parameter
    for param, values in param_variations.items():
        print(f"\nStudying {param}...")
        for value in tqdm(values):
            current_params = base_params.copy()
            current_params[param] = value

            samples, acceptance = metropolis_algorithm(
                n_samples=current_params['n_samples'],
                initial_value=current_params['initial_value'],
                proposal_width=current_params['proposal_width'],
                target_mu=current_params['target_mu'],
                target_sigma=current_params['target_sigma']
            )

            # Calculate post burn-in statistics
            burn_in = int(current_params['burn_in_ratio'] * current_params['n_samples'])
            post_burn = samples[burn_in:]

            # Store results
            results[param]['values'].append(value)
            results[param]['acceptance'].append(acceptance)
            results[param]['error_mu'].append(np.abs(np.mean(post_burn) - current_params['target_mu']))
            results[param]['error_sigma'].append(np.abs(np.std(post_burn) - current_params['target_sigma']))

    return results


def plot_parameter_study(results):
    """Visualize the parameter study results"""
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    metrics = ['acceptance', 'error_mu', 'error_sigma']
    titles = ['Acceptance Rate', 'μ Error', 'σ Error']

    for i, param in enumerate(results.keys()):
        for j, metric in enumerate(metrics):
            ax = axes[i * 3 + j]
            ax.plot(results[param]['values'], results[param][metric],
                    marker='o', color='#1f77b4')
            ax.set_xlabel(param)
            ax.set_ylabel(metric)
            ax.set_title(f"{param} vs {titles[j]}")
            ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('parameter_study.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_hot_cold_starts():
    """Compare hot (near target) vs cold (far from target) starts"""
    starts = {
        'Cold start (x=10)': 10.0,
        'Warm start (x=5)': 5.0,
        'Hot start (x=0)': 0.0
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (label, start) in enumerate(starts.items()):
        samples, acceptance = metropolis_algorithm(
            n_samples=10000,
            initial_value=start,
            proposal_width=2.4,
            target_mu=0,
            target_sigma=1
        )

        # Trace plot
        axes[0].plot(samples[:500], alpha=0.7, label=label)

        # Histogram (post burn-in)
        post_burn = samples[2000:]
        axes[1].hist(post_burn, bins=50, density=True, alpha=0.5, label=label)

        # Convergence plot
        running_mean = [np.mean(samples[:n]) for n in range(1, len(samples))]
        axes[2].plot(running_mean, alpha=0.7, label=label)

    # Formatting
    axes[0].set_title('Early Trace (First 500 samples)')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Parameter Value')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.7)

    axes[1].set_title('Post Burn-in Distribution')
    axes[1].set_xlabel('Parameter Value')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.7)

    axes[2].set_title('Running Mean Convergence')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Mean Estimate')
    axes[2].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[2].legend()
    axes[2].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('hot_cold_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# Run the studies
results = run_parameter_study()
plot_parameter_study(results)
compare_hot_cold_starts()
