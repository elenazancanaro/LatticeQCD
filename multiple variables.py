import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.tsa.stattools import acf


class MultivariateMetropolis:
    def __init__(self, dim=3, target_mean=None, target_cov=None):
        self.dim = dim
        self.target_mean = target_mean if target_mean is not None else np.zeros(dim)
        self.target_cov = target_cov if target_cov is not None else np.eye(dim)
        self.proposal_cov = 2.4 ** 2 / self.dim * np.eye(dim)  # Optimal scaling

    def target_distribution(self, x):
        return multivariate_normal.pdf(x, mean=self.target_mean, cov=self.target_cov)

    def sample(self, n_samples, initial_value=None):
        if initial_value is None:
            initial_value = np.random.randn(self.dim) * 3  # Cold start

        samples = [initial_value]
        accepted = 0

        for _ in range(n_samples):
            current = samples[-1]
            proposed = np.random.multivariate_normal(current, self.proposal_cov)

            acceptance_prob = min(1, self.target_distribution(proposed) / self.target_distribution(current))

            if np.random.rand() < acceptance_prob:
                samples.append(proposed)
                accepted += 1
            else:
                samples.append(current)

        return np.array(samples), accepted / n_samples

    def analyze_moments(self, samples, burn_in=0.2):
        # Discard burn-in
        samples = samples[int(len(samples) * burn_in):]

        # Basic moments
        results = {
            'means': np.mean(samples, axis=0),
            'variances': np.var(samples, axis=0),
            'cross_moments': {}
        }

        # Crossed moments (up to 4th order)
        for i in range(self.dim):
            for j in range(i, self.dim):
                key = f'X_{i + 1}X_{j + 1}'
                results['cross_moments'][key] = np.mean(samples[:, i] * samples[:, j])

                for k in range(j, self.dim):
                    key = f'X_{i + 1}X_{j + 1}X_{k + 1}'
                    results['cross_moments'][key] = np.mean(samples[:, i] * samples[:, j] * samples[:, k])

                    for l in range(k, self.dim):
                        key = f'X_{i + 1}X_{j + 1}X_{k + 1}X_{l + 1}'
                        results['cross_moments'][key] = np.mean(
                            samples[:, i] * samples[:, j] * samples[:, k] * samples[:, l])

        return results

    def theoretical_moments(self):
        moments = {
            'means': self.target_mean,
            'variances': np.diag(self.target_cov),
            'cross_moments': {}
        }

        # Using Isserlis' theorem for Gaussian moments
        cov = self.target_cov
        for i in range(self.dim):
            for j in range(i, self.dim):
                moments['cross_moments'][f'X_{i + 1}X_{j + 1}'] = cov[i, j] + self.target_mean[i] * self.target_mean[j]

                for k in range(j, self.dim):
                    term = (cov[i, j] * self.target_mean[k] + cov[i, k] * self.target_mean[j] + cov[j, k] *
                            self.target_mean[i] +
                            self.target_mean[i] * self.target_mean[j] * self.target_mean[k])
                    moments['cross_moments'][f'X_{i + 1}X_{j + 1}X_{k + 1}'] = term

                    for l in range(k, self.dim):
                        term = (cov[i, j] * cov[k, l] + cov[i, k] * cov[j, l] + cov[i, l] * cov[j, k] +
                                cov[i, j] * self.target_mean[k] * self.target_mean[l] +
                                cov[i, k] * self.target_mean[j] * self.target_mean[l] +
                                cov[i, l] * self.target_mean[j] * self.target_mean[k] +
                                cov[j, k] * self.target_mean[i] * self.target_mean[l] +
                                cov[j, l] * self.target_mean[i] * self.target_mean[k] +
                                cov[k, l] * self.target_mean[i] * self.target_mean[j] +
                                self.target_mean[i] * self.target_mean[j] * self.target_mean[k] * self.target_mean[l])
                        moments['cross_moments'][f'X_{i + 1}X_{j + 1}X_{k + 1}X_{l + 1}'] = term

        return moments


# Example Usage: 3D Case
np.random.seed(42)

# Create correlated 3D Gaussian
dim = 3
mean_3d = np.array([1, -1, 0])
cov_3d = np.array([
    [1.0, 0.5, -0.3],
    [0.5, 2.0, 0.1],
    [-0.3, 0.1, 0.8]
])

mm_3d = MultivariateMetropolis(dim=3, target_mean=mean_3d, target_cov=cov_3d)
samples_3d, acc_rate_3d = mm_3d.sample(n_samples=50000)

# Analyze moments
sample_moments_3d = mm_3d.analyze_moments(samples_3d)
true_moments_3d = mm_3d.theoretical_moments()


# Visualization
def plot_3d_results(samples, true_mean, sample_mean):
    fig = plt.figure(figsize=(18, 6))

    # 3D Scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(samples[-1000:, 0], samples[-1000:, 1], samples[-1000:, 2],
                alpha=0.3, s=10)
    ax1.scatter(*true_mean, c='r', s=100, marker='*', label='True Mean')
    ax1.scatter(*sample_mean, c='g', s=100, marker='o', label='Sample Mean')
    ax1.set_title('3D Sample Distribution')
    ax1.legend()

    # Marginal distributions
    ax2 = fig.add_subplot(132)
    for i in range(3):
        sns.kdeplot(samples[:, i], label=f'X_{i + 1}')
    ax2.set_title('Marginal Distributions')
    ax2.legend()

    # Cross moment comparison
    ax3 = fig.add_subplot(133)
    cross_keys = [k for k in sample_moments_3d['cross_moments'] if 'X_1X_2' in k or 'X_1X_3' in k or 'X_2X_3' in k]
    true_vals = [true_moments_3d['cross_moments'][k] for k in cross_keys]
    sample_vals = [sample_moments_3d['cross_moments'][k] for k in cross_keys]

    x = np.arange(len(cross_keys))
    width = 0.35
    ax3.bar(x - width / 2, true_vals, width, label='True')
    ax3.bar(x + width / 2, sample_vals, width, label='Sample')
    ax3.set_xticks(x)
    ax3.set_xticklabels(cross_keys, rotation=45)
    ax3.set_title('Crossed Moments Comparison')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('3d_mcmc_results.png', dpi=300)
    plt.show()


plot_3d_results(samples_3d, mean_3d, sample_moments_3d['means'])

# 4D Case
dim = 4
mean_4d = np.array([1, -1, 0, 2])
cov_4d = np.array([
    [1.0, 0.5, -0.3, 0.2],
    [0.5, 2.0, 0.1, -0.4],
    [-0.3, 0.1, 0.8, 0.0],
    [0.2, -0.4, 0.0, 1.5]
])

mm_4d = MultivariateMetropolis(dim=4, target_mean=mean_4d, target_cov=cov_4d)
samples_4d, acc_rate_4d = mm_4d.sample(n_samples=100000)

# Analyze moments
sample_moments_4d = mm_4d.analyze_moments(samples_4d)
true_moments_4d = mm_4d.theoretical_moments()

# Print important comparisons
print("\n4D Case Important Crossed Moments:")
print("{:<15} {:<15} {:<15} {:<10}".format("Moment", "True Value", "Sample Estimate", "Error"))
for k in ['X_1X_2', 'X_1X_2X_3', 'X_1X_2X_3X_4']:
    true = true_moments_4d['cross_moments'][k]
    sample = sample_moments_4d['cross_moments'][k]
    print("{:<15} {:<15.4f} {:<15.4f} {:<10.4f}".format(k, true, sample, sample - true))


# Convergence diagnostics
def gelman_rubin_diagnostic(chains):
    m = len(chains)
    n = len(chains[0])

    # Between-chain variance
    chain_means = np.array([np.mean(chain, axis=0) for chain in chains])
    global_mean = np.mean(chain_means, axis=0)
    B = n / (m - 1) * np.sum((chain_means - global_mean) ** 2, axis=0)

    # Within-chain variance
    chain_vars = np.array([np.var(chain, axis=0, ddof=1) for chain in chains])
    W = np.mean(chain_vars, axis=0)

    # Estimated variance
    var_estimate = (n - 1) / n * W + B / n
    R_hat = np.sqrt(var_estimate / W)
    return R_hat


# Run multiple chains for diagnostics
chains_4d = [mm_4d.sample(n_samples=20000)[0] for _ in range(4)]
r_hat = gelman_rubin_diagnostic(chains_4d)
print("\nGelman-Rubin R_hat:", r_hat)