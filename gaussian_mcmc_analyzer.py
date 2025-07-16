import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from statsmodels.tsa.stattools import acf
from sklearn.utils import resample

class MCMCAnalyzer:
    def __init__(self, dim=3, target_mean=None, target_cov=None):
        self.dim = dim
        self.target_mean = target_mean if target_mean is not None else np.zeros(dim)
        self.target_cov = target_cov if target_cov is not None else np.eye(dim)
        self.proposal_cov = (2.4**2/dim) * np.eye(dim)
        
    def sample(self, n_samples, initial_value=None):
        if initial_value is None:
            initial_value = np.random.randn(self.dim) * 3
            
        samples = [initial_value]
        accepted = 0
        
        for _ in range(n_samples):
            current = samples[-1]
            proposed = np.random.multivariate_normal(current, self.proposal_cov)
            
            acceptance_prob = min(1, multivariate_normal.pdf(proposed, self.target_mean, self.target_cov) / 
                                 multivariate_normal.pdf(current, self.target_mean, self.target_cov))
            
            if np.random.rand() < acceptance_prob:
                samples.append(proposed)
                accepted += 1
            else:
                samples.append(current)
                
        return np.array(samples), accepted/n_samples
    
    def analyze_chain(self, samples, burn_in=0.2):
        samples = samples[int(len(samples)*burn_in):]
        
        results = {
            'means': np.mean(samples, axis=0),
            'covariance': np.cov(samples.T),
            'autocorrelation': self._calculate_autocorrelation(samples),
            'effective_sample_size': self._effective_sample_size(samples),
            'jackknife_errors': self._jackknife_estimates(samples),
            'convergence_diagnostics': self._gelman_rubin(samples)
        }
        return results
    
    def _calculate_autocorrelation(self, samples, max_lag=50):
        """Compute autocorrelation for each dimension"""
        acfs = []
        for i in range(samples.shape[1]):
            acf_vals = acf(samples[:, i], nlags=max_lag, fft=True)
            acfs.append(acf_vals)
        return np.array(acfs)
    
    def _effective_sample_size(self, samples):
        """Compute ESS for each dimension"""
        n = len(samples)
        ess = []
        for i in range(samples.shape[1]):
            acf_vals = acf(samples[:, i], nlags=100, fft=True)
            tau = 1 + 2 * np.sum(acf_vals[1:])
            ess.append(n / tau)
        return np.array(ess)
    
    def _jackknife_estimates(self, samples, n_blocks=20):
        """Jackknife resampling for error estimation"""
        block_size = len(samples) // n_blocks
        estimates = []
        
        for i in range(samples.shape[1]):
            # Create blocks
            blocks = [samples[k*block_size:(k+1)*block_size, i] for k in range(n_blocks)]
            
            # Compute partial estimates
            partial_means = []
            for k in range(n_blocks):
                # Leave-one-out ensemble
                loo_ensemble = np.concatenate([blocks[j] for j in range(n_blocks) if j != k])
                partial_means.append(np.mean(loo_ensemble))
            
            # Jackknife estimate
            theta_dot = np.mean(partial_means)
            bias = (n_blocks - 1) * (theta_dot - np.mean(samples[:, i]))
            std_err = np.sqrt((n_blocks - 1)/n_blocks * np.sum([(th - theta_dot)**2 for th in partial_means]))
            
            estimates.append({
                'mean': np.mean(samples[:, i]),
                'bias': bias,
                'std_error': std_err,
                'confidence_interval': (
                    np.mean(samples[:, i]) - 1.96 * std_err,
                    np.mean(samples[:, i]) + 1.96 * std_err
                )
            })
        
        return estimates
    
    def _gelman_rubin(self, samples, n_chains=4):
        """Gelman-Rubin diagnostic using subchains"""
        chain_length = len(samples) // n_chains
        chains = [samples[i*chain_length:(i+1)*chain_length] for i in range(n_chains)]
        
        # Between-chain variance
        chain_means = np.array([np.mean(chain, axis=0) for chain in chains])
        global_mean = np.mean(chain_means, axis=0)
        B = chain_length/(n_chains - 1) * np.sum((chain_means - global_mean)**2, axis=0)
        
        # Within-chain variance
        chain_vars = np.array([np.var(chain, axis=0, ddof=1) for chain in chains])
        W = np.mean(chain_vars, axis=0)
        
        # Estimated variance
        var_estimate = (chain_length - 1)/chain_length * W + B/chain_length
        R_hat = np.sqrt(var_estimate/W)
        return R_hat

def plot_diagnostics(samples, analysis_results, dim_names=None):
    if dim_names is None:
        dim_names = [f'X_{i+1}' for i in range(samples.shape[1])]
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Trace Plots
    ax1 = fig.add_subplot(231)
    for i in range(samples.shape[1]):
        ax1.plot(samples[:2000,i], alpha=0.7, label=dim_names[i])
    ax1.set_title('Trace Plots (First 2000 Samples)')
    ax1.legend()
    
    # 2. Autocorrelation
    ax2 = fig.add_subplot(232)
    for i in range(samples.shape[1]):
        ax2.plot(analysis_results['autocorrelation'][i,:20], 
                marker='o', markersize=4, label=dim_names[i])
    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_title('Autocorrelation (First 20 Lags)')
    ax2.legend()
    
    # 3. ESS and R-hat
    ax3 = fig.add_subplot(233)
    x = np.arange(len(dim_names))
    width = 0.35
    ax3.bar(x - width/2, analysis_results['effective_sample_size'], 
           width, label='ESS')
    ax3.bar(x + width/2, analysis_results['convergence_diagnostics'], 
           width, label='R-hat')
    ax3.set_xticks(x)
    ax3.set_xticklabels(dim_names)
    ax3.axhline(1.1, color='r', linestyle='--', alpha=0.5)
    ax3.set_title('ESS and R-hat Diagnostics')
    ax3.legend()
    
    # 4. Jackknife Results
    ax4 = fig.add_subplot(234)
    means = [est['mean'] for est in analysis_results['jackknife_errors']]
    errors = [est['std_error'] for est in analysis_results['jackknife_errors']]
    ax4.errorbar(x, means, yerr=errors, fmt='o', capsize=5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(dim_names)
    ax4.set_title('Jackknife Estimates with 95% CI')
    
    # 5. Running Means
    ax5 = fig.add_subplot(235)
    for i in range(samples.shape[1]):
        running_mean = np.cumsum(samples[:,i]) / np.arange(1, len(samples)+1)
        ax5.plot(running_mean, alpha=0.7, label=dim_names[i])
    ax5.set_title('Running Means')
    ax5.legend()
    
    # 6. Pairwise Scatter
    if samples.shape[1] >= 2:
        ax6 = fig.add_subplot(236)
        ax6.scatter(samples[-5000:,0], samples[-5000:,1], alpha=0.3)
        ax6.set_xlabel(dim_names[0])
        ax6.set_ylabel(dim_names[1])
        ax6.set_title('Pairwise Scatter Plot')
    
    plt.tight_layout()
    plt.savefig('mcmc_diagnostics.png', dpi=300)
    plt.show()

# Example Usage
np.random.seed(42)

# Create correlated 3D target
dim = 3
mean = np.array([1, -1, 0])
cov = np.array([
    [1.0, 0.5, -0.3],
    [0.5, 2.0, 0.1],
    [-0.3, 0.1, 0.8]
])

# Run MCMC
analyzer = MCMCAnalyzer(dim=dim, target_mean=mean, target_cov=cov)
samples, acc_rate = analyzer.sample(n_samples=50000)

# Analyze chain
results = analyzer.analyze_chain(samples)

# Print key results
print(f"Acceptance rate: {acc_rate:.1%}")
print("\nEffective Sample Sizes:")
for i, ess in enumerate(results['effective_sample_size']):
    print(f"X_{i+1}: {ess:.0f} ({ess/len(samples):.1%})")

print("\nGelman-Rubin R-hat:")
for i, rhat in enumerate(results['convergence_diagnostics']):
    print(f"X_{i+1}: {rhat:.4f}")

print("\nJackknife Results:")
for i, est in enumerate(results['jackknife_errors']):
    print(f"X_{i+1}: mean = {est['mean']:.4f} ± {est['std_error']:.4f}")

# Plot diagnostics
plot_diagnostics(samples, results)
