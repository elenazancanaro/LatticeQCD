import random
import math
import matplotlib.pyplot as plt
import numpy as np

class QuantumPathIntegral:
    def __init__(self, slices, get_params):
        self.warmup, self.mc_steps, self.save_gap, self.step_size, self.slices, self.freq, self.mass = get_params(slices)

        self.path = [random.uniform(-0.5, 0.5) for _ in range(self.slices)]
        self.trial = [0.0] * self.slices
        self.old = [0.0] * self.slices
        self.auto_corr = [0.0] * self.slices
        self.auto_corr_sq = [0.0] * self.slices

        self.avg_x = 0.0
        self.avg_x2 = 0.0
        self.avg_x_sq = 0.0
        self.avg_x2_sq = 0.0

    def relax_system(self):
        print(f"Equilibrating {self.slices} time slices...")

        for _ in range(self.warmup):
            for i in range(self.slices):
                left = self.path[(i - 1) % self.slices]
                right = self.path[(i + 1) % self.slices]

                self.old[i] = self.path[i]
                move = self.step_size * (random.random() - 0.5) * 2
                self.trial[i] = self.path[i] + move

                delta_S = (
                    self.mass * 0.5 * (
                        (self.trial[i] - left)**2 + (right - self.trial[i])**2
                        - (self.old[i] - left)**2 - (right - self.old[i])**2
                    )
                    + 0.5 * self.mass * self.freq**2 * (
                        self.trial[i]**2 - self.old[i]**2
                    )
                )

                if random.random() < math.exp(-delta_S):
                    self.path[i] = self.trial[i]
                else:
                    self.path[i] = self.old[i]

    def collect_statistics(self):
        print(f"Running Monte Carlo for {self.slices} points...")
        norm_factor = self.save_gap / (2.0 * self.mass * self.freq * self.mc_steps)
        num_samples = self.mc_steps // self.save_gap

        for step in range(self.mc_steps):
            for i in range(self.slices):
                left = self.path[(i - 1) % self.slices]
                right = self.path[(i + 1) % self.slices]

                self.old[i] = self.path[i]
                move = self.step_size * (random.random() - 0.5) * 2
                self.trial[i] = self.path[i] + move

                delta_S = (
                    self.mass * 0.5 * (
                        (self.trial[i] - left)**2 + (right - self.trial[i])**2
                        - (self.old[i] - left)**2 - (right - self.old[i])**2
                    )
                    + 0.5 * self.mass * self.freq**2 * (
                        self.trial[i]**2 - self.old[i]**2
                    )
                )

                if random.random() < math.exp(-delta_S):
                    self.path[i] = self.trial[i]

            if step % self.save_gap == 0:
                for j in range(self.slices):
                    x0 = self.path[0]
                    xt = self.path[j]
                    prod = xt * x0 * norm_factor
                    self.auto_corr[j] += prod
                    self.auto_corr_sq[j] += prod**2

                    self.avg_x += self.path[j]
                    x_sq = self.path[j]**2
                    x2_term = x_sq / (2 * self.mass * self.freq)

                    self.avg_x2 += x2_term
                    self.avg_x_sq += x_sq
                    self.avg_x2_sq += x2_term**2

        renorm = self.save_gap / (self.mc_steps * self.slices)
        x_mean = self.avg_x * renorm
        x2_mean = self.avg_x2 * renorm

        x_var = (self.avg_x_sq * renorm - x_mean**2) / num_samples
        x2_var = (self.avg_x2_sq * renorm - x2_mean**2) / num_samples

        x_err = math.sqrt(abs(x_var))
        x2_err = math.sqrt(abs(x2_var))

        energy = 0.5 * self.mass * self.freq**2 * x2_mean + 0.5 * x2_mean / (self.mass * self.freq**2)
        energy_err = (
            0.5 * self.mass * self.freq**2 + 0.5 / (self.mass * self.freq**2)
        ) * x2_err

        print(f"<x^2> = {x2_mean:.6f} ± {x2_err:.6f}")
        print(f"<x>   = {x_mean:.6f} ± {x_err:.6f}")
        print(f"Estimated Ground State Energy: {energy:.6f} ± {energy_err:.6f}\n")

        return self.auto_corr, self.auto_corr_sq, energy, energy_err, x_mean, x_err


# Plotting and parameter setup

def setup_parameters(n):
    thermal_steps = 100000
    monte_carlo_steps = 1000000
    output_interval = 100
    trial_step = 0.5
    frequency = 1.0
    mass = 1.0
    return thermal_steps, monte_carlo_steps, output_interval, trial_step, n, frequency, mass


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


def visualize_correlations(all_corrs, all_errs, ns):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Time-Displaced Correlation $\\langle x(\\tau)x(0) \\rangle$ with Errors", fontsize=18)

    for idx, (corr, err) in enumerate(zip(all_corrs, all_errs)):
        ax = axes[idx // 2, idx % 2]
        normed = corr / corr[0]
        normed_err = err / corr[0]
        tau_vals = np.arange(len(corr))
        ax.errorbar(tau_vals, normed, yerr=normed_err, fmt='o-', lw=2, color='black', ecolor='red', capsize=3)
        ax.set_title(f"Time Slices: N = {ns[idx]}", fontsize=14)
        ax.set_xlabel("$\\tau$", fontsize=13)
        ax.set_ylabel("$C(\\tau)$", fontsize=13)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_avg_x(ns, x_means, x_errs):
    plt.figure(figsize=(8, 6))
    plt.errorbar(ns, x_means, yerr=x_errs, fmt='o-', color='darkred', capsize=4)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title("Average Position $\\langle x \\rangle$ vs Number of Slices", fontsize=16)
    plt.xlabel("Number of Time Slices (N)", fontsize=14)
    plt.ylabel("$\\langle x \\rangle$", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


if __name__ == "__main__":
    time_slices_list = [8, 16, 32, 64]
    correlations = []
    errors = []
    energies = []
    energy_errs = []
    x_means = []
    x_mean_errs = []

    for N in time_slices_list:
        qpi = QuantumPathIntegral(N, setup_parameters)
        qpi.relax_system()
        corr, corr_sq, e0, e0_err, x_mean, x_err = qpi.collect_statistics()

        num_samples = qpi.mc_steps // qpi.save_gap
        corr = np.array(corr)
        corr_sq = np.array(corr_sq)

        mean_corr = corr / num_samples
        mean_corr_sq = corr_sq / num_samples
        std_err = np.sqrt((mean_corr_sq - mean_corr**2) / num_samples)

        correlations.append(mean_corr)
        errors.append(std_err)
        energies.append(e0)
        energy_errs.append(e0_err)
        x_means.append(x_mean)
        x_mean_errs.append(x_err)

        print(f"N = {N}, ⟨x⟩ = {x_mean:.6e} ± {x_err:.2e}")
        print(f"Energy = {e0:.6f} ± {e0_err:.6f}")

    visualize_correlations(correlations, errors, time_slices_list)
    plot_avg_x(time_slices_list, x_means, x_mean_errs)
