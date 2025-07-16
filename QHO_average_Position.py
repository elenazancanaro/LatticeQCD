import random
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

class QuantumPathIntegral:
    def __init__(self, slices, get_params):
        self.warmup, self.mc_steps, self.save_gap, self.step_size, self.slices, self.freq, self.mass = get_params(slices)

        # Initial path and data holders
        self.path = [random.uniform(-0.5, 0.5) for _ in range(self.slices)]
        self.trial = [0.0] * self.slices
        self.old = [0.0] * self.slices
        self.auto_corr = [0.0] * self.slices
        self.auto_corr_sq = [0.0] * self.slices

        # For energy averages
        self.avg_x = 0.0
        self.avg_x2 = 0.0

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
                    self.avg_x2 += self.path[j]**2 / (2 * self.mass * self.freq)

        renorm = self.save_gap / (self.mc_steps * self.slices)
        self.avg_x *= renorm
        self.avg_x2 *= renorm

        # Harmonic oscillator energy: E = <V> + <K>
        energy = 0.5 * self.mass * self.freq**2 * self.avg_x2 + 0.5 * self.avg_x2 / (self.mass * self.freq**2)

        print(f"<x^2> = {self.avg_x2:.6f}")
        print(f"<x> = {self.avg_x:.6f}")
        print(f"Estimated Ground State Energy: {energy:.6f}\n")

        return self.auto_corr, energy



# Style settings
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

def setup_parameters(n):
    thermal_steps = 100000
    monte_carlo_steps = 1000000
    output_interval = 100
    trial_step = 0.5
    frequency = 1.0
    mass = 1.0
    return thermal_steps, monte_carlo_steps, output_interval, trial_step, n, frequency, mass

def visualize_correlations(all_corrs, ns):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Time-Displaced Correlation $\\langle x(\\tau)x(0) \\rangle$", fontsize=18)

    for idx, corr in enumerate(all_corrs):
        ax = axes[idx // 2, idx % 2]
        normed = corr / corr[0]
        tau_vals = np.arange(len(corr))
        ax.plot(tau_vals, normed, lw=2, color='black')
        ax.set_title(f"Time Slices: N = {ns[idx]}", fontsize=14)
        ax.set_xlabel("$\\tau$", fontsize=13)
        ax.set_ylabel("$C(\\tau)$", fontsize=13)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_avg_x(ns, x_means):
    plt.figure(figsize=(8, 6))
    plt.plot(ns, x_means, 'o-', color='darkred', linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title("Average Position $\\langle x \\rangle$ vs Number of Slices", fontsize=16)
    plt.xlabel("Number of Time Slices (N)", fontsize=14)
    plt.ylabel("$\\langle x \\rangle$", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


if __name__ == "__main__":
    time_slices_list = [8, 16, 32, 64]
    correlations = []
    energies = []
    x_means = []

    for N in time_slices_list:
        qpi = QuantumPathIntegral(N, setup_parameters)
        qpi.relax_system()
        corr, e0 = qpi.collect_statistics()

        correlations.append(np.array(corr))
        energies.append(e0)
        x_means.append(qpi.avg_x)  # ⟨x⟩ for this N

        print(f"N = {N}, ⟨x⟩ = {qpi.avg_x:.6e}")

    visualize_correlations(correlations, time_slices_list)
    plot_avg_x(time_slices_list, x_means)

