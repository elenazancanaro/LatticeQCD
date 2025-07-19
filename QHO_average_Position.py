import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

        return self.auto_corr, self.auto_corr_sq, energy, energy_err, x_mean, x_err

def setup_parameters(n):
    return 100000, 1000000, 100, 0.5, n, 1.0, 1.0

def symmetric_exp(tau, A, E, beta):
    return A * (np.exp(-E * tau) + np.exp(-E * (beta - tau)))

def fit_energy_from_corr(correlation, beta):
    N = len(correlation)
    tau_vals = np.arange(N)
    norm_corr = correlation / correlation[0]

    tau_fit = tau_vals[1:N//2]
    corr_fit = norm_corr[1:N//2]

    popt, pcov = curve_fit(lambda tau, A, E: symmetric_exp(tau, A, E, beta), tau_fit, corr_fit, p0=(1.0, 1.0))
    A_fit, E_fit = popt
    E_fit_err = np.sqrt(np.diag(pcov))[1]
    return E_fit, E_fit_err, A_fit

def plot_fit(tau_vals, corr_vals, fit_vals, N):
    plt.figure(figsize=(8, 6))
    plt.plot(tau_vals, corr_vals, 'o', label="Data", markersize=4)
    plt.plot(tau_vals, fit_vals, '-', label="Fit", linewidth=2)
    plt.xlabel(r'$\\tau$', fontsize=14)
    plt.ylabel(r'$C(\\tau)/C(0)$', fontsize=14)
    plt.title(f'Correlation Fit for N = {N}', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    Ns = [8, 16, 32, 64]
    for N in Ns:
        qpi = QuantumPathIntegral(N, setup_parameters)
        qpi.relax_system()
        corr, corr_sq, E_virial, E_virial_err, x_mean, x_err = qpi.collect_statistics()

        beta = 1.0
        num_samples = qpi.mc_steps // qpi.save_gap
        corr = np.array(corr) / num_samples
        corr_sq = np.array(corr_sq) / num_samples
        std_err = np.sqrt((corr_sq - corr**2) / num_samples)

        E_fit, E_fit_err, A_fit = fit_energy_from_corr(corr, beta)
        tau_vals = np.arange(N)
        norm_corr = corr / corr[0]
        fit_vals = symmetric_exp(tau_vals, A_fit, E_fit, beta)

        print(f"N = {N}")
        print(f"  ⟨x²⟩ energy   = {E_virial:.6f} ± {E_virial_err:.6f}")
        print(f"  Fitted energy = {E_fit:.6f} ± {E_fit_err:.6f}\n")

        plot_fit(tau_vals, norm_corr, fit_vals, N)
