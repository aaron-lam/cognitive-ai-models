"""Analysis of neuron spike model reveals exponential distribution maximizes entropy."""

from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.optimize import curve_fit


@dataclass
class ISIAnalysisParams:
    """Parameters for ISI analysis."""

    n_bins: int = 50
    isi_range: tuple[float, float] = (0, 0.25)


class DataLoader:
    """Handles downloading and loading of Steinmetz spike data."""

    STEINMETZ_URL = "https://osf.io/sy5xt/download"
    STATUS_OK = 200

    @classmethod
    def load_spike_times(cls) -> np.ndarray:
        """Download and load spike times data from Steinmetz dataset."""
        response = requests.get(cls.STEINMETZ_URL, timeout=10)
        if response.status_code != cls.STATUS_OK:
            msg = f"Failed to download data: status {response.status_code}"
            raise RuntimeError(msg)

        return np.load(io.BytesIO(response.content), allow_pickle=True)["spike_times"]


class ISIAnalyzer:
    """Analyzes inter-spike intervals from neural spike data."""

    def __init__(self, params: ISIAnalysisParams) -> None:
        """Init with params."""
        self.params = params

    def compute_isis(self, spike_times: np.ndarray) -> np.ndarray:
        """Compute inter-spike intervals from spike times."""
        return np.diff(spike_times)

    def estimate_pmf(
        self,
        isi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate probability mass function from ISI data."""
        bins = np.linspace(*self.params.isi_range, self.params.n_bins + 1)
        counts, _ = np.histogram(isi, bins)
        pmf = self._normalize_counts(counts)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        return bin_centers, pmf, counts

    @staticmethod
    def _normalize_counts(counts: np.ndarray) -> np.ndarray:
        """Normalize counts to create a probability mass function."""
        return counts / np.sum(counts)


class EntropyAnalyzer:
    """Analyzes entropy of spike timing distributions."""

    @staticmethod
    def shannon_entropy(pmf: np.ndarray) -> float:
        """Calculate Shannon entropy in bits for a discrete distribution."""
        pmf = pmf[pmf > 0]  # Remove zero probabilities
        return np.abs(-np.sum(pmf * np.log2(pmf)))

    def theoretical_max_entropy(
        self,
        mean_isi: float,
        bin_centers: np.ndarray,
    ) -> float:
        """Calculate theoretical maximum entropy for given mean ISI."""
        # For exponential distribution with same mean
        lambda_param = 1 / mean_isi
        exp_pmf = ExponentialFitter.exponential_pdf(bin_centers, lambda_param)
        exp_pmf = exp_pmf / np.sum(exp_pmf)  # Normalize
        return self.shannon_entropy(exp_pmf)

    def entropy_efficiency(self, observed_entropy: float, max_entropy: float) -> float:
        """Calculate entropy efficiency (ratio of observed to maximum entropy)."""
        return observed_entropy / max_entropy if max_entropy > 0 else 0


class ExponentialFitter:
    """Fits exponential distribution to ISI data."""

    @staticmethod
    def exponential_pdf(t: float | np.ndarray, lambd: float) -> float | np.ndarray:
        """Exponential probability density function."""
        return lambd * np.exp(-lambd * t)

    def fit(
        self,
        bin_centers: np.ndarray,
        pmf: np.ndarray,
        mean_isi: float,
    ) -> tuple[float, np.ndarray]:
        """Fit exponential distribution to ISI data."""
        valid = pmf > 0
        popt, _ = curve_fit(
            self.exponential_pdf,
            bin_centers[valid],
            pmf[valid],
            p0=[1 / mean_isi],
        )
        lambda_hat = popt[0]
        fitted_curve = self.exponential_pdf(bin_centers, lambda_hat)
        return lambda_hat, fitted_curve


class Visualizer:
    """Handles visualization of ISI analysis results."""

    def plot_analysis(  # noqa: PLR0913
        self,
        bin_centers: np.ndarray,
        pmf: np.ndarray,
        mean_isi: float,
        neuron_idx: int,
        fitted_exp: np.ndarray,
        observed_entropy: float,
        max_entropy: float,
        efficiency: float,
    ) -> None:
        """Create comprehensive plots for ISI distribution and entropy analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # ISI Distribution plot
        self._plot_isi_distribution(
            ax1,
            bin_centers,
            pmf,
            fitted_exp,
            neuron_idx,
            mean_isi,
        )

        # Entropy comparison plot
        self._plot_entropy_comparison(ax2, observed_entropy, max_entropy, efficiency)

        plt.tight_layout()
        plt.show()

    def _plot_isi_distribution(  # noqa: PLR0913
        self,
        ax: plt.Axes,
        bin_centers: np.ndarray,
        pmf: np.ndarray,
        fitted_exp: np.ndarray,
        neuron_idx: int,
        mean_isi: float,
    ) -> None:
        """Plot ISI distribution with fitted exponential."""
        ax.plot(bin_centers, pmf, "o-", label="Observed PMF", alpha=0.7)
        ax.plot(
            bin_centers,
            fitted_exp,
            "r--",
            label=f"Max Entropy Exponential\n(mean ISI = {mean_isi:.3f}s)",
            linewidth=2,
        )
        ax.set_title(f"Neuron {neuron_idx} - ISI Distribution")
        ax.set_xlabel("Inter-spike Interval (s)")
        ax.set_ylabel("Probability Mass")
        ax.legend()
        ax.grid(visible=True, alpha=0.3)

    def _plot_entropy_comparison(
        self,
        ax: plt.Axes,
        observed_entropy: float,
        max_entropy: float,
        efficiency: float,
    ) -> None:
        """Plot entropy comparison between observed and theoretical maximum."""
        bar_width = 0.35
        bars = ax.bar(
            [1, 2],
            [observed_entropy, max_entropy],
            bar_width,
            color=["skyblue", "lightcoral"],
        )

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

        ax.set_title("Entropy Comparison")
        ax.set_ylabel("Entropy (bits)")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Observed", "Maximum\n(Exponential)"])
        ax.grid(visible=True, alpha=0.3)

        # Add efficiency text
        ax.text(
            0.5,
            0.95,
            f"Entropy Efficiency: {efficiency:.1%}",
            transform=ax.transAxes,
            ha="center",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )


def main() -> None:
    """Run enhanced Steinmetz spikes analysis."""
    # Configuration
    params = ISIAnalysisParams()
    neuron_idx = 283

    # Initialize components
    isi_analyzer = ISIAnalyzer(params)
    entropy_analyzer = EntropyAnalyzer()
    exp_fitter = ExponentialFitter()
    visualizer = Visualizer()

    # Load and analyze data
    spike_times = DataLoader.load_spike_times()
    isi = isi_analyzer.compute_isis(spike_times[neuron_idx])
    bin_centers, pmf, _ = isi_analyzer.estimate_pmf(isi)
    mean_isi = np.mean(isi)

    # Fit exponential and calculate entropies
    _, fitted_exp = exp_fitter.fit(bin_centers, pmf, mean_isi)
    observed_entropy = entropy_analyzer.shannon_entropy(pmf)
    max_entropy = entropy_analyzer.theoretical_max_entropy(mean_isi, bin_centers)
    efficiency = entropy_analyzer.entropy_efficiency(observed_entropy, max_entropy)

    # Visualize results
    visualizer.plot_analysis(
        bin_centers=bin_centers,
        pmf=pmf,
        mean_isi=mean_isi,
        neuron_idx=neuron_idx,
        fitted_exp=fitted_exp,
        observed_entropy=observed_entropy,
        max_entropy=max_entropy,
        efficiency=efficiency,
    )


if __name__ == "__main__":
    main()
