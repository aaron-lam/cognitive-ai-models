"""Analysis of neural spike timing patterns and inter-spike intervals.

This module provides functionality for:
1. Loading and preprocessing spike time data
2. Computing spike statistics across neurons
3. Analyzing inter-spike intervals
4. Visualizing spike patterns and distributions

The analysis includes spike count distributions, raster plots, and ISI histograms.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import requests


@dataclass
class AnalysisConfig:
    """Configuration parameters for spike analysis."""

    data_url: str = "https://osf.io/sy5xt/download"
    neuron_step: int = 5  # Step size for neuron subset selection
    n_bins: int = 50  # Number of bins for histograms
    default_neuron: int = 283  # Default neuron index for single-neuron analysis
    alpha: float = 0.6  # Transparency for plots


class DataLoader:
    """Handles loading and preprocessing of spike time data."""

    STATUS_OK = 200

    @classmethod
    def load_spike_times(cls, url: str) -> np.ndarray:
        """Fetch and load spike time dataset.

        Args:
            url: URL to fetch the data from

        Returns:
            np.ndarray: Array of spike times for each neuron

        Raises:
            RuntimeError: If data fetch fails

        """
        response = requests.get(url, timeout=10)
        if response.status_code != cls.STATUS_OK:
            msg = f"Failed to fetch data. Status code: {response.status_code}"
            raise RuntimeError(msg)

        return np.load(io.BytesIO(response.content), allow_pickle=True)["spike_times"]

    @staticmethod
    def restrict_time_window(
        spike_times: np.ndarray,
        interval: tuple[float, float],
    ) -> np.ndarray:
        """Filter spike times to specified time window.

        Args:
            spike_times: Array of spike times for each neuron
            interval: (start_time, end_time) tuple

        Returns:
            np.ndarray: Filtered spike times

        """
        return np.array(
            [
                spikes[(spikes >= interval[0]) & (spikes < interval[1])]
                for spikes in spike_times
            ],
            dtype=object,
        )


class SpikeAnalyzer:
    """Analyzes spike timing patterns and computes statistics."""

    @staticmethod
    def compute_population_statistics(
        spike_times: np.ndarray,
    ) -> tuple[float, float, list[int]]:
        """Compute population-level spike statistics.

        Args:
            spike_times: Array of spike times for each neuron

        Returns:
            tuple containing:
                - mean_spikes: Mean spike count per neuron
                - median_spikes: Median spike count per neuron
                - total_spikes: list of spike counts for each neuron

        """
        total_spikes = [len(neuron_spikes) for neuron_spikes in spike_times]
        return np.mean(total_spikes), np.median(total_spikes), total_spikes

    @staticmethod
    def compute_isis(spike_times: np.ndarray, neuron_idx: int) -> np.ndarray:
        """Compute inter-spike intervals for a specific neuron.

        Args:
            spike_times: Array of spike times for each neuron
            neuron_idx: Index of neuron to analyze

        Returns:
            np.ndarray: Array of inter-spike intervals

        """
        return np.diff(spike_times[neuron_idx])


class Visualizer:
    """Handles visualization of spike analysis results."""

    def __init__(self, config: AnalysisConfig) -> None:
        """Initialize visualizer with configuration.

        Args:
            config: Configuration parameters

        """
        self.config = config

    def plot_spike_distribution(
        self,
        spike_counts: list[int],
        mean_spikes: float,
        median_spikes: float,
    ) -> None:
        """Plot distribution of spike counts across neurons.

        Args:
            spike_counts: list of spike counts per neuron
            mean_spikes: Mean spike count
            median_spikes: Median spike count

        """
        plt.figure(figsize=(10, 6))
        plt.hist(
            spike_counts,
            bins=self.config.n_bins,
            histtype="stepfilled",
            alpha=self.config.alpha,
        )

        # Add reference lines
        plt.axvline(
            median_spikes,
            color="limegreen",
            linestyle="--",
            label="Median neuron",
        )
        plt.axvline(mean_spikes, color="orange", linestyle="--", label="Mean neuron")

        plt.xlabel("Total spikes per neuron")
        plt.ylabel("Number of neurons")
        plt.legend()
        plt.title("Distribution of Total Spikes per Neuron")
        plt.show()

    def plot_raster(
        self,
        spike_times: np.ndarray,
        neuron_step: int | None = None,
    ) -> None:
        """Create raster plot for subset of neurons.

        Args:
            spike_times: Array of spike times for each neuron
            neuron_step: Step size for selecting neurons (default from config)

        """
        if neuron_step is None:
            neuron_step = self.config.neuron_step

        plt.figure(figsize=(12, 6))
        neuron_indices = np.arange(0, len(spike_times), neuron_step)
        plt.eventplot(spike_times[neuron_indices], color=".2")
        plt.xlabel("Time (s)")
        plt.yticks([])
        plt.title("Spike Raster Plot (Subset of Neurons)")
        plt.show()

    def plot_isi_distribution(self, isis: np.ndarray) -> None:
        """Plot inter-spike interval distribution.

        Args:
            isis: Array of inter-spike intervals

        """
        plt.figure(figsize=(10, 6))
        plt.hist(
            isis,
            bins=self.config.n_bins,
            histtype="stepfilled",
            alpha=self.config.alpha,
        )

        # Add mean ISI line
        plt.axvline(isis.mean(), color="orange", linestyle="--", label="Mean ISI")

        plt.xlabel("ISI duration (s)")
        plt.ylabel("Number of spikes")
        plt.legend()
        plt.title("Inter-Spike Interval (ISI) Distribution")
        plt.show()


class SpikeAnalysisRunner:
    """Orchestrates the spike analysis pipeline."""

    def __init__(self, config: AnalysisConfig) -> None:
        """Initialize analysis runner.

        Args:
            config: Analysis configuration parameters

        """
        self.config = config
        self.visualizer = Visualizer(config)

    def run_analysis(
        self,
        time_window: tuple[float, float] | None = None,
        neuron_idx: int | None = None,
    ) -> None:
        """Run complete spike analysis pipeline.

        Args:
            time_window: Optional time window for analysis
            neuron_idx: Optional specific neuron to analyze

        """
        # Load data
        spike_times = DataLoader.load_spike_times(self.config.data_url)

        # Compute population statistics
        mean_spikes, median_spikes, spike_counts = (
            SpikeAnalyzer.compute_population_statistics(spike_times)
        )

        # Plot spike count distribution
        self.visualizer.plot_spike_distribution(
            spike_counts,
            mean_spikes,
            median_spikes,
        )

        # Handle time window restriction if specified
        if time_window is not None:
            limited_spike_times = DataLoader.restrict_time_window(
                spike_times,
                time_window,
            )
            # Plot raster
            self.visualizer.plot_raster(limited_spike_times)

        # Analyze specific neuron
        if neuron_idx is None:
            neuron_idx = self.config.default_neuron

        isis = SpikeAnalyzer.compute_isis(spike_times, neuron_idx)
        self.visualizer.plot_isi_distribution(isis)


def main() -> None:
    """Run example spike analysis."""
    # Configuration
    config = AnalysisConfig()

    # Initialize and run analysis
    runner = SpikeAnalysisRunner(config)
    runner.run_analysis(
        time_window=(5, 11),
        neuron_idx=283,
    )


if __name__ == "__main__":
    main()
