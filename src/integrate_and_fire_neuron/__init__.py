"""Implementation of Linear and Leaky Integrate-and-Fire Neuron Models.

This module provides implementations of:
1. Linear Integrate-and-Fire (LIF) neuron model
2. Leaky Integrate-and-Fire neuron model with inhibitory inputs
3. Visualization utilities for analyzing neuron behavior

The models simulate basic neural dynamics including:
- Membrane potential integration
- Spike generation
- Post-spike reset
- Leaky current (for leaky I&F)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


@dataclass
class NeuronParams:
    """Parameters for neuron models."""

    n_steps: int = 1000  # Number of simulation steps
    alpha: float = 0.01  # Integration constant
    beta: float = 0.1  # Leak constant
    exc_rate: float = 10  # Rate of excitatory inputs
    inh_rate: float = 10  # Rate of inhibitory inputs
    seed: int | None = None  # Random seed for reproducibility


class NeuronSimulation:
    """Base class for neuron simulations."""

    def __init__(self, params: NeuronParams) -> None:
        """Initialize neuron simulation.

        Args:
            params: Configuration parameters for the simulation

        """
        self.params = params
        self.rng = np.random.default_rng(params.seed)
        self.voltage: np.ndarray = np.array([])
        self.spike_times: list[int] = []

    def reset(self) -> None:
        """Reset simulation state."""
        self.voltage = np.zeros(self.params.n_steps)
        self.spike_times = []

    def simulate(self) -> tuple[np.ndarray, list[int]]:
        """Run simulation and return results.

        Returns:
            tuple containing:
                - voltage: Array of membrane potential values
                - spike_times: list of spike time indices

        """
        msg = "Simulate method must be implemented by subclass"
        raise NotImplementedError(msg)

    def _check_spike(self, t: int, v: float) -> float:
        """Check for spike and handle reset.

        Args:
            t: Current time step
            v: Current voltage

        Returns:
            float: Updated voltage (0 if spike occurred, unchanged otherwise)

        """
        if v > 1:  # Threshold crossing
            self.spike_times.append(t)
            return 0
        return v


class LinearIFNeuron(NeuronSimulation):
    """Linear Integrate-and-Fire neuron model."""

    def simulate(self) -> tuple[np.ndarray, list[int]]:
        """Simulate Linear IF neuron dynamics.

        Returns:
            tuple containing:
                - voltage: Array of membrane potential values
                - spike_times: list of spike time indices

        """
        self.reset()
        excitatory_spikes = stats.poisson(self.params.exc_rate).rvs(self.params.n_steps)

        for t in range(1, self.params.n_steps):
            # Update voltage
            v_new = self.voltage[t - 1] + self.params.alpha * excitatory_spikes[t]

            # Check for spike and update
            self.voltage[t] = self._check_spike(t, v_new)

        return self.voltage, self.spike_times


class LeakyIFNeuron(NeuronSimulation):
    """Leaky Integrate-and-Fire neuron model with inhibitory inputs."""

    def simulate(self) -> tuple[np.ndarray, list[int]]:
        """Simulate Leaky IF neuron dynamics.

        Returns:
            tuple containing:
                - voltage: Array of membrane potential values
                - spike_times: list of spike time indices

        """
        self.reset()
        exc_spikes = stats.poisson(self.params.exc_rate).rvs(self.params.n_steps)
        inh_spikes = stats.poisson(self.params.inh_rate).rvs(self.params.n_steps)

        for t in range(1, self.params.n_steps):
            # Update voltage with leak term and input currents
            v_new = self.voltage[t - 1] * (1 - self.params.beta) + self.params.alpha * (
                exc_spikes[t] - inh_spikes[t]
            )

            # Check for spike and update
            self.voltage[t] = self._check_spike(t, v_new)

        return self.voltage, self.spike_times


class NeuronVisualizer:
    """Handles visualization of neuron simulation results."""

    @staticmethod
    def plot_histogram(
        counts: np.ndarray,
        bins: np.ndarray,
        vlines: tuple = (),
        ax: plt.Axes | None = None,
        ax_args: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> plt.Axes:
        """Plot step histogram of spike intervals.

        Args:
            counts: Histogram counts
            bins: Bin edges
            vlines: Vertical lines to add (e.g., mean ISI)
            ax: Matplotlib axes (created if None)
            ax_args: Arguments for axes configuration
            **kwargs: Additional arguments for plotting

        Returns:
            plt.Axes: The plot axes

        """
        if ax is None:
            _, ax = plt.subplots()

        # Adjust counts for step plot
        counts = np.insert(counts, 0, counts[0])

        # Create histogram
        ax.fill_between(bins, counts, step="pre", alpha=0.4, **kwargs)
        ax.plot(bins, counts, drawstyle="steps", **kwargs)

        # Add vertical lines (e.g., for mean)
        for x in vlines:
            ax.axvline(x, color="r", linestyle="dotted")

        # Configure axes
        if ax_args is None:
            ax_args = {}

        ymin, ymax = ax_args.get("ylim", [None, None])
        if ymax is None:
            ymax = np.max(counts) * (
                1.5 if ax_args.get("yscale", "linear") == "log" else 1.1
            )
            if ymin is None:
                ymin = 0
        ax_args["ylim"] = [ymin, ymax]

        ax.set(**ax_args)
        ax.autoscale(enable=False, axis="x", tight=True)

        return ax

    def plot_simulation_results(
        self,
        voltage: np.ndarray,
        spike_times: list[int],
        title: str = "",
    ) -> None:
        """Plot neuron membrane potential and inter-spike intervals.

        Args:
            voltage: Array of membrane potential values
            spike_times: list of spike time indices
            title: Title for the figure

        """
        _, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

        # Plot membrane potential
        self._plot_voltage(ax1, voltage, spike_times)

        # Plot ISI histogram if there are multiple spikes
        self._plot_isi(ax2, spike_times)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def _plot_voltage(
        self,
        ax: plt.Axes,
        voltage: np.ndarray,
        spike_times: list[int],
        max_voltage: int = 100,
    ) -> None:
        """Plot membrane potential trajectory."""
        ax.plot(voltage[:max_voltage])
        ax.set(xlabel="Time", ylabel="Voltage")

        for t in spike_times:
            if t >= max_voltage:
                break
            ax.axvline(t, color="red")

    def _plot_isi(self, ax: plt.Axes, spike_times: list[int]) -> None:
        """Plot inter-spike interval distribution."""
        if len(spike_times) > 1:
            isi = np.diff(spike_times)
            counts, bins = np.histogram(isi, np.arange(isi.min(), isi.max() + 2) - 0.5)
            self.plot_histogram(
                counts,
                bins,
                vlines=[np.mean(isi)],
                ax=ax,
                ax_args={
                    "xlabel": "Inter-spike interval",
                    "ylabel": "Number of intervals",
                    "xlim": [0, max(20, int(bins[-1]) + 5)],
                },
            )
        else:
            ax.set(xlabel="Inter-spike interval", ylabel="Number of intervals")


def main() -> None:
    """Run example simulations of both neuron models."""
    # Configuration
    params = NeuronParams(n_steps=1000, alpha=0.01, exc_rate=10, seed=12)

    # Initialize components
    visualizer = NeuronVisualizer()

    # Simulate and plot Linear IF neuron
    lif_neuron = LinearIFNeuron(params)
    v, spikes = lif_neuron.simulate()
    visualizer.plot_simulation_results(v, spikes, "Linear Integrate-and-Fire Neuron")

    # Update parameters for Leaky IF neuron
    leaky_params = NeuronParams(
        n_steps=1000,
        alpha=0.5,
        beta=0.1,
        exc_rate=10,
        inh_rate=10,
        seed=12,
    )

    # Simulate and plot Leaky IF neuron
    lif_inh_neuron = LeakyIFNeuron(leaky_params)
    v, spikes = lif_inh_neuron.simulate()
    visualizer.plot_simulation_results(
        v,
        spikes,
        "Leaky Integrate-and-Fire Neuron with Inhibition",
    )


if __name__ == "__main__":
    main()
