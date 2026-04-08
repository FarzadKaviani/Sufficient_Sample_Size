"""Plotting helpers for the Bayesian SSS supplementary package."""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayes_sss_core import SeriesResult


def plot_delta_mu_result(result: SeriesResult, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    batches = np.arange(1, len(result.delta_mu_max) + 1, dtype=float)
    ax.plot(batches, result.delta_mu_max, marker="o", linewidth=1.2, label=r"$\Delta\mu_{max}(b)$")
    if len(result.delta_mu_q05) == len(result.delta_mu_q95):
        ax.fill_between(
            batches,
            result.delta_mu_q05,
            result.delta_mu_q95,
            alpha=0.2,
            label="5th–95th percentile envelope",
        )
    ax.axhline(result.tolerance, linestyle="--", linewidth=1.2, label=r"Tolerance $\varepsilon$")
    if result.sufficient_batch is not None:
        ax.axvline(result.sufficient_batch, linestyle="--", linewidth=1.2, label=f"Stop at b={result.sufficient_batch}")
    elif result.predicted_batch is not None:
        ax.axvline(result.predicted_batch, linestyle=":", linewidth=1.2, label=f"Predicted b≈{result.predicted_batch}")

    if result.fit_model and result.fit_params and len(batches) > 0:
        a, b, c = result.fit_params
        if result.fit_model == "exp":
            yhat = a * np.exp(-b * batches) + c
        else:
            yhat = a * (np.maximum(batches, 1e-9) ** (-b)) + c
        ax.plot(batches, yhat, linewidth=1.4, label=f"{result.fit_model} fit")

    ax.set_xlabel("Batch number, b")
    ax.set_ylabel(r"$\Delta\mu$")
    ax.set_title(f"Bayesian sufficient sample size — {result.series}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_monte_carlo_median_plot(
    ax,
    df_plot: pd.DataFrame,
    series_name: str,
    color_limit: float = 10.0,
):
    ax.cla()
    if df_plot.empty:
        ax.set_title(f"Monte Carlo median plot — {series_name} (no data)")
        ax.figure.tight_layout()
        return None

    x = df_plot["batch_size"].to_numpy(dtype=float)
    y = df_plot["median_value"].to_numpy(dtype=float)
    c = df_plot["error_percent"].to_numpy(dtype=float)
    asymptotic_median = float(df_plot["asymptotic_median"].iloc[0])

    scatter = ax.scatter(
        x,
        y,
        c=c,
        cmap="jet_r",
        s=16,
        alpha=0.9,
        linewidths=0,
        vmin=-abs(color_limit),
        vmax=abs(color_limit),
    )
    ax.axhline(
        asymptotic_median,
        color="black",
        linestyle=(0, (4, 6)),
        linewidth=2.0,
        label="Asymptotic median",
    )
    ax.set_xscale("log")
    ax.set_xlim(max(1.9, float(np.min(x) * 0.95)), float(np.max(x) * 1.05))

    ymin = float(np.min(y))
    ymax = float(np.max(y))
    yr = ymax - ymin
    pad = 0.08 * yr if yr > 0 else max(abs(asymptotic_median) * 0.02, 1e-6)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("Sample size, N")
    ax.set_ylabel("Median value")
    ax.set_title(f"Monte Carlo median plot — {series_name}")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower right", frameon=True)
    ax.figure.tight_layout()
    return scatter


def save_median_monte_carlo_plot(
    df_plot: pd.DataFrame,
    series_name: str,
    path: str,
    color_limit: float = 10.0,
) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    scatter = render_monte_carlo_median_plot(ax, df_plot, series_name, color_limit=color_limit)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.12, fraction=0.08)
        cbar.set_label("Error level [%]")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
