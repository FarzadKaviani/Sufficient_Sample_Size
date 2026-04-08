"""Command-line entry point for the Bayesian SSS supplementary package."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import pandas as pd

from bayes_sss_core import (
    BayesianConfig,
    MonteCarloMedianConfig,
    analyze_dataframe,
    run_median_monte_carlo,
    save_run_metadata,
    save_series_result,
    save_summary,
)
from bayes_sss_plots import plot_delta_mu_result, save_median_monte_carlo_plot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bayesian sufficient sample size analysis for granular-material descriptors."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--columns",
        nargs="+",
        required=True,
        help="One or more CSV column names to analyse.",
    )
    parser.add_argument(
        "--mode",
        choices=["bayes", "median-mc", "both"],
        default="both",
        help="Analysis mode.",
    )
    parser.add_argument("--model", choices=["known_var", "invgamma"], default="known_var")
    parser.add_argument("--mu0", type=float, default=0.0)
    parser.add_argument("--sigma0", type=float, default=1.0)
    parser.add_argument("--sigma-obs", dest="sigma_obs", type=float, default=1.0)
    parser.add_argument("--kappa0", type=float, default=1.0)
    parser.add_argument("--alpha0", type=float, default=2.0)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--n-sim", dest="n_sim", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--consecutive-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--mc-min-batch", type=int, default=2)
    parser.add_argument("--mc-max-batch", type=int, default=500)
    parser.add_argument("--mc-n-sizes", type=int, default=70)
    parser.add_argument("--mc-draws", type=int, default=50)
    parser.add_argument("--mc-color-limit", type=float, default=10.0)
    return parser


def _print_progress(payload) -> None:
    series = payload["series"]
    sim_idx = payload["simulation_index"]
    n_sim = payload["n_sim"]
    suff = payload.get("sufficient_batch")
    pred = payload.get("predicted_batch")
    suffix = f" stop_b={suff}" if suff is not None else (f" pred_b≈{pred}" if pred is not None else "")
    print(f"[{series}] simulation {sim_idx}/{n_sim}{suffix}")


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input)
    selected_columns = [col for col in args.columns if col in df.columns]
    if not selected_columns:
        parser.error("None of the requested columns were found in the CSV file.")

    os.makedirs(args.out, exist_ok=True)

    if args.mode in {"bayes", "both"}:
        bayes_config = BayesianConfig(
            model_name=args.model,
            mu0=args.mu0,
            sigma0=args.sigma0,
            sigma_obs=args.sigma_obs,
            kappa0=args.kappa0,
            alpha0=args.alpha0,
            beta0=args.beta0,
            n_sim=args.n_sim,
            batch_size=args.batch_size,
            tolerance=args.tolerance,
            consecutive_k=args.consecutive_k,
            random_seed=args.seed,
            save_traces=args.save_traces,
        )
        results = analyze_dataframe(df, selected_columns, bayes_config, progress_callback=_print_progress)
        bayes_dir = os.path.join(args.out, "bayesian_sss")
        os.makedirs(bayes_dir, exist_ok=True)
        save_summary(results, bayes_dir)
        save_run_metadata(bayes_config, selected_columns, args.input, bayes_dir)
        for result in results:
            save_series_result(result, bayes_dir)
            plot_delta_mu_result(result, os.path.join(bayes_dir, f"{result.series.replace(' ', '_')}_delta_mu_plot.png"))
        print(f"Bayesian SSS outputs saved to: {bayes_dir}")

    if args.mode in {"median-mc", "both"}:
        mc_config = MonteCarloMedianConfig(
            min_batch=args.mc_min_batch,
            max_batch=args.mc_max_batch,
            n_sizes=args.mc_n_sizes,
            draws_per_size=args.mc_draws,
            color_limit_percent=args.mc_color_limit,
            random_seed=args.seed,
        )
        outputs = run_median_monte_carlo(df, selected_columns, mc_config)
        mc_dir = os.path.join(args.out, "median_monte_carlo")
        os.makedirs(mc_dir, exist_ok=True)
        summary_rows = []
        for series, df_plot in outputs.items():
            stem = series.replace(" ", "_")
            csv_path = os.path.join(mc_dir, f"{stem}_median_mc_points.csv")
            png_path = os.path.join(mc_dir, f"{stem}_median_mc_plot.png")
            df_plot.to_csv(csv_path, index=False)
            save_median_monte_carlo_plot(df_plot, series, png_path, color_limit=args.mc_color_limit)
            summary_rows.append(
                {
                    "series": series,
                    "n_points": len(df_plot),
                    "csv_path": csv_path,
                    "png_path": png_path,
                }
            )
        pd.DataFrame(summary_rows).to_csv(os.path.join(mc_dir, "median_mc_summary.csv"), index=False)
        print(f"Median Monte Carlo outputs saved to: {mc_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
