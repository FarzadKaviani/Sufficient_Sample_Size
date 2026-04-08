"""Core scientific engine for Bayesian sufficient sample size estimation.

This module refactors the monolithic UI script into a research-oriented, reusable
analysis layer that can be cited as supplementary material for the manuscript:

    Sample Size Effects on Morphological and Particle Size Distribution
    Characteristics of Granular Materials: 2D and 3D Analyses

Authors:
    Farzad Kaviani-Hamedani
    Arman Khoshghalb
    Mahdi Esmailzade
    Nasser Khalili

Affiliation:
    Civil and Environmental Engineering, The University of New South Wales,
    Sydney, Australia

Scientific intent
-----------------
The implementation follows the Bayesian–Monte Carlo workflow described in the
manuscript. The code estimates a sufficient sample size for each selected input
series by repeatedly shuffling observations, assimilating them batch-by-batch,
tracking the change in the posterior location parameter, and applying a stopping
criterion based on a user-defined tolerance sustained over K consecutive batches.

The module deliberately separates scientific computation from UI concerns to
improve reproducibility, testability, and transparency during peer review.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_PAPER_TITLE = (
    "Sample Size Effects on Morphological and Particle Size Distribution "
    "Characteristics of Granular Materials: 2D and 3D Analyses"
)
DEFAULT_AUTHORS = [
    "Farzad Kaviani-Hamedani",
    "Arman Khoshghalb",
    "Mahdi Esmailzade",
    "Nasser Khalili",
]
DEFAULT_AFFILIATION = (
    "Civil and Environmental Engineering, University of New South Wales, "
    "Sydney, Australia"
)


@dataclass(frozen=True)
class BayesianConfig:
    """Configuration for the Bayesian sufficient sample size analysis."""

    model_name: str = "known_var"
    mu0: float = 0.0
    sigma0: float = 1.0
    sigma_obs: float = 1.0
    kappa0: float = 1.0
    alpha0: float = 2.0
    beta0: float = 1.0
    n_sim: int = 100
    batch_size: int = 10
    tolerance: float = 0.02
    consecutive_k: int = 5
    random_seed: Optional[int] = 42
    save_traces: bool = False
    fit_max_multiplier: int = 10
    fit_min_horizon: int = 50
    paper_title: str = DEFAULT_PAPER_TITLE
    authors: Tuple[str, ...] = tuple(DEFAULT_AUTHORS)
    affiliation: str = DEFAULT_AFFILIATION

    def validate(self) -> None:
        if self.model_name not in {"known_var", "invgamma"}:
            raise ValueError("model_name must be 'known_var' or 'invgamma'.")
        if self.n_sim < 1:
            raise ValueError("n_sim must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.consecutive_k < 1:
            raise ValueError("consecutive_k must be at least 1.")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if self.model_name == "known_var":
            if self.sigma0 <= 0 or self.sigma_obs <= 0:
                raise ValueError("sigma0 and sigma_obs must be positive.")
        else:
            if self.kappa0 <= 0 or self.alpha0 <= 0 or self.beta0 <= 0:
                raise ValueError("kappa0, alpha0, and beta0 must be positive.")


@dataclass(frozen=True)
class MonteCarloMedianConfig:
    min_batch: int = 2
    max_batch: int = 500
    n_sizes: int = 70
    draws_per_size: int = 50
    color_limit_percent: float = 10.0
    random_seed: Optional[int] = 42

    def validate(self) -> None:
        if self.min_batch < 1:
            raise ValueError("min_batch must be at least 1.")
        if self.max_batch < self.min_batch:
            raise ValueError("max_batch must be >= min_batch.")
        if self.n_sizes < 2:
            raise ValueError("n_sizes must be at least 2.")
        if self.draws_per_size < 1:
            raise ValueError("draws_per_size must be at least 1.")
        if self.color_limit_percent <= 0:
            raise ValueError("color_limit_percent must be positive.")


@dataclass
class StateKnownVar:
    mu: float
    var_mu: float


@dataclass
class StateInvGamma:
    mu: float
    kappa: float
    alpha: float
    beta: float
    var_mu: Optional[float]


@dataclass
class SeriesResult:
    series: str
    n_observations: int
    model: str
    n_sim: int
    batch_size: int
    tolerance: float
    consecutive_k: int
    sufficient_batch: Optional[int]
    sufficient_samples: Optional[int]
    predicted_batch: Optional[int]
    predicted_samples: Optional[int]
    fit_model: Optional[str]
    fit_rmse: Optional[float]
    fit_params: Optional[Tuple[float, float, float]]
    delta_mu_max: np.ndarray
    delta_mu_q05: np.ndarray
    delta_mu_q50: np.ndarray
    delta_mu_q95: np.ndarray
    all_delta_traces: Optional[List[np.ndarray]] = None

    def to_summary_row(self) -> Dict[str, object]:
        return {
            "series": self.series,
            "n_observations": self.n_observations,
            "model": self.model,
            "n_sim": self.n_sim,
            "batch_size": self.batch_size,
            "tolerance": self.tolerance,
            "consecutive_k": self.consecutive_k,
            "sufficient_batch": self.sufficient_batch,
            "sufficient_samples": self.sufficient_samples,
            "predicted_batch": self.predicted_batch,
            "predicted_samples": self.predicted_samples,
            "fit_model": self.fit_model,
            "fit_rmse": self.fit_rmse,
            "fit_param_a": None if self.fit_params is None else self.fit_params[0],
            "fit_param_b": None if self.fit_params is None else self.fit_params[1],
            "fit_param_c": None if self.fit_params is None else self.fit_params[2],
        }


class NormalKnownVar:
    """Conjugate normal-normal updating for known observation variance."""

    def __init__(self, mu0: float, sigma0: float, sigma_obs: float):
        self.mu = float(mu0)
        self.var_mu = float(max(sigma0, 1e-12) ** 2)
        self.sigma_obs2 = float(max(sigma_obs, 1e-12) ** 2)

    def update(self, n: int, xbar: float) -> Tuple[StateKnownVar, StateKnownVar]:
        prior = StateKnownVar(self.mu, self.var_mu)
        if n <= 0:
            return prior, StateKnownVar(self.mu, self.var_mu)
        inv_post = (1.0 / self.var_mu) + (n / self.sigma_obs2)
        var_post = 1.0 / inv_post
        mu_post = var_post * ((self.mu / self.var_mu) + (n * xbar / self.sigma_obs2))
        self.mu, self.var_mu = float(mu_post), float(var_post)
        return prior, StateKnownVar(self.mu, self.var_mu)


class NormalInvGamma:
    """Conjugate normal-inverse-gamma updating with unknown variance."""

    def __init__(self, mu0: float, kappa0: float, alpha0: float, beta0: float):
        self.mu = float(mu0)
        self.kappa = float(max(kappa0, 1e-12))
        self.alpha = float(max(alpha0, 1e-12))
        self.beta = float(max(beta0, 1e-12))
        self.var_mu = safe_var_from_inv_gamma(self.alpha, self.beta, self.kappa)

    def update(self, n: int, xbar: float, ss: float) -> Tuple[StateInvGamma, StateInvGamma]:
        prior = StateInvGamma(self.mu, self.kappa, self.alpha, self.beta, self.var_mu)
        if n <= 0:
            return prior, StateInvGamma(self.mu, self.kappa, self.alpha, self.beta, self.var_mu)
        kappa_n = self.kappa + n
        mu_n = (self.kappa * self.mu + n * xbar) / kappa_n
        alpha_n = self.alpha + 0.5 * n
        beta_n = self.beta + 0.5 * (ss + (self.kappa * n / kappa_n) * (xbar - self.mu) ** 2)
        self.kappa = float(kappa_n)
        self.mu = float(mu_n)
        self.alpha = float(alpha_n)
        self.beta = float(beta_n)
        self.var_mu = safe_var_from_inv_gamma(self.alpha, self.beta, self.kappa)
        post = StateInvGamma(self.mu, self.kappa, self.alpha, self.beta, self.var_mu)
        return prior, post


def safe_var_from_inv_gamma(alpha: float, beta: float, kappa: float) -> Optional[float]:
    if alpha > 1 and kappa > 0:
        return float(beta / ((alpha - 1.0) * kappa))
    return None


def rolling_k_consecutive(mask: np.ndarray, k: int) -> Optional[int]:
    if k <= 1:
        for i, value in enumerate(mask):
            if bool(value):
                return i
        return None
    count = 0
    start = None
    for i, value in enumerate(mask):
        if bool(value):
            count = count + 1 if count else 1
            if count == 1:
                start = i
            if count >= k:
                return start
        else:
            count = 0
            start = None
    return None


def nanmax_aligned(list_of_arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not list_of_arrays:
        return np.array([], dtype=float)
    length = max(len(arr) for arr in list_of_arrays)
    matrix = np.full((len(list_of_arrays), length), np.nan, dtype=float)
    for i, arr in enumerate(list_of_arrays):
        matrix[i, : len(arr)] = arr
    return np.nanmax(matrix, axis=0)


def nanpercentile_aligned(list_of_arrays: Sequence[np.ndarray], q: float) -> np.ndarray:
    if not list_of_arrays:
        return np.array([], dtype=float)
    length = max(len(arr) for arr in list_of_arrays)
    matrix = np.full((len(list_of_arrays), length), np.nan, dtype=float)
    for i, arr in enumerate(list_of_arrays):
        matrix[i, : len(arr)] = arr
    return np.nanpercentile(matrix, q=q, axis=0)


def batched_stats(arr: np.ndarray, start: int, batch_size: int) -> Tuple[int, float, float]:
    chunk = arr[start : start + batch_size]
    if len(chunk) == 0:
        return 0, 0.0, 0.0
    n = len(chunk)
    xbar = float(np.mean(chunk))
    ss = float(np.sum((chunk - xbar) ** 2))
    return n, xbar, ss


def build_log_batch_sizes(min_n: int, max_n: int, n_points: int) -> np.ndarray:
    min_n = max(2, int(min_n))
    max_n = max(min_n, int(max_n))
    n_points = max(2, int(n_points))
    values = np.unique(np.round(np.geomspace(min_n, max_n, n_points)).astype(int))
    values[0] = min_n
    values[-1] = max_n
    return values


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return cleaned.strip("_") or "series"


def fit_exponential_with_offset(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    if len(x) < 3:
        return (np.nan, np.nan, np.nan, np.inf)
    best = (np.nan, np.nan, np.nan, np.inf)
    for c in [np.percentile(y, p) for p in (0, 5, 10, 15, 20)]:
        z = y - c
        mask = z > 0
        if mask.sum() < 3:
            continue
        xx = x[mask]
        zz = z[mask]
        design = np.vstack([np.ones_like(xx), -xx]).T
        sol, *_ = np.linalg.lstsq(design, np.log(zz), rcond=None)
        ln_a, b = sol[0], sol[1]
        a = float(np.exp(ln_a))
        y_hat = a * np.exp(-b * x) + c
        rmse = float(np.sqrt(np.nanmean((y_hat - y) ** 2)))
        if rmse < best[3]:
            best = (a, float(b), float(c), rmse)
    return best


def fit_powerlaw_with_offset(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    mask_posx = x > 0
    x = x[mask_posx]
    y = y[mask_posx]
    if len(x) < 3:
        return (np.nan, np.nan, np.nan, np.inf)
    best = (np.nan, np.nan, np.nan, np.inf)
    for c in [np.percentile(y, p) for p in (0, 5, 10, 15, 20)]:
        z = y - c
        mask = z > 0
        if mask.sum() < 3:
            continue
        xx = x[mask]
        zz = z[mask]
        design = np.vstack([np.ones_like(xx), -np.log(xx)]).T
        sol, *_ = np.linalg.lstsq(design, np.log(zz), rcond=None)
        ln_a, b = sol[0], sol[1]
        a = float(np.exp(ln_a))
        y_hat = a * (np.maximum(x, 1e-9) ** (-b)) + c
        rmse = float(np.sqrt(np.nanmean((y_hat - y) ** 2)))
        if rmse < best[3]:
            best = (a, float(b), float(c), rmse)
    return best


def predict_batch_for_k_consecutive(
    predictor: Callable[[int], float],
    eps: float,
    k: int,
    max_b: int,
) -> Optional[int]:
    values = np.array([predictor(i) for i in range(1, max_b + 1)], dtype=float)
    return _batch_from_mask(values < eps, k)


def _batch_from_mask(mask: np.ndarray, consecutive_k: int) -> Optional[int]:
    idx0 = rolling_k_consecutive(mask, consecutive_k)
    if idx0 is None:
        return None
    return int(idx0 + 1)


def _make_model(config: BayesianConfig):
    if config.model_name == "known_var":
        return NormalKnownVar(mu0=config.mu0, sigma0=config.sigma0, sigma_obs=config.sigma_obs)
    return NormalInvGamma(
        mu0=config.mu0,
        kappa0=config.kappa0,
        alpha0=config.alpha0,
        beta0=config.beta0,
    )


def run_one_sim(
    values: np.ndarray,
    batch_size: int,
    model_name: str,
    params: Dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    permuted = rng.permutation(values)
    deltas: List[float] = []
    if model_name == "known_var":
        model = NormalKnownVar(
            mu0=params["mu0"],
            sigma0=params["sigma0"],
            sigma_obs=params["sigma_obs"],
        )
    else:
        model = NormalInvGamma(
            mu0=params["mu0"],
            kappa0=params["kappa0"],
            alpha0=params["alpha0"],
            beta0=params["beta0"],
        )

    i = 0
    while i < len(permuted):
        n, xbar, ss = batched_stats(permuted, i, batch_size)
        if model_name == "known_var":
            prior, post = model.update(n=n, xbar=xbar)
        else:
            prior, post = model.update(n=n, xbar=xbar, ss=ss)
        deltas.append(abs(post.mu - prior.mu))
        i += batch_size
    return np.asarray(deltas, dtype=float)


def monte_carlo_batch_medians(
    values: np.ndarray,
    batch_sizes: np.ndarray,
    draws_per_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    asymptotic_median = float(np.median(values))
    n_total = len(values)
    for batch_size in batch_sizes:
        if batch_size < 1 or batch_size > n_total:
            continue
        for draw_id in range(1, draws_per_size + 1):
            sample = rng.choice(values, size=int(batch_size), replace=False)
            median_value = float(np.median(sample))
            if abs(asymptotic_median) > 1e-12:
                err_pct = 100.0 * (median_value - asymptotic_median) / asymptotic_median
            else:
                err_pct = 100.0 * (median_value - asymptotic_median)
            records.append(
                {
                    "batch_size": int(batch_size),
                    "draw_id": int(draw_id),
                    "median_value": median_value,
                    "asymptotic_median": asymptotic_median,
                    "error_percent": err_pct,
                    "abs_error_percent": abs(err_pct),
                }
            )
    return pd.DataFrame.from_records(records)


def _fit_and_predict(delta_mu_max: np.ndarray, config: BayesianConfig) -> Tuple[Optional[str], Optional[Tuple[float, float, float]], Optional[float], Optional[int]]:
    if len(delta_mu_max) < 3:
        return None, None, None, None

    x = np.arange(1, len(delta_mu_max) + 1, dtype=float)
    exp_params = fit_exponential_with_offset(x, delta_mu_max)
    pow_params = fit_powerlaw_with_offset(x, delta_mu_max)
    if exp_params[3] <= pow_params[3]:
        fit_kind = "exp"
        fit_params = exp_params[:3]
        fit_rmse = exp_params[3]
    else:
        fit_kind = "pow"
        fit_params = pow_params[:3]
        fit_rmse = pow_params[3]

    if not all(np.isfinite(fit_params)):
        return None, None, None, None

    if fit_kind == "exp":
        a, b, c = fit_params
        predictor = lambda t: a * math.exp(-b * t) + c
    else:
        a, b, c = fit_params
        predictor = lambda t: a * (max(t, 1e-9) ** (-b)) + c

    max_b = int(max(config.fit_min_horizon, len(delta_mu_max) * config.fit_max_multiplier))
    predicted_batch = predict_batch_for_k_consecutive(
        predictor=predictor,
        eps=config.tolerance,
        k=config.consecutive_k,
        max_b=max_b,
    )
    return fit_kind, tuple(float(v) for v in fit_params), float(fit_rmse), predicted_batch


def analyze_series(
    series_name: str,
    values: Sequence[float],
    config: BayesianConfig,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> SeriesResult:
    config.validate()
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(array) < 2:
        raise ValueError(f"Series '{series_name}' has fewer than two numeric observations.")

    rng = np.random.default_rng(config.random_seed)
    params: Dict[str, float] = {"mu0": config.mu0}
    if config.model_name == "known_var":
        params.update({"sigma0": config.sigma0, "sigma_obs": config.sigma_obs})
    else:
        params.update({"kappa0": config.kappa0, "alpha0": config.alpha0, "beta0": config.beta0})

    traces: List[np.ndarray] = []
    sufficient_batch: Optional[int] = None
    predicted_batch: Optional[int] = None
    fit_model: Optional[str] = None
    fit_params: Optional[Tuple[float, float, float]] = None
    fit_rmse: Optional[float] = None
    delta_mu_max = np.array([], dtype=float)

    for sim_index in range(config.n_sim):
        deltas = run_one_sim(array, config.batch_size, config.model_name, params, rng)
        traces.append(deltas)

        delta_mu_max = nanmax_aligned(traces)
        sufficient_batch = _batch_from_mask(delta_mu_max < config.tolerance, config.consecutive_k)
        if sufficient_batch is None:
            fit_model, fit_params, fit_rmse, predicted_batch = _fit_and_predict(delta_mu_max, config)
        else:
            predicted_batch = None
            fit_model = None
            fit_params = None
            fit_rmse = None

        if progress_callback is not None:
            progress_callback(
                {
                    "series": series_name,
                    "simulation_index": sim_index + 1,
                    "n_sim": config.n_sim,
                    "delta_mu_max": delta_mu_max.copy(),
                    "sufficient_batch": sufficient_batch,
                    "predicted_batch": predicted_batch,
                    "fit_model": fit_model,
                    "fit_params": fit_params,
                    "fit_rmse": fit_rmse,
                }
            )

    q05 = nanpercentile_aligned(traces, 5)
    q50 = nanpercentile_aligned(traces, 50)
    q95 = nanpercentile_aligned(traces, 95)

    return SeriesResult(
        series=series_name,
        n_observations=int(len(array)),
        model=config.model_name,
        n_sim=config.n_sim,
        batch_size=config.batch_size,
        tolerance=config.tolerance,
        consecutive_k=config.consecutive_k,
        sufficient_batch=sufficient_batch,
        sufficient_samples=None if sufficient_batch is None else int(sufficient_batch * config.batch_size),
        predicted_batch=predicted_batch,
        predicted_samples=None if predicted_batch is None else int(predicted_batch * config.batch_size),
        fit_model=fit_model,
        fit_rmse=fit_rmse,
        fit_params=fit_params,
        delta_mu_max=delta_mu_max,
        delta_mu_q05=q05,
        delta_mu_q50=q50,
        delta_mu_q95=q95,
        all_delta_traces=traces if config.save_traces else None,
    )


def analyze_dataframe(
    dataframe: pd.DataFrame,
    selected_columns: Sequence[str],
    config: BayesianConfig,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> List[SeriesResult]:
    results: List[SeriesResult] = []
    for col_index, column in enumerate(selected_columns, start=1):
        if column not in dataframe.columns:
            continue
        values = pd.to_numeric(dataframe[column], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) < 2:
            continue

        series_callback = None
        if progress_callback is not None:
            def series_callback(payload: Dict[str, object], *, _col_index=col_index, _n_cols=len(selected_columns)) -> None:
                payload = dict(payload)
                payload["column_index"] = _col_index
                payload["n_columns"] = _n_cols
                progress_callback(payload)

        result = analyze_series(column, values, config, progress_callback=series_callback)
        results.append(result)
    return results


def save_series_result(result: SeriesResult, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stem = sanitize_name(result.series)
    paths: Dict[str, str] = {}

    envelope_df = pd.DataFrame(
        {
            "batch": np.arange(1, len(result.delta_mu_max) + 1, dtype=int),
            "delta_mu_max": result.delta_mu_max,
            "delta_mu_q05": result.delta_mu_q05,
            "delta_mu_q50": result.delta_mu_q50,
            "delta_mu_q95": result.delta_mu_q95,
        }
    )
    envelope_path = os.path.join(out_dir, f"{stem}_delta_mu_envelope.csv")
    envelope_df.to_csv(envelope_path, index=False)
    paths["delta_mu_envelope_csv"] = envelope_path

    if result.all_delta_traces is not None:
        trace_path = os.path.join(out_dir, f"{stem}_delta_mu_traces.csv")
        max_len = max(len(arr) for arr in result.all_delta_traces)
        trace_df = pd.DataFrame(
            {
                f"sim_{i+1}": np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
                for i, arr in enumerate(result.all_delta_traces)
            }
        )
        trace_df.insert(0, "batch", np.arange(1, max_len + 1, dtype=int))
        trace_df.to_csv(trace_path, index=False)
        paths["delta_mu_traces_csv"] = trace_path

    return paths


def save_summary(results: Sequence[SeriesResult], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "sufficient_sample_summary.csv")
    pd.DataFrame([r.to_summary_row() for r in results]).to_csv(summary_path, index=False)
    return summary_path


def save_run_metadata(
    config: BayesianConfig,
    selected_columns: Sequence[str],
    input_path: Optional[str],
    out_dir: str,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    metadata = asdict(config)
    metadata["selected_columns"] = list(selected_columns)
    metadata["input_path"] = input_path
    metadata["software_note"] = (
        "Research supplementary implementation for Bayesian sufficient sample "
        "size estimation and Monte Carlo median analysis."
    )
    metadata_path = os.path.join(out_dir, "run_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return metadata_path


def run_median_monte_carlo(
    dataframe: pd.DataFrame,
    selected_columns: Sequence[str],
    config: MonteCarloMedianConfig,
) -> Dict[str, pd.DataFrame]:
    config.validate()
    rng = np.random.default_rng(config.random_seed)
    outputs: Dict[str, pd.DataFrame] = {}
    for column in selected_columns:
        if column not in dataframe.columns:
            continue
        values = pd.to_numeric(dataframe[column], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) < 2:
            continue
        max_batch = min(config.max_batch, len(values))
        min_batch = min(config.min_batch, max_batch)
        batch_sizes = build_log_batch_sizes(min_batch, max_batch, config.n_sizes)
        outputs[column] = monte_carlo_batch_medians(values, batch_sizes, config.draws_per_size, rng)
    return outputs
