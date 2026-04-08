"""Tkinter desktop UI for the journal-ready Bayesian SSS package."""

from __future__ import annotations

import json
import math
import os
import queue
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from bayes_sss_core import (
    BayesianConfig,
    MonteCarloMedianConfig,
    NormalInvGamma,
    NormalKnownVar,
    SeriesResult,
    run_median_monte_carlo,
    save_run_metadata,
    save_series_result,
    save_summary,
)
from bayes_sss_plots import plot_delta_mu_result, render_monte_carlo_median_plot


def _rolling_k_consecutive(mask: np.ndarray, k: int):
    if k <= 1:
        idx = np.flatnonzero(mask)
        return int(idx[0] + 1) if idx.size else None
    count = 0
    start = None
    for i, v in enumerate(mask):
        if bool(v):
            count = count + 1 if count else 1
            if count == 1:
                start = i
            if count >= k:
                return int(start + 1)
        else:
            count = 0
            start = None
    return None


def _nanmax_aligned(arrays):
    if not arrays:
        return np.array([], dtype=float)
    L = max(len(a) for a in arrays)
    M = np.full((len(arrays), L), np.nan, dtype=float)
    for i, arr in enumerate(arrays):
        M[i, :len(arr)] = arr
    return np.nanmax(M, axis=0)


def _nanpercentile_aligned(arrays, q):
    if not arrays:
        return np.array([], dtype=float)
    L = max(len(a) for a in arrays)
    M = np.full((len(arrays), L), np.nan, dtype=float)
    for i, arr in enumerate(arrays):
        M[i, :len(arr)] = arr
    return np.nanpercentile(M, q, axis=0)


def _fit_exponential_with_offset(x, y):
    if len(x) < 3:
        return (np.nan, np.nan, np.nan, np.inf)
    c_candidates = [np.percentile(y, p) for p in (0, 5, 10, 15, 20)]
    best = (np.nan, np.nan, np.nan, np.inf)
    for c in c_candidates:
        z = y - c
        if np.any(z <= 0):
            mask = z > 0
            if mask.sum() < 3:
                continue
            xx = x[mask]
            zz = z[mask]
        else:
            xx, zz = x, z
        Y = np.log(zz)
        A = np.vstack([np.ones_like(xx), -xx]).T
        sol, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        ln_a, b = sol[0], sol[1]
        a = float(np.exp(ln_a))
        y_hat = a * np.exp(-b * x) + c
        rmse = float(np.sqrt(np.nanmean((y_hat - y) ** 2)))
        if rmse < best[3]:
            best = (a, b, c, rmse)
    return best


def _fit_powerlaw_with_offset(x, y):
    if len(x) < 3:
        return (np.nan, np.nan, np.nan, np.inf)
    mask_pos = x > 0
    x = x[mask_pos]
    y = y[mask_pos]
    if len(x) < 3:
        return (np.nan, np.nan, np.nan, np.inf)
    c_candidates = [np.percentile(y, p) for p in (0, 5, 10, 15, 20)]
    best = (np.nan, np.nan, np.nan, np.inf)
    for c in c_candidates:
        z = y - c
        mask = z > 0
        if mask.sum() < 3:
            continue
        xx = x[mask]
        zz = z[mask]
        Y = np.log(zz)
        X = np.log(xx)
        A = np.vstack([np.ones_like(X), -X]).T
        sol, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        ln_a, b = sol[0], sol[1]
        a = float(np.exp(ln_a))
        y_hat = a * (np.maximum(x, 1e-9) ** (-b)) + c
        rmse = float(np.sqrt(np.nanmean((y_hat - y) ** 2)))
        if rmse < best[3]:
            best = (a, b, c, rmse)
    return best


def _predict_batch_for_k_consecutive(f, eps, k, max_b):
    vals = np.array([f(i) for i in range(1, max_b + 1)], dtype=float)
    return _rolling_k_consecutive(vals < eps, k)


def _fit_and_predict(delta_mu_max, config):
    """Predict the required batch using a power-law-with-offset fit.

    If the fitted curve does not cross the tolerance within the initial search
    horizon, this function falls back to an analytical estimate for the first
    crossing and then expands that into a K-consecutive-batch prediction.
    """
    if len(delta_mu_max) < 3:
        return None, None, None, None
    x = np.arange(1, len(delta_mu_max) + 1, dtype=float)
    pow_params = _fit_powerlaw_with_offset(x, delta_mu_max)
    fit_model = 'pow'
    fit_params = pow_params[:3]
    fit_rmse = pow_params[3]
    if not all(np.isfinite(fit_params)):
        return None, None, None, None
    a, b, c = fit_params
    if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c) or b <= 0:
        return fit_model, tuple(float(v) for v in fit_params), float(fit_rmse), None

    f = lambda t: a * (max(t, 1e-9) ** (-b)) + c
    max_b = int(max(config.fit_min_horizon, len(delta_mu_max) * config.fit_max_multiplier))
    pred_b = _predict_batch_for_k_consecutive(f, config.tolerance, config.consecutive_k, max_b)

    if pred_b is None and c < config.tolerance and a > 0:
        # Solve a*t^(-b) + c = tolerance for the first crossing.
        rhs = config.tolerance - c
        if rhs > 0:
            first_cross = int(math.ceil((a / rhs) ** (1.0 / b)))
            pred_b = max(1, first_cross)
            # Require K consecutive batches below tolerance.
            pred_b = pred_b + max(int(config.consecutive_k) - 1, 0)

    if pred_b is not None:
        pred_b = int(pred_b)
    return fit_model, tuple(float(v) for v in fit_params), float(fit_rmse), pred_b


def _run_one_sim_with_paths(values, config, rng):
    permuted = rng.permutation(values)
    model = (
        NormalKnownVar(mu0=config.mu0, sigma0=config.sigma0, sigma_obs=config.sigma_obs)
        if config.model_name == 'known_var'
        else NormalInvGamma(mu0=config.mu0, kappa0=config.kappa0, alpha0=config.alpha0, beta0=config.beta0)
    )
    delta_trace = []
    prior_mu_path = []
    prior_var_path = []
    post_mu_path = []
    post_var_path = []
    batches = []
    i = 0
    batch_idx = 0
    while i < len(permuted):
        chunk = permuted[i:i + config.batch_size]
        if len(chunk) == 0:
            break
        n = len(chunk)
        xbar = float(np.mean(chunk))
        ss = float(np.sum((chunk - xbar) ** 2))
        if config.model_name == 'known_var':
            prior, post = model.update(n=n, xbar=xbar)
            prior_var = float(prior.var_mu)
            post_var = float(post.var_mu)
        else:
            prior, post = model.update(n=n, xbar=xbar, ss=ss)
            prior_var = float(prior.var_mu) if prior.var_mu is not None else np.nan
            post_var = float(post.var_mu) if post.var_mu is not None else np.nan
        batch_idx += 1
        batches.append(batch_idx)
        prior_mu_path.append(float(prior.mu))
        prior_var_path.append(prior_var)
        post_mu_path.append(float(post.mu))
        post_var_path.append(post_var)
        delta_trace.append(abs(float(post.mu) - float(prior.mu)))
        i += config.batch_size
    return {
        'delta_trace': np.asarray(delta_trace, dtype=float),
        'batches': np.asarray(batches, dtype=float),
        'prior_mu_path': np.asarray(prior_mu_path, dtype=float),
        'prior_var_path': np.asarray(prior_var_path, dtype=float),
        'post_mu_path': np.asarray(post_mu_path, dtype=float),
        'post_var_path': np.asarray(post_var_path, dtype=float),
    }



class BayesSSSApp(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.title("Bayesian SSS — Journal Supplementary App")
        self.master.geometry("1260x860")
        self.master.minsize(1140, 760)
        self._resize_after_id = None

        self.msg_queue = queue.Queue()
        self.running_thread = None
        self.current_mc_cbar = None
        self.current_bayes_result = None
        self.current_df = None

        self.csv_path_var = tk.StringVar()
        self.out_dir_var = tk.StringVar(value=os.path.abspath("./bayes_sss_outputs_journal"))
        self.random_seed_var = tk.StringVar(value="42")
        self.model_var = tk.StringVar(value="known_var")
        self.mu0_var = tk.StringVar(value="0.0")
        self.sigma0_var = tk.StringVar(value="1.0")
        self.sigma_obs_var = tk.StringVar(value="1.0")
        self.kappa0_var = tk.StringVar(value="1.0")
        self.alpha0_var = tk.StringVar(value="2.0")
        self.beta0_var = tk.StringVar(value="1.0")
        self.n_sim_var = tk.StringVar(value="100")
        self.batch_size_var = tk.StringVar(value="10")
        self.tolerance_var = tk.StringVar(value="0.02")
        self.consecutive_k_var = tk.StringVar(value="5")
        self.save_traces_var = tk.BooleanVar(value=False)
        self.show_prior_post_var = tk.BooleanVar(value=False)
        self.yscale_var = tk.StringVar(value="log")
        self.auto_scale_var = tk.BooleanVar(value=True)

        self.mc_min_batch_var = tk.StringVar(value="2")
        self.mc_max_batch_var = tk.StringVar(value="500")
        self.mc_n_sizes_var = tk.StringVar(value="70")
        self.mc_draws_var = tk.StringVar(value="50")
        self.mc_color_limit_var = tk.StringVar(value="10")

        self._build_ui()
        self.pack(fill="both", expand=True)
        self.master.after(100, self._process_queue)

    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}

        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text="Input CSV:").pack(side="left", **pad)
        ttk.Entry(top, textvariable=self.csv_path_var, width=76).pack(side="left", **pad)
        ttk.Button(top, text="Browse…", command=self._choose_csv).pack(side="left", **pad)

        mid = ttk.Frame(self)
        mid.pack(fill="x")
        ttk.Label(mid, text="Output folder:").pack(side="left", **pad)
        ttk.Entry(mid, textvariable=self.out_dir_var, width=64).pack(side="left", **pad)
        ttk.Button(mid, text="Browse…", command=self._choose_outdir).pack(side="left", **pad)

        model = ttk.LabelFrame(self, text="Bayesian model")
        model.pack(fill="x", padx=8, pady=6)
        ttk.Radiobutton(model, text="Normal–Normal (known observation variance)", variable=self.model_var, value="known_var", command=self._toggle_model).grid(row=0, column=0, sticky="w", **pad)
        ttk.Radiobutton(model, text="Normal–Inverse-Gamma", variable=self.model_var, value="invgamma", command=self._toggle_model).grid(row=0, column=1, sticky="w", **pad)
        self._add_labeled(model, "μ0", self.mu0_var).grid(row=1, column=0, sticky="w", **pad)
        self.known_box = ttk.Frame(model)
        self._add_labeled(self.known_box, "σ0", self.sigma0_var).grid(row=0, column=0, sticky="w", **pad)
        self._add_labeled(self.known_box, "σ_obs", self.sigma_obs_var).grid(row=0, column=1, sticky="w", **pad)
        self.known_box.grid(row=1, column=1, sticky="w", **pad)
        self.inv_box = ttk.Frame(model)
        self._add_labeled(self.inv_box, "κ0", self.kappa0_var, 8).grid(row=0, column=0, sticky="w", **pad)
        self._add_labeled(self.inv_box, "α0", self.alpha0_var, 8).grid(row=0, column=1, sticky="w", **pad)
        self._add_labeled(self.inv_box, "β0", self.beta0_var, 8).grid(row=0, column=2, sticky="w", **pad)
        self.inv_box.grid(row=1, column=2, sticky="w", **pad)

        sim = ttk.LabelFrame(self, text="Simulation")
        sim.pack(fill="x", padx=8, pady=6)
        self._add_labeled(sim, "N_sim", self.n_sim_var, 8).grid(row=0, column=0, sticky="w", **pad)
        self._add_labeled(sim, "Batch size", self.batch_size_var, 8).grid(row=0, column=1, sticky="w", **pad)
        self._add_labeled(sim, "Tolerance ε", self.tolerance_var, 10).grid(row=0, column=2, sticky="w", **pad)
        self._add_labeled(sim, "Consecutive K", self.consecutive_k_var, 10).grid(row=0, column=3, sticky="w", **pad)
        self._add_labeled(sim, "Random seed", self.random_seed_var, 10).grid(row=0, column=4, sticky="w", **pad)
        ttk.Checkbutton(sim, text="Save per-simulation Δμ traces", variable=self.save_traces_var).grid(row=0, column=5, sticky="w", **pad)
        ttk.Checkbutton(sim, text="Show prior/posterior distributions", variable=self.show_prior_post_var, command=self._toggle_prior_post_view).grid(row=0, column=6, sticky="w", **pad)

        cols = ttk.LabelFrame(self, text="Columns")
        cols.pack(fill="both", expand=False, padx=8, pady=6)
        self.listbox = tk.Listbox(cols, selectmode="extended", height=8)
        self.listbox.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        sb = ttk.Scrollbar(cols, command=self.listbox.yview)
        sb.pack(side="left", fill="y")
        self.listbox.configure(yscrollcommand=sb.set)
        btns = ttk.Frame(cols)
        btns.pack(side="left", fill="y", padx=6, pady=6)
        ttk.Button(btns, text="Select all", command=lambda: self.listbox.select_set(0, tk.END)).pack(fill="x", pady=3)
        ttk.Button(btns, text="Clear", command=lambda: self.listbox.selection_clear(0, tk.END)).pack(fill="x", pady=3)
        ttk.Button(btns, text="Reload", command=self._reload_columns).pack(fill="x", pady=3)

        run = ttk.Frame(self)
        run.pack(fill="x", padx=8, pady=6)
        ttk.Button(run, text="Run Bayesian SSS", command=self._run_bayes).pack(side="left", padx=4)
        ttk.Button(run, text="Run Median Monte Carlo", command=self._run_median_mc).pack(side="left", padx=4)
        ttk.Button(run, text="Open output folder", command=self._open_outdir).pack(side="left", padx=4)
        ttk.Label(run, text="Y-scale:").pack(side="right", padx=(8, 4))
        scale_combo = ttk.Combobox(run, textvariable=self.yscale_var, values=["linear", "log"], width=8, state="readonly")
        scale_combo.pack(side="right", padx=4)
        scale_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_scale_only())
        ttk.Checkbutton(run, text="Auto", variable=self.auto_scale_var, command=self._refresh_scale_only).pack(side="right", padx=4)
        ttk.Button(run, text="Help", command=self._show_help).pack(side="right", padx=4)

        progress = ttk.Frame(self)
        progress.pack(fill="x", padx=8, pady=6)
        self.pb = ttk.Progressbar(progress, orient="horizontal", mode="determinate", maximum=100)
        self.pb.pack(fill="x", padx=4, pady=4)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(progress, textvariable=self.status_var).pack(anchor="w")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        bayes_tab = ttk.Frame(self.notebook)
        self.notebook.add(bayes_tab, text="Bayesian SSS")

        bayes_info = ttk.LabelFrame(bayes_tab, text="Live result summary")
        bayes_info.pack(fill="x", padx=2, pady=2)
        self.live_series_var = tk.StringVar(value="Series: -")
        self.live_batch_var = tk.StringVar(value="Required batch number: -")
        self.live_sample_var = tk.StringVar(value="Sufficient sample size Z: -")
        self.live_pred_var = tk.StringVar(value="Predicted batch/sample size: -")
        self.live_note_var = tk.StringVar(value="")
        ttk.Label(bayes_info, textvariable=self.live_series_var).grid(row=0, column=0, sticky="w", padx=8, pady=3)
        ttk.Label(bayes_info, textvariable=self.live_batch_var).grid(row=0, column=1, sticky="w", padx=8, pady=3)
        ttk.Label(bayes_info, textvariable=self.live_sample_var).grid(row=1, column=0, sticky="w", padx=8, pady=3)
        ttk.Label(bayes_info, textvariable=self.live_pred_var).grid(row=1, column=1, sticky="w", padx=8, pady=3)
        ttk.Label(bayes_info, textvariable=self.live_note_var, justify="left", wraplength=520, foreground="#8B0000").grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 4))

        frame1 = ttk.LabelFrame(bayes_tab, text="Live Δμ envelope")
        frame1.pack(fill="both", expand=True, padx=2, pady=2)
        self.fig = Figure(figsize=(14.0, 5.6), constrained_layout=True)
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.2])
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax_prior = self.fig.add_subplot(gs[0, 1])
        self.ax_zoom = self.fig.add_subplot(gs[0, 2])
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame1)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.ax.set_xlabel("Batch number, b")
        self.ax.set_ylabel("Δμ")
        self.ax.set_title("Live Bayesian convergence")
        self.ax_prior.set_title("Prior vs posterior distributions")
        self.ax_prior.set_xlabel("Parameter value")
        self.ax_prior.set_ylabel("Density")
        self.ax_prior.set_title("Prior vs posterior distributions")
        self.ax_prior.set_xlabel("Parameter value")
        self.ax_prior.set_ylabel("Density")
        self.ax_zoom.set_xlabel("Batch number, b (zoomed)")
        self.ax_zoom.set_ylabel("Δμ")
        self.ax_zoom.set_title("Zoomed view near required / power-fit batch")
        self._toggle_prior_post_view()
        frame1.bind("<Configure>", self._on_plot_resize)
        self._toggle_prior_post_view()

        mc_tab = ttk.Frame(self.notebook)
        self.notebook.add(mc_tab, text="Median Monte Carlo")
        mc_ctrl = ttk.LabelFrame(mc_tab, text="Median Monte Carlo controls")
        mc_ctrl.pack(fill="x", padx=2, pady=2)
        self._add_labeled(mc_ctrl, "Min batch", self.mc_min_batch_var, 8).grid(row=0, column=0, sticky="w", **pad)
        self._add_labeled(mc_ctrl, "Max batch", self.mc_max_batch_var, 8).grid(row=0, column=1, sticky="w", **pad)
        self._add_labeled(mc_ctrl, "No. batch sizes", self.mc_n_sizes_var, 8).grid(row=0, column=2, sticky="w", **pad)
        self._add_labeled(mc_ctrl, "Draws / size", self.mc_draws_var, 8).grid(row=0, column=3, sticky="w", **pad)
        self._add_labeled(mc_ctrl, "Color limit (%)", self.mc_color_limit_var, 8).grid(row=0, column=4, sticky="w", **pad)
        frame2 = ttk.LabelFrame(mc_tab, text="Median Monte Carlo plot")
        frame2.pack(fill="both", expand=True, padx=2, pady=2)
        self.mc_fig = Figure(figsize=(8.8, 5.5), constrained_layout=True)
        self.mc_ax = self.mc_fig.add_subplot(111)
        self.mc_canvas = FigureCanvasTkAgg(self.mc_fig, master=frame2)
        self.mc_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.mc_ax.set_xlabel("Sample size, N")
        self.mc_ax.set_ylabel("Median value")
        self.mc_ax.set_title("Median Monte Carlo")

        self._toggle_model()

    def _add_labeled(self, parent, label, var, width=10):
        frm = ttk.Frame(parent)
        ttk.Label(frm, text=f"{label}: ").pack(side="left")
        ttk.Entry(frm, textvariable=var, width=width).pack(side="left")
        return frm

    def _toggle_model(self):
        known = self.model_var.get() == "known_var"
        for widget in self.known_box.winfo_children():
            try:
                widget.configure(state="normal" if known else "disabled")
            except Exception:
                pass
        for widget in self.inv_box.winfo_children():
            try:
                widget.configure(state="disabled" if known else "normal")
            except Exception:
                pass

    def _choose_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.csv_path_var.set(path)
            self._reload_columns()

    def _choose_outdir(self):
        path = filedialog.askdirectory()
        if path:
            self.out_dir_var.set(path)

    def _reload_columns(self):
        try:
            df = pd.read_csv(self.csv_path_var.get().strip())
        except Exception as exc:
            self._error("Failed to load columns", exc)
            return
        self.listbox.delete(0, tk.END)
        for col in df.columns:
            self.listbox.insert(tk.END, col)
        self._set_status(f"Loaded {len(df.columns)} columns.")

    def _selected_columns(self):
        return [self.listbox.get(i) for i in self.listbox.curselection()]

    def _open_outdir(self):
        path = self.out_dir_var.get().strip()
        if not path:
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f"open '{path}'")
            else:
                os.system(f"xdg-open '{path}'")
        except Exception as exc:
            self._error("Could not open output folder", exc)

    def _bayes_config(self) -> BayesianConfig:
        return BayesianConfig(
            model_name=self.model_var.get().strip(),
            mu0=float(self.mu0_var.get()),
            sigma0=float(self.sigma0_var.get()),
            sigma_obs=float(self.sigma_obs_var.get()),
            kappa0=float(self.kappa0_var.get()),
            alpha0=float(self.alpha0_var.get()),
            beta0=float(self.beta0_var.get()),
            n_sim=int(self.n_sim_var.get()),
            batch_size=int(self.batch_size_var.get()),
            tolerance=float(self.tolerance_var.get()),
            consecutive_k=int(self.consecutive_k_var.get()),
            random_seed=int(self.random_seed_var.get()) if self.random_seed_var.get().strip() else None,
            save_traces=bool(self.save_traces_var.get()),
        )

    def _mc_config(self) -> MonteCarloMedianConfig:
        return MonteCarloMedianConfig(
            min_batch=int(self.mc_min_batch_var.get()),
            max_batch=int(self.mc_max_batch_var.get()),
            n_sizes=int(self.mc_n_sizes_var.get()),
            draws_per_size=int(self.mc_draws_var.get()),
            color_limit_percent=float(self.mc_color_limit_var.get()),
            random_seed=int(self.random_seed_var.get()) if self.random_seed_var.get().strip() else None,
        )

    def _load_dataframe(self):
        path = self.csv_path_var.get().strip()
        if not path or not os.path.exists(path):
            raise ValueError("Please select a valid CSV file.")
        return path, pd.read_csv(path)

    def _run_bayes(self):
        if self.running_thread and self.running_thread.is_alive():
            self._set_status("A run is already in progress.")
            return
        try:
            input_path, df = self._load_dataframe()
            self.current_df = df.copy()
            selected = self._selected_columns()
            if not selected:
                raise ValueError("Select at least one column to analyse.")
            config = self._bayes_config()
            config.validate()
        except Exception as exc:
            self._error("Invalid Bayesian setup", exc)
            return

        self.pb.configure(value=0, maximum=max(1, len(selected)))
        self.current_bayes_result = None
        self.current_df = df.copy()
        self.ax.cla()
        self.ax_prior.cla()
        self.ax_zoom.cla()
        self.ax.set_xlabel("Batch number, b")
        self.ax.set_ylabel("Δμ")
        self.ax.set_title("Live Bayesian convergence")
        self.ax_prior.set_title("Prior vs posterior distributions")
        self.ax_prior.set_xlabel("Parameter value")
        self.ax_prior.set_ylabel("Density")
        self.ax_zoom.set_xlabel("Batch number, b (zoomed)")
        self.ax_zoom.set_ylabel("Δμ")
        self.ax_zoom.set_title("Zoomed view near required / power-fit batch")
        self._toggle_prior_post_view()
        self.live_series_var.set("Series: -")
        self.live_batch_var.set("Required batch number: -")
        self.live_sample_var.set("Sufficient sample size Z: -")
        self.live_pred_var.set("Predicted batch/sample size: -")
        self.live_note_var.set("")
        self.canvas.draw_idle()

        def worker():
            try:
                out_dir = os.path.join(self.out_dir_var.get().strip(), "bayesian_sss")
                os.makedirs(out_dir, exist_ok=True)
                summary_rows = []
                results = []
                for col_index, col in enumerate(selected, start=1):
                    values = pd.to_numeric(df[col], errors='coerce').dropna().to_numpy(dtype=float)
                    if len(values) < 2:
                        continue
                    rng = np.random.default_rng(config.random_seed)
                    traces = []
                    sufficient_batch = None
                    predicted_batch = None
                    fit_model = None
                    fit_params = None
                    fit_rmse = None
                    delta_mu_max = np.array([], dtype=float)
                    q05 = np.array([], dtype=float)
                    q50 = np.array([], dtype=float)
                    q95 = np.array([], dtype=float)
                    last_sim_path = None

                    for sim_index in range(config.n_sim):
                        sim_path = _run_one_sim_with_paths(values, config, rng)
                        last_sim_path = sim_path
                        traces.append(sim_path['delta_trace'])
                        delta_mu_max = _nanmax_aligned(traces)
                        q05 = _nanpercentile_aligned(traces, 5)
                        q50 = _nanpercentile_aligned(traces, 50)
                        q95 = _nanpercentile_aligned(traces, 95)
                        sufficient_batch = _rolling_k_consecutive(delta_mu_max < config.tolerance, config.consecutive_k)
                        prediction_note = ""
                        prediction_status = "criterion_satisfied"
                        if sufficient_batch is None:
                            fit_model, fit_params, fit_rmse, predicted_batch = _fit_and_predict(delta_mu_max, config)
                            if predicted_batch is None:
                                prediction_status = "power_fit_not_predictable"
                                prediction_note = (
                                    "Power-fit prediction is not reliable with the current particle count. "
                                    "Please provide a much larger number of particles/observations and rerun the analysis."
                                )
                            else:
                                prediction_status = "power_fit_prediction"
                        else:
                            predicted_batch = None
                            fit_model = None
                            fit_params = None
                            fit_rmse = None

                        self.msg_queue.put(("bayes_progress", {
                            'series': col,
                            'simulation_index': sim_index + 1,
                            'n_sim': config.n_sim,
                            'delta_mu_max': delta_mu_max.copy(),
                            'delta_mu_q05': q05.copy(),
                            'delta_mu_q50': q50.copy(),
                            'delta_mu_q95': q95.copy(),
                            'sufficient_batch': sufficient_batch,
                            'predicted_batch': predicted_batch,
                            'fit_model': fit_model,
                            'fit_params': fit_params,
                            'fit_rmse': fit_rmse,
                            'prediction_status': prediction_status,
                            'prediction_note': prediction_note,
                            'batches': sim_path['batches'].copy(),
                            'prior_mu_path': sim_path['prior_mu_path'].copy(),
                            'prior_var_path': sim_path['prior_var_path'].copy(),
                            'post_mu_path': sim_path['post_mu_path'].copy(),
                            'post_var_path': sim_path['post_var_path'].copy(),
                            'column_index': col_index,
                            'n_columns': len(selected),
                        }))

                    result = SeriesResult(
                        series=col,
                        n_observations=int(len(values)),
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
                    results.append(result)
                    summary_rows.append(result.to_summary_row())
                    exported_paths = save_series_result(result, out_dir)
                    envelope_csv = exported_paths.get("delta_mu_envelope_csv")
                    if envelope_csv and os.path.exists(envelope_csv):
                        try:
                            env_df = pd.read_csv(envelope_csv)
                            recommended_batch = result.sufficient_batch if result.sufficient_batch is not None else result.predicted_batch
                            recommended_samples = result.sufficient_samples if result.sufficient_samples is not None else result.predicted_samples
                            sample_size_basis = "observed_convergence" if result.sufficient_batch is not None else ("power_fit_prediction" if result.predicted_batch is not None else None)
                            env_df["sufficient_batch"] = result.sufficient_batch
                            env_df["sufficient_samples"] = result.sufficient_samples
                            env_df["predicted_batch"] = result.predicted_batch
                            env_df["predicted_samples"] = result.predicted_samples
                            prediction_status = (
                                "observed_convergence"
                                if result.sufficient_batch is not None
                                else ("power_fit_prediction" if result.predicted_batch is not None else "power_fit_not_predictable")
                            )
                            prediction_note = (
                                ""
                                if result.sufficient_batch is not None or result.predicted_batch is not None
                                else "Power-fit prediction was not reliable with the current particle count. Please provide a much larger number of particles/observations and rerun the analysis."
                            )
                            env_df["recommended_batch"] = recommended_batch
                            env_df["recommended_samples"] = recommended_samples
                            env_df["sample_size_basis"] = sample_size_basis
                            env_df["fit_model"] = result.fit_model
                            env_df["prediction_status"] = prediction_status
                            env_df["prediction_note"] = prediction_note
                            env_df.to_csv(envelope_csv, index=False)
                        except Exception:
                            pass
                    plot_delta_mu_result(result, os.path.join(out_dir, f"{result.series.replace(' ', '_')}_delta_mu_plot.png"))

                if summary_rows:
                    save_summary(results, out_dir)
                save_run_metadata(config, selected, input_path, out_dir)
                self.msg_queue.put(("done", {"message": f"Bayesian analysis completed. Outputs saved to:\n{out_dir}"}))
            except Exception as exc:
                self.msg_queue.put(("error", {"title": "Bayesian run failed", "error": f"{exc}\n\n{traceback.format_exc()}"}))

        self.running_thread = threading.Thread(target=worker, daemon=True)
        self.running_thread.start()
        self._set_status("Running Bayesian SSS…")

    def _run_median_mc(self):
        if self.running_thread and self.running_thread.is_alive():
            self._set_status("A run is already in progress.")
            return
        try:
            _, df = self._load_dataframe()
            selected = self._selected_columns()
            if not selected:
                raise ValueError("Select at least one column to analyse.")
            config = self._mc_config()
            config.validate()
        except Exception as exc:
            self._error("Invalid Monte Carlo setup", exc)
            return

        def worker():
            try:
                outputs = run_median_monte_carlo(df, selected, config)
                out_dir = os.path.join(self.out_dir_var.get().strip(), "median_monte_carlo")
                os.makedirs(out_dir, exist_ok=True)
                for idx, (series, df_plot) in enumerate(outputs.items(), start=1):
                    csv_path = os.path.join(out_dir, f"{series.replace(' ', '_')}_median_mc_points.csv")
                    png_path = os.path.join(out_dir, f"{series.replace(' ', '_')}_median_mc_plot.png")
                    df_plot.to_csv(csv_path, index=False)
                    fig, ax = plt.subplots(figsize=(13.5, 5.2))
                    scatter = render_monte_carlo_median_plot(ax, df_plot, series, config.color_limit_percent)
                    if scatter is not None:
                        cbar = fig.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.12, fraction=0.08)
                        cbar.set_label("Error level [%]")
                    fig.tight_layout()
                    fig.savefig(png_path, dpi=200, bbox_inches="tight")
                    plt.close(fig)
                    self.msg_queue.put(("mc_progress", {"series": series, "df_plot": df_plot, "idx": idx, "total": len(outputs), "color_limit": config.color_limit_percent}))
                self.msg_queue.put(("done", {"message": f"Median Monte Carlo analysis completed. Outputs saved to:\n{out_dir}"}))
            except Exception as exc:
                self.msg_queue.put(("error", {"title": "Median Monte Carlo run failed", "error": f"{exc}\n\n{traceback.format_exc()}"}))

        self.running_thread = threading.Thread(target=worker, daemon=True)
        self.running_thread.start()
        self._set_status("Running median Monte Carlo…")

    def _process_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "bayes_progress":
                    self._handle_bayes_progress(payload)
                elif kind == "mc_progress":
                    self._handle_mc_progress(payload)
                elif kind == "done":
                    self._set_status(payload["message"].splitlines()[0])
                    messagebox.showinfo("Completed", payload["message"])
                elif kind == "error":
                    self._error(payload["title"], payload["error"])
        except queue.Empty:
            pass
        self.master.after(100, self._process_queue)

    def _handle_bayes_progress(self, payload):
        self.pb.configure(value=payload["column_index"], maximum=payload["n_columns"])
        self.current_bayes_result = payload
        self._last_progress_payload = payload
        self.ax.cla()

        delta_mu = np.asarray(payload["delta_mu_max"], dtype=float)
        q05 = np.asarray(payload.get("delta_mu_q05", []), dtype=float)
        q50 = np.asarray(payload.get("delta_mu_q50", []), dtype=float)
        q95 = np.asarray(payload.get("delta_mu_q95", []), dtype=float)
        batches = np.arange(1, len(delta_mu) + 1, dtype=float)
        tol_raw = float(self.tolerance_var.get())

        scale = self._choose_y_scale([delta_mu, q05, q50, q95, np.array([tol_raw])])
        delta_mu_plot = self._safe_for_scale(delta_mu, scale)
        q05_plot = self._safe_for_scale(q05, scale) if q05.size else q05
        q50_plot = self._safe_for_scale(q50, scale) if q50.size else q50
        q95_plot = self._safe_for_scale(q95, scale) if q95.size else q95
        tol_plot = max(tol_raw, 1e-12) if scale == "log" else tol_raw

        main_color = "#0072B2"
        band_color = "#9ecae1"
        median_color = "#8B0000"
        tol_color = "#D55E00"
        req_color = "#009E73"
        pred_color = "#CC79A7"
        Hex_color = "#8B0000"

        if q05.size == delta_mu.size and q95.size == delta_mu.size and delta_mu.size > 0:
            self.ax.fill_between(batches, q05_plot, q95_plot, color=band_color, alpha=0.28, label="Simulation envelope (5–95%)")
        if q50.size == delta_mu.size and delta_mu.size > 0:
            self.ax.plot(batches, q50_plot, color=median_color, linewidth=1.35, linestyle='-', label="Median Δμ")

        self.ax.plot(
            batches,
            delta_mu_plot,
            marker="o",
            markersize=2.0,
            markerfacecolor=main_color,
            markeredgewidth=0.0,
            color=main_color,
            linewidth=1.25,
            label="Worst-case Δμ_max",
        )

        fit_y = None
        if payload.get("sufficient_batch") is None and payload.get("fit_model") == "pow" and payload.get("fit_params") is not None:
            fit_x = np.arange(1, max(len(delta_mu), int(payload.get("predicted_batch") or 0)) + 1, dtype=float)
            fit_y = self._power_fit_values(fit_x, payload.get("fit_params"))
            if fit_y is not None:
                self.ax.plot(fit_x, self._safe_for_scale(fit_y, scale), color=pred_color, linewidth=2.0, linestyle='-.', label='Power-fit prediction')

        self.ax.axhline(tol_plot, linestyle="--", color=tol_color, linewidth=1.2, label="Tolerance ε")
        if payload.get("sufficient_batch") is not None:
            self.ax.axvline(payload["sufficient_batch"], linestyle="--", color=req_color, linewidth=1.6, label="Required batch")
        elif payload.get("predicted_batch") is not None:
            self.ax.axvline(payload["predicted_batch"], linestyle=":", color=pred_color, linewidth=2.0, label="Predicted batch (power fit)")
            pred_b = payload.get("predicted_batch")
            if fit_y is not None and pred_b is not None and 1 <= int(pred_b) <= len(fit_y):
                pred_y = float(self._safe_for_scale(np.array([fit_y[int(pred_b) - 1]]), scale)[0])
                self.ax.scatter([pred_b], [pred_y], s=32, color=pred_color, edgecolors='white', linewidths=0.6, zorder=5)
                self.ax.annotate(
                    f"Power-fit b≈{int(pred_b)}",
                    xy=(pred_b, pred_y),
                    xytext=(8, -14),
                    textcoords='offset points',
                    fontsize=8.5,
                    color=pred_color,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=pred_color, alpha=0.85),
                )

        self.ax.set_xlabel("Batch number, b")
        self._apply_y_scale(self.ax, scale, "Δμ")
        self.ax.set_title(f"{payload['series']} — {payload['simulation_index']}/{payload['n_sim']} simulations")
        self.ax.grid(True, alpha=0.22)
        self.ax.legend(loc="best", frameon=True, fontsize=8.5)

        self._draw_prior_posterior_view(payload)
        self._draw_zoom_view(
            batches=batches,
            delta_mu=delta_mu,
            tol=tol_raw,
            sufficient_batch=payload.get("sufficient_batch"),
            predicted_batch=payload.get("predicted_batch"),
        )
        self._format_live_summary(payload)

        self._apply_plot_margins()
        self.canvas.draw_idle()
        self._set_status(f"[{payload['series']}] {payload['simulation_index']}/{payload['n_sim']} simulations")

    def _handle_mc_progress(self, payload):
        self.pb.configure(value=payload["idx"], maximum=payload["total"])
        if self.current_mc_cbar is not None:
            try:
                self.current_mc_cbar.remove()
            except Exception:
                pass
            self.current_mc_cbar = None
        self.current_bayes_result = None
        self.current_df = None
        scatter = render_monte_carlo_median_plot(self.mc_ax, payload["df_plot"], payload["series"], payload["color_limit"])
        if scatter is not None:
            self.current_mc_cbar = self.mc_fig.colorbar(scatter, ax=self.mc_ax, orientation="horizontal", pad=0.12, fraction=0.08)
            self.current_mc_cbar.set_label("Error level [%]")
        self.mc_canvas.draw_idle()
        self.notebook.select(1)
        self._set_status(f"Median Monte Carlo saved for [{payload['series']}] ({payload['idx']}/{payload['total']})")


    def _choose_y_scale(self, arrays=None):
        manual = (self.yscale_var.get().strip().lower() or "log")
        if not self.auto_scale_var.get():
            return manual if manual in ("linear", "log") else "log"
        vals = []
        if arrays:
            for arr in arrays:
                if arr is None:
                    continue
                try:
                    a = np.asarray(arr, dtype=float).ravel()
                    a = a[np.isfinite(a)]
                    a = a[a > 0]
                    if a.size:
                        vals.append(a)
                except Exception:
                    pass
        if not vals:
            return manual if manual in ("linear", "log") else "log"
        cat = np.concatenate(vals)
        if cat.size < 2:
            return manual if manual in ("linear", "log") else "log"
        vmin = float(np.min(cat))
        vmax = float(np.max(cat))
        if vmin <= 0:
            return "linear"
        return "log" if (vmax / vmin) >= 50.0 else "linear"

    def _apply_y_scale(self, ax, scale, ylabel=None):
        ax.set_yscale(scale)
        if ylabel is not None:
            ax.set_ylabel(ylabel + (" (log scale)" if scale == "log" else ""))
        if scale == "log":
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
            ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs="auto"))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    def _safe_for_scale(self, arr, scale):
        a = np.asarray(arr, dtype=float)
        if scale == "log":
            return np.maximum(a, 1e-12)
        return a

    def _refresh_scale_only(self):
        try:
            if hasattr(self, "_last_progress_payload") and self._last_progress_payload:
                self._handle_bayes_progress(self._last_progress_payload)
        except Exception:
            pass

    def _show_help(self):
        help_text = """Bayesian SSS tab
• Select one or more numeric columns from the input CSV.
• N_sim is the number of Monte Carlo shuffles.
• Batch size is the number of particles/observations assimilated per update.
• Tolerance ε is the convergence threshold applied to the live Δμ envelope.
• Consecutive K is the number of successive batches that must remain below ε.

Live outputs
• Required batch number = first batch where the K-consecutive stopping rule is satisfied.
• Sufficient sample size Z = required batch number × batch size.
• If convergence is not reached yet, the app reports the sufficient sample size from a highlighted power-fit curve.
• If the power fit is still not reliable, the app explicitly warns that the current particle count is insufficient for prediction and recommends using a much larger dataset.
• Enable “Show prior/posterior distributions” to display an overlay of the representative prior and posterior densities in the middle panel.
• The right-hand graph is a zoomed real-time view centred near the detected or predicted batch.
• Y-scale can be switched between linear and log. In Auto mode, the app uses log scale when Δμ spans a wide range.

Median Monte Carlo tab
• Generates random-batch median plots against sample size and saves CSV/PNG outputs.
"""
        messagebox.showinfo("Help — Bayesian SSS App", help_text)

    def _format_live_summary(self, payload):
        batch_size = int(self.batch_size_var.get()) if self.batch_size_var.get().strip() else 1
        self.live_series_var.set(f"Series: {payload['series']}")
        sufficient_batch = payload.get("sufficient_batch")
        predicted_batch = payload.get("predicted_batch")
        prediction_note = payload.get("prediction_note", "") or ""
        if sufficient_batch is not None:
            z = int(sufficient_batch) * batch_size
            self.live_batch_var.set(f"Required batch number: {int(sufficient_batch)}")
            self.live_sample_var.set(f"Sufficient sample size Z: {z} = {int(sufficient_batch)} × {batch_size}")
            self.live_pred_var.set("Predicted batch/sample size: not needed (criterion satisfied)")
            self.live_note_var.set("")
        else:
            self.live_batch_var.set("Required batch number: criterion not yet satisfied")
            if predicted_batch is not None:
                z_pred = int(predicted_batch) * batch_size
                self.live_sample_var.set(f"Sufficient sample size Z (power fit): ≈ {z_pred} = {int(predicted_batch)} × {batch_size}")
                self.live_pred_var.set(
                    f"Predicted batch/sample size from power fit: b ≈ {int(predicted_batch)}, Z ≈ {z_pred}"
                )
                self.live_note_var.set("")
            else:
                self.live_sample_var.set("Sufficient sample size Z (power fit): not predictable from current fit")
                self.live_pred_var.set("Predicted batch/sample size from power fit: unavailable")
                self.live_note_var.set(
                    prediction_note
                    or "Power-fit prediction is not reliable with the current particle count. Please provide a much larger number of particles/observations and rerun the analysis."
                )

    def _toggle_prior_post_view(self):
        show = bool(self.show_prior_post_var.get())
        try:
            self.ax_prior.set_visible(show)
        except Exception:
            pass
        if show:
            self.ax_prior.set_title("Prior vs posterior distributions")
            self.ax_prior.set_xlabel("Parameter value")
            self.ax_prior.set_ylabel("Density")
        else:
            self.ax_prior.cla()
            self.ax_prior.set_visible(False)
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    @staticmethod
    def _normal_pdf(x, mu, var):
        if not np.isfinite(mu) or not np.isfinite(var) or var <= 0:
            return None
        sigma = float(np.sqrt(var))
        return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    @staticmethod
    def _power_fit_values(x, fit_params):
        if fit_params is None:
            return None
        try:
            a, b, c = fit_params
            x = np.asarray(x, dtype=float)
            y = float(a) * np.power(np.maximum(x, 1e-9), -float(b)) + float(c)
            return y
        except Exception:
            return None


    def _draw_prior_posterior_view(self, payload):
        if not bool(self.show_prior_post_var.get()):
            self.ax_prior.cla()
            self.ax_prior.set_visible(False)
            return

        self.ax_prior.set_visible(True)
        self.ax_prior.cla()

        batches = np.asarray(payload.get("batches", []), dtype=float)
        prior_mu_path = np.asarray(payload.get("prior_mu_path", []), dtype=float)
        prior_var_path = np.asarray(payload.get("prior_var_path", []), dtype=float)
        post_mu_path = np.asarray(payload.get("post_mu_path", []), dtype=float)
        post_var_path = np.asarray(payload.get("post_var_path", []), dtype=float)

        if batches.size == 0 or prior_mu_path.size == 0 or post_mu_path.size == 0:
            self.ax_prior.set_title("Prior vs posterior distributions")
            self.ax_prior.text(0.5, 0.5, "No distribution data available", ha="center", va="center", transform=self.ax_prior.transAxes)
            return

        idx = int(len(batches) - 1)
        mu_prior = float(prior_mu_path[idx])
        mu_post = float(post_mu_path[idx])
        var_prior = float(prior_var_path[idx])
        var_post = float(post_var_path[idx])

        finite_vars = [v for v in [var_prior, var_post] if np.isfinite(v) and v > 0]
        if not finite_vars:
            self.ax_prior.set_title("Prior vs posterior distributions")
            self.ax_prior.text(0.5, 0.5, "Variance unavailable for current state", ha="center", va="center", transform=self.ax_prior.transAxes)
            return

        spread = 4.5 * max(np.sqrt(finite_vars))
        x_min = min(mu_prior, mu_post) - spread
        x_max = max(mu_prior, mu_post) + spread
        if x_max <= x_min:
            x_max = x_min + 1.0
        x = np.linspace(x_min, x_max, 500)

        prior_pdf = self._normal_pdf(x, mu_prior, var_prior)
        post_pdf = self._normal_pdf(x, mu_post, var_post)
        prior_color = "#56B4E9"
        post_color = "#E69F00"
        ci_color = "#999999"

        # faint historical posterior envelopes for visual convergence context
        hist_count = min(6, len(batches))
        hist_indices = np.linspace(0, len(batches) - 1, hist_count, dtype=int)
        for h in hist_indices[:-1]:
            h_mu = float(post_mu_path[h])
            h_var = float(post_var_path[h])
            h_pdf = self._normal_pdf(x, h_mu, h_var)
            if h_pdf is not None:
                self.ax_prior.plot(x, h_pdf, color=post_color, linewidth=0.9, alpha=0.18)

        if prior_pdf is not None:
            self.ax_prior.plot(x, prior_pdf, color=prior_color, linewidth=1.8, label="Prior")
            self.ax_prior.fill_between(x, 0.0, prior_pdf, color=prior_color, alpha=0.14)
        if post_pdf is not None:
            self.ax_prior.plot(x, post_pdf, color=post_color, linewidth=1.9, label="Posterior")
            self.ax_prior.fill_between(x, 0.0, post_pdf, color=post_color, alpha=0.18)

        self.ax_prior.axvline(mu_prior, color=prior_color, linestyle="--", linewidth=1.1)
        self.ax_prior.axvline(mu_post, color=post_color, linestyle="--", linewidth=1.1)

        if np.isfinite(var_prior) and var_prior > 0:
            sig_prior = np.sqrt(var_prior)
            self.ax_prior.axvspan(mu_prior - 1.96 * sig_prior, mu_prior + 1.96 * sig_prior, color=prior_color, alpha=0.06)
        if np.isfinite(var_post) and var_post > 0:
            sig_post = np.sqrt(var_post)
            self.ax_prior.axvspan(mu_post - 1.96 * sig_post, mu_post + 1.96 * sig_post, color=post_color, alpha=0.08)
            self.ax_prior.annotate(
                f"95% CI width ≈ {2 * 1.96 * sig_post:.3g}",
                xy=(mu_post, np.nanmax(post_pdf) if post_pdf is not None else 0.0),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8.5,
                color=ci_color,
            )

        self.ax_prior.set_title(f"Current prior/posterior at b = {int(batches[idx])}")
        self.ax_prior.set_xlabel("Parameter value")
        self.ax_prior.set_ylabel("Density")
        self.ax_prior.grid(True, alpha=0.20)
        self.ax_prior.legend(loc="best", frameon=True, fontsize=8)

    def _draw_zoom_view(self, batches, delta_mu, tol, sufficient_batch, predicted_batch):
        self.ax_zoom.cla()
        main_color = "#0072B2"
        tol_color = "#D55E00"
        req_color = "#009E73"
        pred_color = "#CC79A7"

        scale = self._choose_y_scale([delta_mu, np.array([tol])])
        delta_mu_plot = self._safe_for_scale(delta_mu, scale)
        tol_plot = max(float(tol), 1e-12) if scale == "log" else float(tol)

        self.ax_zoom.plot(
            batches,
            delta_mu_plot,
            marker="o",
            markersize=2.0,
            markerfacecolor=main_color,
            markeredgewidth=0.0,
            color=main_color,
            linewidth=1.25,
        )
        self.ax_zoom.axhline(tol_plot, linestyle="--", color=tol_color, linewidth=1.2)

        focus_batch = sufficient_batch if sufficient_batch is not None else predicted_batch
        if focus_batch is None:
            focus_batch = int(max(1, np.ceil(0.8 * len(batches)))) if len(batches) else 1
        focus_batch = int(focus_batch)
        window = max(6, min(25, max(4, len(batches) // 3)))
        xmin = max(1, focus_batch - window)
        xmax = max(xmin + 4, min(int(batches[-1]) if len(batches) else focus_batch + window, focus_batch + window))
        if len(batches) and xmax <= xmin:
            xmax = xmin + 4
        self.ax_zoom.set_xlim(xmin, xmax)

        zoom_mask = (batches >= xmin) & (batches <= xmax)
        if np.any(zoom_mask):
            y_zoom = delta_mu_plot[zoom_mask]
            y_low = min(float(np.nanmin(y_zoom)), tol_plot)
            y_high = max(float(np.nanmax(y_zoom)), tol_plot)
            if scale == "log":
                low = max(1e-12, y_low * 0.85)
                high = y_high * 1.15 if y_high > 0 else 1.0
                if high <= low:
                    high = low * 10.0
                self.ax_zoom.set_ylim(low, high)
            else:
                pad = 0.08 * (y_high - y_low) if y_high > y_low else max(abs(y_high) * 0.08, tol_plot * 0.15, 1e-6)
                self.ax_zoom.set_ylim(max(0.0, y_low - pad), y_high + pad)

        fit_params = self.current_bayes_result.get("fit_params") if isinstance(self.current_bayes_result, dict) else None
        fit_model = self.current_bayes_result.get("fit_model") if isinstance(self.current_bayes_result, dict) else None
        fit_y = None
        if sufficient_batch is None and predicted_batch is not None and fit_model == 'pow' and fit_params is not None:
            fit_x = np.arange(1, max(int(batches[-1]) if len(batches) else 1, int(predicted_batch)) + 1, dtype=float)
            fit_y = self._power_fit_values(fit_x, fit_params)
            if fit_y is not None:
                self.ax_zoom.plot(fit_x, self._safe_for_scale(fit_y, scale), color=pred_color, linewidth=1.4, linestyle='-.')

        if sufficient_batch is not None:
            self.ax_zoom.axvline(sufficient_batch, linestyle="--", color=req_color, linewidth=1.6)
        elif predicted_batch is not None:
            self.ax_zoom.axvline(predicted_batch, linestyle=":", color=pred_color, linewidth=2.0)
            if fit_y is not None and 1 <= int(predicted_batch) <= len(fit_y):
                pred_y = float(self._safe_for_scale(np.array([fit_y[int(predicted_batch) - 1]]), scale)[0])
                self.ax_zoom.scatter([predicted_batch], [pred_y], s=26, color=pred_color, edgecolors='white', linewidths=0.6, zorder=5)

        self.ax_zoom.set_xlabel("Batch number, b")
        self._apply_y_scale(self.ax_zoom, scale, "Δμ")
        self.ax_zoom.set_title("Zoomed live view")
        self.ax_zoom.grid(True, alpha=0.22)

    def _apply_plot_margins(self):
        try:
            self.ax.margins(x=0.03, y=0.08)
            self.ax_prior.margins(x=0.03, y=0.08)
            self.ax_zoom.margins(x=0.03, y=0.08)
        except Exception:
            pass

    def _on_plot_resize(self, event=None):
        if self._resize_after_id is not None:
            try:
                self.after_cancel(self._resize_after_id)
            except Exception:
                pass
        self._resize_after_id = self.after(120, self._redraw_plots)

    def _redraw_plots(self):
        self._resize_after_id = None
        try:
            self._apply_plot_margins()
            self.canvas.draw_idle()
            self.mc_ax.margins(x=0.03, y=0.08)
            self.mc_canvas.draw_idle()
        except Exception:
            pass

    def _set_status(self, message: str):
        self.status_var.set(message)
        self.update_idletasks()

    def _error(self, title, err):
        self.status_var.set(f"{title} — see dialog")
        messagebox.showerror(title, str(err))


def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = BayesSSSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
