# Bayesian Sufficient Sample Size Package

This package is a journal-oriented refactoring of the uploaded research code for the manuscript:

**Sample Size Effects on Morphological and Particle Size Distribution Characteristics of Granular Materials: 2D and 3D Analyses**

**Authors**
- Farzad Kaviani-Hamedani
- Arman Khoshghalb
- Mahdi Esmailzade
- Nasser Khalili

**Affiliation**
Civil and Environmental Engineering, University of New South Wales, Sydney, Australia.

## What was changed

The original single-file UI script was reorganized into four modules:

- `bayes_sss_core.py` — scientific computation and reproducible outputs
- `bayes_sss_plots.py` — figure generation
- `bayes_sss_cli.py` — command-line interface for supplementary-material reproducibility
- `bayes_sss_ui.py` — desktop Tkinter interface for interactive use

## Methodological consistency with the manuscript

The package preserves the manuscript workflow:

1. read one or more CSV columns containing descriptor values;
2. perform repeated Monte Carlo reshuffling;
3. update the Bayesian model batch-by-batch;
4. compute the change in the posterior location parameter, `Δμ`;
5. use the maximum envelope across simulations as a conservative convergence indicator;
6. define sufficient sample size when `Δμ < ε` for `K` consecutive batches;
7. if direct stopping is not reached, estimate the required batch by curve fitting.

Default values consistent with the manuscript are retained:

- `N_sim = 100`
- `batch_size = 10`
- `consecutive_k = 5`

## Outputs

The package writes reproducible outputs such as:

- `run_metadata.json`
- `sufficient_sample_summary.csv`
- `*_delta_mu_envelope.csv`
- `*_delta_mu_plot.png`
- optional `*_delta_mu_traces.csv`
- `median_monte_carlo/*.csv`
- `median_monte_carlo/*.png`

## Example CLI use

```bash
python bayes_sss_cli.py \
  --input morphological_indices.csv \
  --out outputs \
  --columns Sphericity Roundness Elongation \
  --mode both \
  --model known_var \
  --mu0 0.0 \
  --sigma0 1.0 \
  --sigma-obs 1.0 \
  --n-sim 100 \
  --batch-size 10 \
  --tolerance 0.02 \
  --consecutive-k 5 \
  --seed 42
```

## Example UI use

```bash
python bayes_sss_ui.py
```

## Python requirements

- Python 3.10+
- numpy
- pandas
- matplotlib

Tkinter is included with most standard Python desktop installations.

## Recommended citation note for supplementary material

Use this package as the archived supplementary implementation associated with the manuscript above. If you upload it to a repository, add the final DOI or repository URL to the paper where the manuscript currently shows a placeholder `(link)`.
