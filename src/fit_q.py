# -----------------------------------------------------------------------------
# Prof. Dr. Bruno Duarte Gomes
# Laboratorio de Neurofisiologia Eduardo Oswaldo Cruz - NEXUS
# Laboratorio de Simulacao e Biologia Computacional
# Universidade Federal do Para
# -----------------------------------------------------------------------------
# fit_q.py
#   Recover (q, beta) from a 1-D series by the SAME idea used in the paper:
#   build the amplitude histogram, then find the (q, beta) whose q-Gaussian
#   best fits it (grid search on R^2). Deliberately simple and transparent.
#
#   This reproduces, in caricature, the estimator the manuscript uses -- and
#   its known weaknesses (grid bounds, R^2-maximization). It is a teaching
#   artifact, not the production estimator. For real work, prefer MLE /
#   q-CLT methods with confidence intervals (see README, "Honest caveats").
# -----------------------------------------------------------------------------
import numpy as np
from qgaussian import q_gaussian_pdf


def amplitude_histogram(x, bins=101, range_sd=6.0):
    """Density histogram of z-scored amplitudes, centers + density returned."""
    xz = (x - np.mean(x)) / np.std(x)
    lo, hi = -range_sd, range_sd
    counts, edges = np.histogram(xz, bins=bins, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep = counts > 0                      # drop empty bins for a stable fit
    return centers[keep], counts[keep]


def fit_q_grid(x, q_grid=None, beta_grid=None, bins=101):
    """Grid search for (q, beta) maximizing R^2 against a q-Gaussian pdf.

    Returns dict with q, beta, r2, and a flag `on_boundary` that is True when
    the optimum sits on the edge of either grid (an estimator red flag,
    exactly the pathology seen in 3 of the 184 real fits).
    """
    if q_grid is None:
        q_grid = np.round(np.arange(1.05, 2.50 + 1e-9, 0.01), 3)
    if beta_grid is None:
        beta_grid = np.logspace(-1, 2, 60)   # 0.1 ... 100

    cx, cy = amplitude_histogram(x, bins=bins)
    ss_tot = np.sum((cy - cy.mean()) ** 2)

    best = dict(q=np.nan, beta=np.nan, r2=-np.inf, on_boundary=False)
    for q in q_grid:
        for beta in beta_grid:
            yhat = q_gaussian_pdf(cx, q, beta)
            r2 = 1.0 - np.sum((cy - yhat) ** 2) / ss_tot
            if r2 > best["r2"]:
                best.update(q=float(q), beta=float(beta), r2=float(r2))

    best["on_boundary"] = bool(
        best["q"] in (q_grid[0], q_grid[-1]) or
        best["beta"] in (beta_grid[0], beta_grid[-1]))
    return best


if __name__ == "__main__":
    from simulate_series import simulate
    # Make a series with a KNOWN q, then try to recover it.
    x = simulate(n=40000, mechanism="qgaussian", q=1.5, beta=1.0, seed=3)
    res = fit_q_grid(x)
    print(f"true q=1.50   recovered q={res['q']:.2f}  "
          f"beta={res['beta']:.2f}  R^2={res['r2']:.3f}  "
          f"on_boundary={res['on_boundary']}")
