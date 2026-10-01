# -----------------------------------------------------------------------------
# Prof. Dr. Bruno Duarte Gomes
# Laboratorio de Neurofisiologia Eduardo Oswaldo Cruz - NEXUS
# Laboratorio de Simulacao e Biologia Computacional
# Universidade Federal do Para
# -----------------------------------------------------------------------------
# simulate_series.py
#   Generate a 1-D synthetic time series with KNOWN, EDITABLE statistics that
#   caricatures an intraoperative microelectrode recording (MER).
#
#   Two mechanisms are offered, both producing heavy tails (q > 1):
#     1) "qgaussian"     : i.i.d. draws from a q-Gaussian of chosen (q, beta).
#     2) "superstat"     : a Gaussian whose LOCAL VARIANCE drifts slowly in
#                          time (superstatistics). Slow variance mixing
#                          generates an effective q > 1 on its own -- the
#                          physical mechanism argued for in the manuscript.
#
#   Everything is a knob. Run `python simulate_series.py` for a demo.
# -----------------------------------------------------------------------------
import numpy as np
from qgaussian import sample_q_gaussian


def simulate(n=20000, mechanism="superstat", q=1.5, beta=1.0,
             tau=500.0, sigma_log=0.6, fs=1.0, seed=0):
    """Return a synthetic series `x` (np.ndarray, length n).

    Parameters
    ----------
    n          : number of samples.
    mechanism  : "qgaussian" (direct) or "superstat" (slow-variance mixing).
    q, beta    : target q-Gaussian parameters (mechanism="qgaussian").
    tau        : correlation length (samples) of the slow variance drift
                 (mechanism="superstat"). Larger tau -> slower drift.
    sigma_log  : spread of the log-variance (mechanism="superstat"). Larger
                 -> heavier tails -> larger effective q.
    fs         : sampling rate (only used for the time axis).
    seed       : RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    if mechanism == "qgaussian":
        x = sample_q_gaussian(n, q=q, beta=beta, rng=rng)

    elif mechanism == "superstat":
        # Slowly varying local std via an Ornstein-Uhlenbeck-like log-variance.
        a = np.exp(-1.0 / tau)
        logvar = np.zeros(n)
        noise = rng.normal(0.0, sigma_log * np.sqrt(1 - a * a), n)
        for t in range(1, n):
            logvar[t] = a * logvar[t - 1] + noise[t]
        local_std = np.exp(0.5 * logvar)
        x = rng.normal(0.0, 1.0, n) * local_std

    else:
        raise ValueError("mechanism must be 'qgaussian' or 'superstat'")

    return x


def zscore(x):
    """Unit-variance normalization -- the same step used on the real MERs."""
    return (x - np.mean(x)) / np.std(x)


def excess_kurtosis(x):
    x = (x - x.mean()) / x.std()
    return float(np.mean(x ** 4) - 3.0)


if __name__ == "__main__":
    x = simulate(n=20000, mechanism="superstat", tau=500, sigma_log=0.6, seed=1)
    xz = zscore(x)
    print(f"n={xz.size}  mean={xz.mean():+.3f}  std={xz.std():.3f}  "
          f"excess_kurtosis={excess_kurtosis(xz):.2f}  "
          f"(>0 => heavier-than-Gaussian tails, i.e. q>1)")
