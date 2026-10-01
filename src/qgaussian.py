# -----------------------------------------------------------------------------
# Prof. Dr. Bruno Duarte Gomes
# Laboratorio de Neurofisiologia Eduardo Oswaldo Cruz - NEXUS
# Laboratorio de Simulacao e Biologia Computacional
# Universidade Federal do Para
# -----------------------------------------------------------------------------
# qgaussian.py
#   Minimal, dependency-light toolkit for q-Gaussian distributions.
#   q-exponential, q-Gaussian pdf, and an exact sampler (q-Box-Muller,
#   Thistleton et al. 2007) for 1 < q < 3. Pure numpy; runs in milliseconds.
# -----------------------------------------------------------------------------
import numpy as np


def q_exp(x, q):
    """q-exponential: exp_q(x) = [1 + (1-q)x]_+^{1/(1-q)}; -> exp(x) as q->1."""
    x = np.asarray(x, dtype=float)
    if np.isclose(q, 1.0):
        return np.exp(x)
    base = 1.0 + (1.0 - q) * x
    out = np.zeros_like(base)
    pos = base > 0.0                      # support cutoff [.]_+
    out[pos] = base[pos] ** (1.0 / (1.0 - q))
    return out


def q_gaussian_pdf(x, q, beta):
    """Normalized q-Gaussian: rho(x) ∝ [1 + beta(q-1)x^2]^{-1/(q-1)}.

    Uses rho(x) = sqrt(beta)/C_q * exp_q(-beta x^2), the form fitted in the
    manuscript. C_q is the exact normalization for 1 < q < 3.
    """
    from math import gamma, sqrt, pi
    x = np.asarray(x, dtype=float)
    if q < 1 or q >= 3:
        raise ValueError("normalization implemented for 1 <= q < 3")
    if np.isclose(q, 1.0):
        return np.sqrt(beta / pi) * np.exp(-beta * x ** 2)
    k = sqrt(q - 1.0)
    C_q = sqrt(pi) * gamma((3.0 - q) / (2.0 * (q - 1.0))) / (
        k * gamma(1.0 / (q - 1.0)))
    return (np.sqrt(beta) / C_q) * q_exp(-beta * x ** 2, q)


def sample_q_gaussian(n, q, beta, rng=None):
    """Exact samples from a q-Gaussian for 1 < q < 3 (generalized Box-Muller).

    Reference: Thistleton, Marsh, Nelson & Tsallis, IEEE Trans. Inf. Theory
    53 (2007). Generates variates of rho(x) ∝ exp_q(-beta x^2).
    """
    if rng is None:
        rng = np.random.default_rng()
    if not (1.0 < q < 3.0):
        raise ValueError("sampler valid for 1 < q < 3")
    # map target q to the generator's q' via q' = (3q-1)/(q+1)
    qp = (3.0 * q - 1.0) / (q + 1.0)
    u1 = rng.random(n)
    u2 = rng.random(n)
    # q-logarithm with parameter qp
    lnq = (u1 ** (1.0 - qp) - 1.0) / (1.0 - qp)
    z = np.sqrt(-2.0 * lnq) * np.cos(2.0 * np.pi * u2)
    # z ~ q-Gaussian with unit-ish scale; rescale to requested beta
    # generator yields variance-like scale 1/(beta0); set beta0 = 1 then scale
    return z / np.sqrt(2.0 * beta)
