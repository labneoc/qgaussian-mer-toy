# -----------------------------------------------------------------------------
# Prof. Dr. Bruno Duarte Gomes
# Laboratorio de Neurofisiologia Eduardo Oswaldo Cruz - NEXUS
# Laboratorio de Simulacao e Biologia Computacional
# Universidade Federal do Para
# -----------------------------------------------------------------------------
# demo.py
#   End-to-end mini-demo (seconds, no heavy compute):
#     1) simulate a MER-like series with a slowly fluctuating variance
#        (superstatistics)  ->  2) z-score it  ->  3) recover (q, beta) by the
#        curve-straightening grid fit  ->  4) plot histogram + fitted q-Gaussian
#        and the slow-variance envelope.
# -----------------------------------------------------------------------------
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulate_series import simulate, zscore, excess_kurtosis
from fit_q import amplitude_histogram, fit_q_grid
from qgaussian import q_gaussian_pdf
from _signature import SIGNATURE

NAVY, TEAL, AMBER = "#0B2545", "#0E7C7B", "#E09F3E"


def main(seed=7):
    x = simulate(n=40000, mechanism="superstat", tau=600,
                 sigma_log=0.6, seed=seed)
    xz = zscore(x)
    res = fit_q_grid(xz)
    print(f"excess kurtosis = {excess_kurtosis(xz):+.2f}  "
          f"->  recovered q = {res['q']:.2f}, beta = {res['beta']:.2f}, "
          f"R^2 = {res['r2']:.3f}")

    cx, cy = amplitude_histogram(xz)
    xs = np.linspace(-6, 6, 400)
    yhat = q_gaussian_pdf(xs, res["q"], res["beta"])
    gauss = np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].plot(xz[:4000], color=NAVY, lw=0.5)
    ax[0].set_title("Serie sintetica tipo-MER (z-score)", color=NAVY)
    ax[0].set_xlabel("amostra"); ax[0].set_ylabel("amplitude")

    ax[1].semilogy(cx, cy, "o", ms=3, color=TEAL, label="histograma")
    ax[1].semilogy(xs, yhat, "-", color=AMBER, lw=2,
                   label=f"q-Gaussiana q={res['q']:.2f}")
    ax[1].semilogy(xs, gauss, "--", color="0.5", lw=1, label="Gaussiana (q=1)")
    ax[1].set_ylim(1e-4, 1); ax[1].set_xlim(-6, 6)
    ax[1].set_title("Caudas pesadas: q > 1", color=NAVY)
    ax[1].set_xlabel("amplitude"); ax[1].set_ylabel("densidade (log)")
    ax[1].legend(frameon=False, fontsize=8)

    fig.suptitle("q-Gaussian MER toy  —  " + SIGNATURE, fontsize=7, y=0.02,
                 color="0.4")
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    figdir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
    os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "demo.png")
    fig.savefig(out, dpi=130)
    print(f"figura salva em: {out}")


if __name__ == "__main__":
    main()
