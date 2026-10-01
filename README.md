# q-Gaussian MER toy

**Companion code — SBNeC 2026 talk / manuscript _"Deep brain microelectrode signal: a q-statistical approach"_ (Tavares, Lima, Martins Neto, Gomes & Tsallis; submitted to _Chaos, Solitons & Fractals_).**

A deliberately small, readable sandbox that reproduces — *in caricature* — the single idea at the core of the study: a brain microelectrode signal is a stationary-looking stream of amplitudes whose histogram is **not** Gaussian but **q-Gaussian** (heavy-tailed, `q > 1`), and that heavy tail can be *generated* by a variance that drifts slowly in time (**superstatistics**).

Everything here runs in **seconds on a laptop**, uses only `numpy` + `matplotlib`, and every statistic is a **knob you can edit**. This is a teaching artifact — it is **not** the pipeline applied to the real recordings.

---

## The idea in one paragraph

A q-Gaussian is the distribution that maximizes the Tsallis entropy *S_q*:

```
rho(x)  ∝  [ 1 + beta (q - 1) x^2 ]^(-1/(q-1))  =  sqrt(beta)/C_q · exp_q(-beta x^2)
```

As `q -> 1` it collapses to the ordinary Gaussian. For `1 < q < 3` it has power-law tails. Here `q` is a **shape** parameter (tail heaviness) and `beta` is a **scale** parameter. The toy lets you (1) *generate* a series with a chosen `(q, beta)` or with a slowly fluctuating variance, and (2) *recover* `q` back from the amplitude histogram the way the paper does — by finding the `q` that "straightens" the curve.

---

## Install & run

```bash
git clone <this-repo>
cd qgaussian-mer-toy
pip install -r requirements.txt

# one-shot end-to-end demo (simulate -> z-score -> fit q -> plot)
python src/demo.py
```

Expected console output (seed-dependent):

```
excess kurtosis = +1.06  ->  recovered q = 1.21, beta = 0.73, R^2 = 0.999
figura salva em: .../figures/demo.png
```

`figures/demo.png` shows, left, a MER-like z-scored trace, and right, its amplitude histogram (log scale) with the fitted q-Gaussian over a reference Gaussian — the heavy tail is the whole point.

---

## What each file does

| File | Role |
|------|------|
| `src/qgaussian.py` | q-exponential, normalized q-Gaussian pdf, and an **exact q-Gaussian sampler** (generalized Box–Muller; Thistleton et al. 2007). |
| `src/simulate_series.py` | Generates a MER-like series with **known, editable** statistics. Two mechanisms: direct `qgaussian` draws, or `superstat` (a Gaussian whose local variance drifts slowly — heavy tails emerge on their own). |
| `src/fit_q.py` | Recovers `(q, beta)` from a series by **grid search on R²** against the q-Gaussian — the same estimator family as the paper, including its flaws (grid bounds, R²-maximization). Flags optima pinned to a grid edge. |
| `src/demo.py` | Ties it all together and saves a figure. |
| `src/_signature.py` | Authorship string, imported by the demo. |

---

## The knobs (edit these)

In `simulate_series.simulate(...)`:

- `mechanism="superstat"` — slow-variance mixing (the physical mechanism argued in the paper) — or `"qgaussian"` for direct draws.
- `q`, `beta` — target shape and scale (`mechanism="qgaussian"`).
- `tau` — how *slowly* the local variance drifts (larger = slower; `superstat`).
- `sigma_log` — spread of the log-variance; **larger = heavier tails = larger effective `q`** (`superstat`).
- `n`, `seed` — length and reproducibility.

Try it:

```python
from src.simulate_series import simulate, zscore
from src.fit_q import fit_q_grid

# heavier drift -> larger recovered q
x = simulate(n=40000, mechanism="superstat", tau=600, sigma_log=0.9, seed=1)
print(fit_q_grid(zscore(x)))   # watch q climb as sigma_log grows
```

---

## How this mirrors the study (and how it does *not*)

**Mirrors:** amplitude distribution → q-Gaussian fit → `q > 1`; z-score normalization as a preprocessing step; superstatistics as a tail-generating mechanism; a histogram-based `q` estimator with explicit grid bounds and a boundary flag.

**Does *not* mirror (honest caveats):**

- The real study fits **184 recordings from 46 Parkinson's patients**, inside and outside the subthalamic nucleus. This toy fits one synthetic trace at a time.
- The headline result of the manuscript — the **`q(beta)` collapse onto a single curve** `q = 3 − 1.85 beta^(−0.33)` — is a *population* property across recordings; it is **not** reproduced here and should not be read off a single toy run.
- **z-scoring imposes its own `q(beta)` constraint.** For `1 < q < 5/3` a unit-variance q-Gaussian satisfies `q = 5/3 − 1/(3 beta)` exactly. Disentangling that identity from a genuine emergent constraint is a central methodological question in the paper — and is *outside* the scope of this toy.
- The grid-R² estimator here is **intentionally simple**. For real inference prefer maximum likelihood / q-CLT methods with confidence intervals, and model selection (AIC/BIC) against competing heavy-tailed families (Student-*t* — formally a q-Gaussian — α-stable, lognormal).

Use this to build intuition, not to draw scientific conclusions.

---

## Citation

If this code helps a class or a talk, please point to the manuscript:

> Tavares, A. L. S.; Lima, H. S.; Martins Neto, A. P.; Gomes, B. D.; Tsallis, C.
> *Deep brain microelectrode signal: a q-statistical approach* (submitted, _Chaos, Solitons & Fractals_).

---

## Authorship

All code authored and signed by:

**Prof. Dr. Bruno Duarte Gomes**
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz — **NEXUS**
Laboratório de Simulação e Biologia Computacional
**Universidade Federal do Pará**

Educational companion material. No patient data are included in this repository.
