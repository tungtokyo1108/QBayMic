# QBayMic — Quantum Bayesian clustering of microbiome count data

QBayMic fits a **Dirichlet–multinomial mixture with stochastic variable
selection** (DMM-SVVS) to microbiome count data, and lets you run the E-step with
either a **classical** or a **quantum** engine. A barrier diagnostic, the *signal
fraction* **σ**, predicts *before* clustering whether the quantum E-step will
provide an advantage.

All methods share one variational objective and one scikit-learn-style
interface — they differ only in how the E-step responsibilities are computed.

| method       | E-step                                        | type                |
|--------------|-----------------------------------------------|---------------------|
| `greedy_vb`  | coordinate-ascent variational Bayes           | classical           |
| `pt`         | parallel-tempering VB (multi-temperature)     | classical           |
| `davb`       | deterministic annealing VB (`s0=0` control)   | classical control   |
| `ed`         | exact quantum Gibbs E-step (diagonalisation)  | quantum (exact)     |
| `varqite`    | variational imaginary-time evolution          | quantum (circuit)   |
| `vqt`        | variational quantum thermalizer               | quantum (circuit)   |

## Install

```bash
git clone <this-repo> && cd QBayMic-main
pip install -r requirements.txt
```

The classical methods (`greedy_vb`, `pt`) need only NumPy/SciPy/scikit-learn.
The quantum methods (`ed`, `varqite`, `vqt`) additionally use JAX and PennyLane
(CPU builds are sufficient — **no GPU required**).

## Quickstart

```python
from qbaymic import QBayMic, make_counts, signal_fraction, regime
from sklearn.metrics import adjusted_rand_score

# Synthetic microbiome counts in the advantage band (σ ≈ 0.29)
X, y = make_counts(n_samples=400, n_taxa=5000, n_clusters=3,
                   separation=0.2, zero_inflation=0.80, seed=0)

# The recoverability diagnostic — no clustering run needed
sig = signal_fraction(N=400, S=5000, K=3, separation=0.2, zero_inflation=0.80)
print(sig["sigma"], regime(sig["sigma"]))          # ≈ 0.29, "advantage band"

# Fit the exact quantum E-step and score it
model = QBayMic(method="vqt", K_max=4, random_state=0).fit(X)
print(model.K_, adjusted_rand_score(y, model.labels_))
```

`X` is an integer count matrix of shape `(n_samples, n_taxa)`. `K_max`
over-specifies the number of clusters; the effective `K` is inferred by pruning.

## The signal fraction σ

σ = B_signal / B ∈ [0, 1] orders clustering **difficulty**, where `B` is the
between-cluster barrier height and `B_signal` is the part due to genuine
separation (rather than finite-sample noise). It predicts the regime:

| σ            | regime            | behaviour                                    |
|--------------|-------------------|----------------------------------------------|
| ≲ 0.20       | unrecoverable     | no method succeeds                           |
| ≈ 0.25–0.45  | **advantage band**| the quantum E-step separates from classical  |
| ≳ 0.45       | signal-dominated  | all methods succeed (small quantum margin)   |

- `signal_fraction(...)` — **exact** σ for a synthetic configuration.
- `signal_fraction_estimate(X, labels)` — σ **estimated from real data** and a
  (possibly approximate) partition, by debiasing the empirical barrier with a
  within-group split-half sampling null. Use a fast partition (k-means on
  relative abundances, or one `greedy_vb` pass) when true labels are unknown.

## Examples

```bash
python examples/01_quickstart.py          # data → σ → fit → ARI
python examples/02_quantum_vs_classical.py # all six methods on one instance
python examples/03_sigma_diagnostic.py     # σ on real (unlabelled) data
```

## Tests

```bash
python -m pytest tests/          # or: python tests/test_smoke.py
```

The smoke tests fit every method on a tiny instance and exercise the σ
diagnostic on both paths; they run in well under a minute.

## Choosing hyperparameters

`QBayMic(...)` ships with the paper's shared reference configuration. To tune per
dataset, the validated random-search routines are vendored under
`qbaymic/_engines/` (`random_search_*`); a fair best-vs-best protocol gives each
method its own `K`-aware search.

## Repository layout

```
qbaymic/
  __init__.py        public API
  models.py          the unified QBayMic estimator (thin wrappers)
  barrier.py         the σ diagnostic (synthetic + real-data paths)
  data.py            the synthetic count generator
  _engines/          the validated method implementations (used as-is)
examples/            runnable scripts
tests/               CI smoke tests
```

The modules under `qbaymic/_engines/` are the exact implementations used in the
paper; the top-level package wraps them behind a clean, documented API.

## Citation

If you use QBayMic, please cite the accompanying paper (see `CITATION`/preprint).

## License

See `LICENSE`.
